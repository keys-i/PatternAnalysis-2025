"""
Train, validate, and test the TimeLOB TimeGAN on LOBSTER sequences.

This module orchestrates the three-phase TimeGAN schedule (autoencoder
pretrain, supervisor pretrain, joint adversarial training), log losses,
computes validation metrics (e.g., KL on spread/returns; SSIM on heatmaps),
and saves model checkpoints and plots. The model is imported from ``modules.py``
and data loaders from ``dataset.py``.

Typical Usage:
    python3 -m predict --ckpt checkpoints/best.pt --n 8 --seq_len 120 --out outputs/predictions 

Created By: Radhesh Goel (Keys-I)
ID: s49088276

References:
- 
"""
from __future__ import annotations
import os, json, math, time, argparse, random
from dataclasses import asdict
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# local imports
from dataset import LOBSTERData
from modules import (
    TimeGAN, sample_noise, make_optim,
    timegan_autoencoder_step, timegan_supervisor_step, timegan_joint_step,
    LossWeights
)

# -------------------------
# utils
# -------------------------
def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def shape_from_npz(npz_path: str) -> Tuple[int,int,int]:
    d = np.load(npz_path)
    w = d["train"]
    return tuple(w.shape)  # num_seq, seq_len, x_dim

def build_loaders_from_npz(npz_path: str, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    d = np.load(npz_path)
    W_train = torch.from_numpy(d["train"]).float()
    W_val   = torch.from_numpy(d["val"]).float()
    W_test  = torch.from_numpy(d["test"]).float()
    T = W_train.size(1); D = W_train.size(2)
    train_dl = DataLoader(TensorDataset(W_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(TensorDataset(W_val),   batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(TensorDataset(W_test),  batch_size=batch_size, shuffle=False)
    return train_dl, val_dl, test_dl, T, D

def build_loaders_from_csv(args, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    ds = LOBSTERData(
        data_dir=args.data_dir,
        message_file=args.message,
        orderbook_file=args.orderbook,
        feature_set=args.feature_set,
        seq_len=args.seq_len,
        stride=args.stride,
        splits=tuple(args.splits),
        scaler=args.scaler,
        headerless_message=args.headerless_message,
        headerless_orderbook=args.headerless_orderbook,
        # optional whitening & aug flags if you want them in training too:
        whiten=args.whiten, pca_var=args.pca_var,
        aug_prob=args.aug_prob, aug_jitter_std=args.aug_jitter_std,
        aug_scaling_std=args.aug_scaling_std, aug_timewarp_max=args.aug_timewarp_max,
        save_dir=args.save_dir,
    )
    W_train, W_val, W_test = ds.load_arrays()
    T = W_train.shape[1]; D = W_train.shape[2]
    train_dl = DataLoader(TensorDataset(torch.from_numpy(W_train).float()), batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(TensorDataset(torch.from_numpy(W_val).float()),   batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(TensorDataset(torch.from_numpy(W_test).float()),  batch_size=batch_size, shuffle=False)
    # Persist meta if saving:
    if args.save_dir:
        meta = ds.get_meta()
        with open(os.path.join(args.save_dir, "meta.train.json"), "w") as f:
            json.dump(meta, f, indent=2)
    return train_dl, val_dl, test_dl, T, D

def save_ckpt(path: str, model: TimeGAN, opt_gs, opt_d, step: int, args, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "step": step,
        "args": vars(args),
        "embedder": model.embedder.state_dict(),
        "recovery": model.recovery.state_dict(),
        "generator": model.generator.state_dict(),
        "supervisor": model.supervisor.state_dict(),
        "discriminator": model.discriminator.state_dict(),
        "opt_gs": opt_gs.state_dict(),
        "opt_d": opt_d.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)

# -------------------------
# train loops
# -------------------------
def run_autoencoder_phase(model, train_dl, device, opt_gs, epochs: int, amp: bool, clip: Optional[float]):
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    for ep in range(1, epochs+1):
        t0 = time.time()
        logs = []
        for (xb,) in train_dl:
            xb = xb.to(device, non_blocking=True)
            opt_gs.zero_grad(set_to_none=True)
            if amp:
                with torch.amp.autocast('cuda'):
                    out = timegan_autoencoder_step(model, xb, opt_gs)
            else:
                out = timegan_autoencoder_step(model, xb, opt_gs)
            # timegan_autoencoder_step already steps opt; clip if needed
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.embedder.parameters(), clip)
                torch.nn.utils.clip_grad_norm_(model.recovery.parameters(), clip)
            logs.append(out["recon"])
        dt = time.time()-t0
        print(f"[AE] epoch {ep}/{epochs} recon={np.mean(logs):.6f} ({dt:.1f}s)")

def run_supervisor_phase(model, train_dl, device, opt_gs, epochs: int, amp: bool, clip: Optional[float]):
    for ep in range(1, epochs+1):
        t0 = time.time()
        logs = []
        for (xb,) in train_dl:
            xb = xb.to(device, non_blocking=True)
            out = timegan_supervisor_step(model, xb, opt_gs)
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.supervisor.parameters(), clip)
            logs.append(out["sup"])
        dt = time.time()-t0
        print(f"[SUP] epoch {ep}/{epochs} sup={np.mean(logs):.6f} ({dt:.1f}s)")

def evaluate_moment(model, loader, device, z_dim: int) -> float:
    # rough eval: moment loss on validation set (lower is better)
    from modules import moment_loss
    model.eval()
    vals = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            z = sample_noise(xb.size(0), xb.size(1), z_dim, device)
            # generate one batch
            paths = model.forward_gen_paths(xb, z)
            x_tilde = paths["X_tilde"]
            vals.append(float(moment_loss(xb, x_tilde).cpu()))
    return float(np.mean(vals)) if vals else math.inf

def run_joint_phase(model, train_dl, val_dl, device, opt_gs, opt_d,
                    z_dim: int, epochs: int, amp: bool, clip: Optional[float],
                    loss_weights: LossWeights, ckpt_dir: Optional[str], args=None):
    best_val = math.inf
    step = 0
    for ep in range(1, epochs+1):
        t0 = time.time()
        logs = {"d": [], "g_adv": [], "g_sup": [], "g_mom": [], "g_fm": [], "recon": [], "cons": [], "g_total": []}
        for (xb,) in train_dl:
            xb = xb.to(device, non_blocking=True)
            z  = sample_noise(xb.size(0), xb.size(1), z_dim, device)
            out = timegan_joint_step(model, xb, z, opt_gs, opt_d, loss_weights)
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(list(model.embedder.parameters())+
                                               list(model.recovery.parameters())+
                                               list(model.generator.parameters())+
                                               list(model.supervisor.parameters()), clip)
                torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), clip)
            for k, v in out.items(): logs[k].append(v)
            step += 1

        # validation (moment)
        val_m = evaluate_moment(model, val_dl, device, z_dim)
        dt = time.time()-t0
        log_line = " ".join([f"{k}={np.mean(v):.4f}" for k,v in logs.items()])
        print(f"[JOINT] epoch {ep}/{epochs} {log_line} | val_moment={val_m:.4f} ({dt:.1f}s)")

        # save best
        if ckpt_dir:
            if val_m < best_val:
                best_val = val_m
                save_ckpt(os.path.join(ckpt_dir, "best.pt"), model, opt_gs, opt_d, step, args=args,
                        extra={"val_moment": val_m})
            save_ckpt(os.path.join(ckpt_dir, f"step_{step}.pt"), model, opt_gs, opt_d, step, args=args,
                    extra={"val_moment": val_m})

# -------------------------
# main
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train TimeGAN on LOBSTERData.")
    # data sources
    p.add_argument("--npz", type=str, help="Path to windows.npz (train/val/test). If set, ignores --data-dir.")
    p.add_argument("--data-dir", type=str, help="Folder with message_10.csv and orderbook_10.csv")
    p.add_argument("--message", default="message_10.csv")
    p.add_argument("--orderbook", default="orderbook_10.csv")
    p.add_argument("--feature-set", choices=["core","raw10"], default="core")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--splits", type=float, nargs=3, default=(0.7,0.15,0.15))
    p.add_argument("--scaler", choices=["standard","minmax","robust","quantile","power","none"], default="robust")
    p.add_argument("--whiten", choices=["pca","zca",None], default="pca")
    p.add_argument("--pca-var", type=float, default=0.999)
    p.add_argument("--headerless-message", action="store_true")
    p.add_argument("--headerless-orderbook", action="store_true")
    p.add_argument("--save-dir", type=str, default=None, help="If set during CSV mode, saves NPZ/meta here.")

    # model
    p.add_argument("--x-dim", type=str, default="auto", help="'auto' infers from data; else int")
    p.add_argument("--z-dim", type=int, default=24)
    p.add_argument("--h-dim", type=int, default=64)
    p.add_argument("--rnn-type", choices=["gru","lstm"], default="gru")
    p.add_argument("--enc-layers", type=int, default=2)
    p.add_argument("--dec-layers", type=int, default=2)
    p.add_argument("--gen-layers", type=int, default=2)
    p.add_argument("--sup-layers", type=int, default=1)
    p.add_argument("--dis-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)

    # training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    p.add_argument("--clip", type=float, default=1.0, help="Grad clip norm; set <=0 to disable.")
    p.add_argument("--ae-epochs", type=int, default=10)
    p.add_argument("--sup-epochs", type=int, default=10)
    p.add_argument("--joint-epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ckpt-dir", type=str, default="./ckpts")

    # augmentation passthrough when using CSV mode
    p.add_argument("--aug-prob", type=float, default=0.0)
    p.add_argument("--aug-jitter-std", type=float, default=0.01)
    p.add_argument("--aug-scaling-std", type=float, default=0.05)
    p.add_argument("--aug-timewarp-max", type=float, default=0.1)

    args = p.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    run_dir = os.path.join(args.ckpt_dir, f"timegan_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    # Data
    if args.npz:
        train_dl, val_dl, test_dl, T, D = build_loaders_from_npz(args.npz, args.batch_size)
    elif args.data_dir:
        train_dl, val_dl, test_dl, T, D = build_loaders_from_csv(args, args.batch_size)
    else:
        raise SystemExit("Provide either --npz or --data-dir")

    x_dim = D if args.x_dim == "auto" else int(args.x_dim)

    # Model & optims
    model = TimeGAN(
        x_dim=x_dim, z_dim=args.z_dim, h_dim=args.h_dim,
        rnn_type=args.rnn_type, enc_layers=args.enc_layers, dec_layers=args.dec_layers,
        gen_layers=args.gen_layers, sup_layers=args.sup_layers, dis_layers=args.dis_layers,
        dropout=args.dropout
    ).to(device)

    opt_gs = make_optim(list(model.embedder.parameters()) +
                        list(model.recovery.parameters()) +
                        list(model.generator.parameters()) +
                        list(model.supervisor.parameters()), lr=args.lr)
    opt_d  = make_optim(model.discriminator.parameters(), lr=args.lr)

    # Phase 1: autoencoder pretrain
    if args.ae_epochs > 0:
        run_autoencoder_phase(model, train_dl, device, opt_gs, args.ae_epochs, amp=args.amp, clip=args.clip if args.clip>0 else None)
        save_ckpt(os.path.join(run_dir, "after_autoencoder.pt"), model, opt_gs, opt_d, step=0, args=args)

    # Phase 2: supervisor pretrain
    if args.sup_epochs > 0:
        run_supervisor_phase(model, train_dl, device, opt_gs, args.sup_epochs, amp=args.amp, clip=args.clip if args.clip>0 else None)
        save_ckpt(os.path.join(run_dir, "after_supervisor.pt"), model, opt_gs, opt_d, step=0, args=args)

    # Phase 3: joint training
    if args.joint_epochs > 0:
        run_joint_phase(
            model, train_dl, val_dl, device, opt_gs, opt_d,
            z_dim=args.z_dim, epochs=args.joint_epochs, amp=args.amp,
            clip=args.clip if args.clip>0 else None,
            loss_weights=LossWeights(), ckpt_dir=run_dir, args=args
        )


    # Final test moment score
    test_m = evaluate_moment(model, test_dl, device, args.z_dim)
    print(f"[DONE] test moment loss: {test_m:.6f}")

