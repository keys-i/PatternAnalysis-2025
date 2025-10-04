#!/usr/bin/env python3
"""
Sample synthetic sequences using a trained TimeGAN model and visualise results.

This module loads a saved checkpoint, generates synthetic limit order book
windows, prints summary statistics, and produces basic visualisations
(e.g., feature lines and depth heatmaps) to compare real vs. synthetic data.

Typical Usage:
    # Using preprocessed windows
    python sample_viz.py --npz ./preproc_final/windows.npz \
        --ckpt ./ckpts/timegan_run/best.pt --z-dim 24 --h-dim 64

    # Preprocess on-the-fly (same flags as dataset.py)
    python sample_viz.py --data-dir /PATH/TO/SESSION --feature-set core \
        --seq-len 128 --stride 32 --scaler robust --whiten pca --pca-var 0.999 \
        --ckpt ./ckpts/timegan_run/best.pt --z-dim 24 --h-dim 64

Created By: Radhesh Goel (Keys-I)
ID: s49088276
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

import torch

# local modules
from modules import TimeGAN, sample_noise
from dataset import LOBSTERData


# ---------------------------
# Data loading helpers
# ---------------------------

def load_windows_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(npz_path)
    return d["train"], d["val"], d["test"]

def load_windows_csv(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        whiten=args.whiten, pca_var=args.pca_var,
        aug_prob=0.0,  # no aug for visualisation builds
        save_dir=None,
    )
    return ds.load_arrays()


# ---------------------------
# Model restore + sampling
# ---------------------------

def build_model_from_ckpt(ckpt_path: str, x_dim: int, z_dim: int, h_dim: int, device: torch.device) -> TimeGAN:
    ckpt = torch.load(ckpt_path, map_location=device)
    args_in_ckpt = ckpt.get("args", {}) or {}
    rnn_type  = args_in_ckpt.get("rnn_type", "gru")
    enc_layers = int(args_in_ckpt.get("enc_layers", 2))
    dec_layers = int(args_in_ckpt.get("dec_layers", 2))
    gen_layers = int(args_in_ckpt.get("gen_layers", 2))
    sup_layers = int(args_in_ckpt.get("sup_layers", 1))
    dis_layers = int(args_in_ckpt.get("dis_layers", 1))
    dropout   = float(args_in_ckpt.get("dropout", 0.1))

    model = TimeGAN(
        x_dim=x_dim, z_dim=z_dim, h_dim=h_dim,
        rnn_type=rnn_type, enc_layers=enc_layers, dec_layers=dec_layers,
        gen_layers=gen_layers, sup_layers=sup_layers, dis_layers=dis_layers,
        dropout=dropout
    ).to(device)

    model.embedder.load_state_dict(ckpt["embedder"])
    model.recovery.load_state_dict(ckpt["recovery"])
    model.generator.load_state_dict(ckpt["generator"])
    model.supervisor.load_state_dict(ckpt["supervisor"])
    model.discriminator.load_state_dict(ckpt["discriminator"])
    model.eval()
    return model

@torch.no_grad()
def sample_synthetic(model: TimeGAN, n_seq: int, seq_len: int, z_dim: int, device: torch.device) -> np.ndarray:
    z = sample_noise(n_seq, seq_len, z_dim, device)
    e_tilde = model.generator(z)
    h_tilde = model.supervisor(e_tilde)
    x_tilde = model.recovery(h_tilde)
    return x_tilde.detach().cpu().numpy()


# ---------------------------
# Stats + simple similarity
# ---------------------------

def summarize(name: str, W: np.ndarray) -> dict:
    # mean/std over batch+time, per-feature
    mu = W.mean(axis=(0, 1))
    sd = W.std(axis=(0, 1))
    return {"name": name, "mean": mu, "std": sd}

def kl_hist_avg(real: np.ndarray, synth: np.ndarray, bins: int = 64, eps: float = 1e-9) -> float:
    """
    Quick histogram-based KL(real || synth) averaged over features.
    """
    from scipy.special import rel_entr
    F = real.shape[2]
    vals = []
    R = real.reshape(-1, F)
    S = synth.reshape(-1, F)
    for f in range(F):
        r = R[:, f]; s = S[:, f]
        lo = np.nanpercentile(np.concatenate([r, s]), 0.5)
        hi = np.nanpercentile(np.concatenate([r, s]), 99.5)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        pr, _ = np.histogram(r, bins=bins, range=(lo, hi), density=True)
        ps, _ = np.histogram(s, bins=bins, range=(lo, hi), density=True)
        pr = pr + eps; ps = ps + eps
        pr = pr / pr.sum(); ps = ps / ps.sum()
        vals.append(np.sum(rel_entr(pr, ps)))
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------
# Visualisations
# ---------------------------

def plot_feature_lines(real: np.ndarray, synth: np.ndarray, outdir: str, max_feats: int = 4, idx: int = 0):
    """
    Plot a few feature time-series (same sequence index) real vs synthetic.
    """
    os.makedirs(outdir, exist_ok=True)
    T, F = real.shape[1], real.shape[2]
    feats = min(F, max_feats)

    fig, axes = plt.subplots(feats, 1, figsize=(10, 2.2 * feats), sharex=True)
    if feats == 1:
        axes = [axes]
    for i in range(feats):
        axes[i].plot(real[idx, :, i], label="real", linewidth=1.2)
        axes[i].plot(synth[idx, :, i], label="synthetic", linewidth=1.2, linestyle="--")
        axes[i].set_ylabel(f"feat {i}")
    axes[-1].set_xlabel("time")
    axes[0].legend(loc="upper right")
    fig.suptitle("Feature lines: real vs synthetic")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "feature_lines.png"), dpi=150)
    plt.close(fig)

def plot_heatmaps(real: np.ndarray, synth: np.ndarray, outdir: str, idx: int = 0):
    """
    Plot depth heatmaps (time x features) for a single sequence.
    """
    os.makedirs(outdir, exist_ok=True)
    a = real[idx]; b = synth[idx]
    # normalize each to [0,1] for visibility
    def norm01(x):
        lo, hi = np.percentile(x, 1), np.percentile(x, 99)
        return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

    a = norm01(a); b = norm01(b)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    im0 = axes[0].imshow(a, aspect="auto", origin="lower")
    axes[0].set_title("Real (heatmap)")
    axes[0].set_xlabel("feature"); axes[0].set_ylabel("time")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(b, aspect="auto", origin="lower")
    axes[1].set_title("Synthetic (heatmap)")
    axes[1].set_xlabel("feature"); axes[1].set_ylabel("time")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "heatmaps.png"), dpi=150)
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sample & visualise TimeGAN outputs vs real.")
    # data
    ap.add_argument("--npz", type=str, help="Path to windows.npz (train/val/test). If set, ignores --data-dir.")
    ap.add_argument("--data-dir", type=str, help="Folder with message_10.csv and orderbook_10.csv")
    ap.add_argument("--message", default="message_10.csv")
    ap.add_argument("--orderbook", default="orderbook_10.csv")
    ap.add_argument("--feature-set", choices=["core","raw10"], default="core")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--splits", type=float, nargs=3, default=(0.7,0.15,0.15))
    ap.add_argument("--scaler", choices=["standard","minmax","robust","quantile","power","none"], default="robust")
    ap.add_argument("--whiten", choices=["pca","zca",None], default="pca")
    ap.add_argument("--pca-var", type=float, default=0.999)
    ap.add_argument("--headerless-message", action="store_true")
    ap.add_argument("--headerless-orderbook", action="store_true")

    # model restore
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--z-dim", type=int, required=True)
    ap.add_argument("--h-dim", type=int, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # viz
    ap.add_argument("--n-synth", type=int, default=128, help="How many synthetic windows to sample.")
    ap.add_argument("--seq-index", type=int, default=0, help="Which sequence index to plot.")
    ap.add_argument("--max-feats", type=int, default=4, help="Max features to show in line plot.")
    ap.add_argument("--outdir", type=str, default="./viz_out")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)

    # Load real windows
    if args.npz:
        Wtr, Wval, Wte = load_windows_npz(args.npz)
    elif args.data_dir:
        Wtr, Wval, Wte = load_windows_csv(args)
    else:
        raise SystemExit("Provide either --npz or --data-dir")

    # Pick a real reference set (test split)
    real = Wte
    _, T, D = real.shape

    # Build model & restore
    model = build_model_from_ckpt(args.ckpt, x_dim=D, z_dim=args.z_dim, h_dim=args.h_dim, device=device)
    model.eval()

    # Sample synthetic
    n_synth = min(args.n_synth, len(real))
    synth = sample_synthetic(model, n_synth, T, args.z_dim, device)

    # Basic stats
    s_real = summarize("real(test)", real)
    s_synth = summarize("synthetic", synth)
    print("=== Summary (per-feature mean/std) ===")
    print(f"{s_real['name']}: mean[0:5]={s_real['mean'][:5]}, std[0:5]={s_real['std'][:5]}")
    print(f"{s_synth['name']}: mean[0:5]={s_synth['mean'][:5]}, std[0:5]={s_synth['std'][:5]}")

    # Quick KL(hist) similarity
    try:
        kl = kl_hist_avg(real[:n_synth], synth)
        print(f"KL(real || synth) ~ {kl:.4f} (lower is better)")
    except Exception as e:
        print(f"KL computation skipped: {e}")

    # Visualisations
    idx = max(0, min(args.seq_index, n_synth - 1))
    plot_feature_lines(real, synth, args.outdir, max_feats=args.max_feats, idx=idx)
    plot_heatmaps(real, synth, args.outdir, idx=idx)

    print(f"Saved plots to: {args.outdir}")
