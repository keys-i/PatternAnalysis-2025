"""
Define the core TimeGAN components for limit order book sequences.

This module declares the building blocks of the TimeGAN adapted to LOBSTER
level-10 order book data (e.g., AMZN). It typically includes the Embedder,
Recovery, Generator, Supervisor, and Discriminator, and a TimeGAN wrapper that
wires them together. Inputs are sequences shaped
``(batch_size, seq_len, feature_dim)`` and outputs mirror that shape.

Exports:
    - Embedder
    - Recovery
    - Generator
    - Supervisor
    - Discriminator
    - TimeGAN

Created By: Radhesh Goel (Keys-I)
ID: s49088276

References:
- 
"""
# modules.py
# Basic TimeGAN components implemented in PyTorch
# ------------------------------------------------
# Components:
#  - Embedder (encoder)     : X -> H
#  - Recovery (decoder)     : H -> X_hat
#  - Generator              : Z -> E_tilde (latent)
#  - Supervisor             : H -> H_hat (one-step future)
#  - Discriminator          : {H, H_tilde} -> real/fake logit
# Wrapper:
#  - TimeGAN                : convenience forward helpers
# Losses:
#  - reconstruction_loss, supervised_loss, generator_adv_loss,
#    discriminator_loss, moment_loss, generator_feature_matching_loss
# Utils:
#  - sample_noise, init_weights, make_optim

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Small building blocks
# -------------------------

class RNNSeq(nn.Module):
    """
    Multi-layer GRU/LSTM that returns sequence outputs [B, T, H].
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        rnn_type: str = "gru",
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        assert rnn_type in {"gru", "lstm"}
        self.rnn_type = rnn_type
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        y, _ = self.rnn(x)
        return y  # [B, T, H']


def _linear_head(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
    )


def init_weights(m: nn.Module, gain: float = 1.0) -> None:
    """
    He init for Linear; orthogonal for RNN; zeros for bias.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="linear")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, (nn.GRU, nn.LSTM)):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param, gain=gain)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param, gain=gain)
            elif "bias" in name:
                nn.init.zeros_(param)


def make_optim(params, lr: float = 1e-3, betas=(0.9, 0.999), weight_decay: float = 0.0):
    return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)


# -------------------------
# TimeGAN components
# -------------------------

class Embedder(nn.Module):
    """X -> H (latent)"""
    def __init__(
        self,
        x_dim: int,
        h_dim: int,
        num_layers: int = 2,
        rnn_type: str = "gru",
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.rnn = RNNSeq(x_dim, h_dim, num_layers, rnn_type, dropout, bidirectional)
        self.proj = _linear_head(self.rnn.out_dim, h_dim)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, x_dim]
        h_seq = self.rnn(x)
        h = self.proj(h_seq)
        return h  # [B, T, h_dim]


class Recovery(nn.Module):
    """H -> X_hat (reconstruct data space)"""
    def __init__(
        self,
        h_dim: int,
        x_dim: int,
        num_layers: int = 2,
        rnn_type: str = "gru",
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.rnn = RNNSeq(h_dim, h_dim, num_layers, rnn_type, dropout, bidirectional)
        self.proj = _linear_head(self.rnn.out_dim, x_dim)
        self.apply(init_weights)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.rnn(h)
        x_hat = self.proj(z)
        return x_hat  # [B, T, x_dim]


class Generator(nn.Module):
    """Z -> E_tilde (latent space fake)"""
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        num_layers: int = 2,
        rnn_type: str = "gru",
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.rnn = RNNSeq(z_dim, h_dim, num_layers, rnn_type, dropout, bidirectional)
        self.proj = _linear_head(self.rnn.out_dim, h_dim)
        self.apply(init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        g = self.rnn(z)
        e_tilde = self.proj(g)
        return e_tilde  # [B, T, h_dim]


class Supervisor(nn.Module):
    """H -> H_hat (one-step ahead in latent)"""
    def __init__(
        self,
        h_dim: int,
        num_layers: int = 1,
        rnn_type: str = "gru",
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.rnn = RNNSeq(h_dim, h_dim, num_layers, rnn_type, dropout, bidirectional)
        self.proj = _linear_head(self.rnn.out_dim, h_dim)
        self.apply(init_weights)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        s = self.rnn(h)
        h_hat = self.proj(s)
        return h_hat  # [B, T, h_dim], meant to approximate next-step H


class Discriminator(nn.Module):
    """
    Sequence-level discriminator: encodes sequence and outputs a single real/fake logit per sequence.
    """
    def __init__(
        self,
        h_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        rnn_type: str = "gru",
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.rnn = RNNSeq(h_dim, hidden_dim, num_layers, rnn_type, dropout, bidirectional)
        rnn_out = self.rnn.out_dim
        self.head = nn.Sequential(
            nn.Linear(rnn_out, rnn_out),
            nn.ReLU(inplace=True),
            nn.Linear(rnn_out, 1),
        )
        self.apply(init_weights)

    def forward(self, h_like: torch.Tensor) -> torch.Tensor:
        # h_like: [B, T, h_dim] (real H or fake H_tilde)
        z = self.rnn(h_like)              # [B, T, H]
        pooled = z.mean(dim=1)            # [B, H] simple temporal pooling
        logit = self.head(pooled)         # [B, 1]
        return logit


# -------------------------
# TimeGAN wrapper
# -------------------------

@dataclass
class TimeGANOutputs:
    H: torch.Tensor                 # real latent from embedder
    X_tilde: torch.Tensor           # recovered from H_tilde (generator path)
    X_hat: torch.Tensor             # reconstruction of X (autoencoder path)
    H_hat_supervise: torch.Tensor   # supervisor(H)
    H_tilde: torch.Tensor           # supervisor(generator(Z))
    D_real: torch.Tensor            # discriminator(H)
    D_fake: torch.Tensor            # discriminator(H_tilde)


class TimeGAN(nn.Module):
    """
    Convenience wrapper that holds all components and exposes common forward passes.
    """
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        h_dim: int,
        rnn_type: str = "gru",
        enc_layers: int = 2,
        dec_layers: int = 2,
        gen_layers: int = 2,
        sup_layers: int = 1,
        dis_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedder   = Embedder(x_dim, h_dim, enc_layers, rnn_type, dropout)
        self.recovery   = Recovery(h_dim, x_dim, dec_layers, rnn_type, dropout)
        self.generator  = Generator(z_dim, h_dim, gen_layers, rnn_type, dropout)
        self.supervisor = Supervisor(h_dim, sup_layers, rnn_type, dropout)
        self.discriminator = Discriminator(h_dim, hidden_dim=max(64, h_dim), num_layers=dis_layers, rnn_type=rnn_type, dropout=dropout)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedder(x)

    @torch.no_grad()
    def recover(self, h: torch.Tensor) -> torch.Tensor:
        return self.recovery(h)

    def forward_all(self, x: torch.Tensor, z: torch.Tensor) -> TimeGANOutputs:
        """
        Full graph for joint training steps.
        """
        H = self.embedder(x)                      # real latent
        X_hat = self.recovery(H)                  # reconstruction

        E_tilde = self.generator(z)               # generator latent
        H_hat_supervise = self.supervisor(H)      # supervisor on real latent
        H_tilde = self.supervisor(E_tilde)        # supervised generator path

        X_tilde = self.recovery(H_tilde)          # map fake latent back to data space

        D_real = self.discriminator(H.detach())   # detach to avoid leaking gradients to embedder in D update
        D_fake = self.discriminator(H_tilde.detach())

        return TimeGANOutputs(
            H=H, X_hat=X_hat, X_tilde=X_tilde,
            H_hat_supervise=H_hat_supervise,
            H_tilde=H_tilde,
            D_real=D_real, D_fake=D_fake
        )

    # convenience for generator forward (no detach on fake for Gen loss)
    def forward_gen_paths(self, x: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        H = self.embedder(x)
        H_hat_supervise = self.supervisor(H)
        E_tilde = self.generator(z)
        H_tilde = self.supervisor(E_tilde)
        X_tilde = self.recovery(H_tilde)
        D_fake_for_gen = self.discriminator(H_tilde)  # no detach: grad goes to G/S
        return dict(H=H, H_hat_supervise=H_hat_supervise, H_tilde=H_tilde, X_tilde=X_tilde, D_fake=D_fake_for_gen)

    # convenience for autoencoder pretrain
    def forward_autoencoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H = self.embedder(x)
        X_hat = self.recovery(H)
        return H, X_hat


# -------------------------
# Losses (canonical TimeGAN style)
# -------------------------

def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    # MSE across batch, time, features
    return F.mse_loss(x_hat, x)

def supervised_loss(h: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
    """
    One-step ahead prediction in latent space:
    compare h[:, 1:, :] with h_hat[:, :-1, :].
    """
    return F.mse_loss(h_hat[:, :-1, :], h[:, 1:, :])

def discriminator_loss(d_real: torch.Tensor, d_fake: torch.Tensor, label_smooth: float = 0.1) -> torch.Tensor:
    """
    Standard non-saturating GAN BCE loss for discriminator.
    """
    # real labels in [1 - label_smooth, 1]
    real_tgt = torch.ones_like(d_real) * (1.0 - label_smooth)
    fake_tgt = torch.zeros_like(d_fake)
    loss_real = F.binary_cross_entropy_with_logits(d_real, real_tgt)
    loss_fake = F.binary_cross_entropy_with_logits(d_fake, fake_tgt)
    return loss_real + loss_fake

def generator_adv_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """
    Non-saturating generator loss (wants discriminator to output 1 for fake).
    """
    tgt = torch.ones_like(d_fake)
    return F.binary_cross_entropy_with_logits(d_fake, tgt)

def moment_loss(x: torch.Tensor, x_tilde: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Feature-wise mean/variance matching across time+batch dims.
    """
    # collapse batch/time for per-feature moments
    dim = (0, 1)
    mu_real = x.mean(dim=dim)
    mu_fake = x_tilde.mean(dim=dim)
    var_real = x.var(dim=dim, unbiased=False) + eps
    var_fake = x_tilde.var(dim=dim, unbiased=False) + eps
    return F.l1_loss(mu_fake, mu_real) + F.l1_loss(torch.sqrt(var_fake), torch.sqrt(var_real))

def generator_feature_matching_loss(h: torch.Tensor, h_tilde: torch.Tensor) -> torch.Tensor:
    """
    Optional latent-level matching (helps stability).
    """
    return F.mse_loss(h_tilde.mean(dim=(0, 1)), h.mean(dim=(0, 1)))


# -------------------------
# Noise utility
# -------------------------

def sample_noise(batch_size: int, seq_len: int, z_dim: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Standard normal noise sequence for the generator.
    """
    z = torch.randn(batch_size, seq_len, z_dim)
    return z.to(device) if device is not None else z


# -------------------------
# Minimal training scaffolds (optional)
# -------------------------

@dataclass
class LossWeights:
    lambda_embed: float = 10.0     # autoencoder recon weight during embedder pretrain
    lambda_sup: float = 1.0        # supervisor loss weight
    lambda_gen: float = 1.0        # adversarial generator weight
    lambda_moment: float = 10.0    # moment matching weight
    lambda_fm: float = 1.0         # feature/latent matching weight


def timegan_autoencoder_step(
    model: TimeGAN,
    x: torch.Tensor,
    opt: torch.optim.Optimizer,
) -> Dict[str, float]:
    """
    Pretrain the embedder+recovery (autoencoder) with reconstruction loss.
    """
    model.train()
    opt.zero_grad(set_to_none=True)
    _, x_hat = model.forward_autoencoder(x)
    loss_recon = reconstruction_loss(x, x_hat)
    loss_recon.backward()
    opt.step()
    return {"recon": float(loss_recon.detach().cpu())}


def timegan_supervisor_step(
    model: TimeGAN,
    x: torch.Tensor,
    opt: torch.optim.Optimizer,
) -> Dict[str, float]:
    """
    Pretrain the supervisor to predict next-step in latent space.
    """
    model.train()
    opt.zero_grad(set_to_none=True)
    h, _ = model.forward_autoencoder(x)
    h_hat = model.supervisor(h)
    loss_sup = supervised_loss(h, h_hat)
    loss_sup.backward()
    opt.step()
    return {"sup": float(loss_sup.detach().cpu())}


def timegan_joint_step(
    model: TimeGAN,
    x: torch.Tensor,
    z: torch.Tensor,
    opt_gs: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    weights: LossWeights = LossWeights(),
) -> Dict[str, float]:
    """
    Joint adversarial training step:
      1) Update Discriminator
      2) Update Generator + Supervisor (+ Embedder via recon & consistency)
    """
    model.train()

    # ---- 1) Discriminator update
    with torch.no_grad():
        H_real = model.embedder(x)
        E_tilde = model.generator(z)
        H_tilde = model.supervisor(E_tilde)
    D_real = model.discriminator(H_real)
    D_fake = model.discriminator(H_tilde)

    loss_d = discriminator_loss(D_real, D_fake)
    opt_d.zero_grad(set_to_none=True)
    loss_d.backward()
    opt_d.step()

    # ---- 2) Generator/Supervisor/Embedder update
    paths = model.forward_gen_paths(x, z)  # keeps gradient through G/S
    H, H_hat, H_tilde, X_tilde, D_fake_for_gen = (
        paths["H"], paths["H_hat_supervise"], paths["H_tilde"], paths["X_tilde"], paths["D_fake"]
    )

    # adversarial
    loss_g_adv = generator_adv_loss(D_fake_for_gen)
    # supervised (latent next-step)
    loss_g_sup = supervised_loss(H, H_hat)
    # moment matching in data space
    # Optionally generate X via recovery of H_tilde (already X_tilde)
    loss_g_mom = moment_loss(x, X_tilde)
    # latent feature matching
    loss_g_fm = generator_feature_matching_loss(H, H_tilde)

    # total generator loss
    loss_g_total = (
        weights.lambda_gen * loss_g_adv
        + weights.lambda_sup * loss_g_sup
        + weights.lambda_moment * loss_g_mom
        + weights.lambda_fm * loss_g_fm
    )

    # optional small reconstruction on embedder to preserve representation
    H_e, X_hat = model.forward_autoencoder(x)  # reuse embedder/recovery path
    loss_recon = reconstruction_loss(x, X_hat)
    # encourage E_tilde to be close to H via supervisor (consistency)
    loss_consistency = F.mse_loss(H_tilde, H_e).mul(0.1)  # small weight

    total = loss_g_total + loss_recon + loss_consistency

    opt_gs.zero_grad(set_to_none=True)
    total.backward()
    opt_gs.step()

    return {
        "d": float(loss_d.detach().cpu()),
        "g_adv": float(loss_g_adv.detach().cpu()),
        "g_sup": float(loss_g_sup.detach().cpu()),
        "g_mom": float(loss_g_mom.detach().cpu()),
        "g_fm": float(loss_g_fm.detach().cpu()),
        "recon": float(loss_recon.detach().cpu()),
        "cons": float(loss_consistency.detach().cpu()),
        "g_total": float(loss_g_total.detach().cpu()),
    }


# -------------------------
# Example (for reference)
# -------------------------
# if __name__ == "__main__":
#     B, T, x_dim, z_dim, h_dim = 16, 24, 8, 16, 24
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = TimeGAN(x_dim, z_dim, h_dim).to(device)
#     opt_gs = make_optim(list(model.embedder.parameters()) +
#                         list(model.recovery.parameters()) +
#                         list(model.generator.parameters()) +
#                         list(model.supervisor.parameters()), lr=1e-3)
#     opt_d = make_optim(model.discriminator.parameters(), lr=1e-3)
#     x = torch.randn(B, T, x_dim, device=device)
#     z = sample_noise(B, T, z_dim, device=device)
#     # Pretrain autoencoder
#     print(timegan_autoencoder_step(model, x, opt_gs))
#     # Pretrain supervisor
#     print(timegan_supervisor_step(model, x, opt_gs))
#     # Joint step
#     print(timegan_joint_step(model, x, z, opt_gs, opt_d))
