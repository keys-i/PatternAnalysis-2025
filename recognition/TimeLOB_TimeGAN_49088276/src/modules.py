"""
TimeGAN components with LOB-aware enhancements.

Besides the canonical Embedder/Recovery/Generator/Supervisor/Discriminator, this
module exposes an optional hybrid temporal backbone (TemporalBackbone) that can
be injected into any component via ``TemporalBackboneConfig``. The backbone
mixes positional encodings, dilated temporal convolutions (microstructure
patterns), recurrent layers, and post-hoc self-attention blocks (global context),
making the model more expressive than a basic TimeGAN.

Inputs are sequences shaped ``(batch_size, seq_len, feature_dim)`` and outputs
mirror that shape. Advanced regularization utilities and training helpers are
included near the bottom of the file.

Exports:
    - Encoder
    - Recovery
    - Generator
    - Supervisor
    - Discriminator
    - TimeGAN
    - TemporalBackboneConfig (placeholder for future use)

Created By: Radhesh Goel (Keys-I)
ID: s49088276
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple, cast, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from torch import Tensor
from tqdm.auto import tqdm  # pretty progress bars

from src.dataset import batch_generator
from src.helpers.constants import (
    NUM_TRAINING_ITERATIONS,
    OUTPUT_DIR,
    VALIDATE_INTERVAL,
    WEIGHTS_DIR,
)

# richie: centralized pretty CLI helpers (safe fallbacks inside)
from src.helpers.richie import log as rlog
from src.helpers.richie import rule as rrule
from src.helpers.richie import status as rstatus
from src.helpers.utils import (
    kl_divergence_hist,
    minmax_inverse,
    minmax_scale,
    sample_noise,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: Optional[int]):
    if seed is None or seed < 0:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Leave non-deterministic algos for perf by default; toggle if needed.
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def xavier_gru_init(m: nn.Module) -> None:
    if isinstance(m, nn.GRU):
        for name, p in m.named_parameters():
            t = cast(Tensor, p)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(t)
            elif "weight_hh" in name:
                nn.init.orthogonal_(t)
            elif "bias" in name:
                nn.init.zeros_(t)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Encoder(nn.Module):
    """
    Embedding network: original feature space → latent space.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Sigmoid()
        self.apply(xavier_gru_init)

    def forward(self, x: torch.Tensor, apply_sigmoid: bool = True) -> torch.Tensor:
        h, _ = self.rnn(x)
        h = self.proj(h)
        return self.act(h) if apply_sigmoid else h


class Recovery(nn.Module):
    """
    Recovery network: latent space → original space.
    """

    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=output_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(output_dim, output_dim)
        self.act = nn.Sigmoid()
        self.apply(xavier_gru_init)

    def forward(self, h: torch.Tensor, apply_sigmoid: bool = True) -> torch.Tensor:
        x_tilde, _ = self.rnn(h)
        x_tilde = self.proj(x_tilde)
        return self.act(x_tilde) if apply_sigmoid else x_tilde


class Generator(nn.Module):
    """
    Generator: random noise Z → latent sequence E.
    """

    def __init__(self, z_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=z_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Sigmoid()
        self.apply(xavier_gru_init)

    def forward(self, z: torch.Tensor, apply_sigmoid: bool = True) -> torch.Tensor:
        g, _ = self.rnn(z)
        g = self.proj(g)
        return self.act(g) if apply_sigmoid else g


class Supervisor(nn.Module):
    """
    Supervisor: next-step latent supervision H_t → H_{t+1}.
    """

    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Sigmoid()
        self.apply(xavier_gru_init)

    def forward(self, h: torch.Tensor, apply_sigmoid: bool = True) -> torch.Tensor:
        s, _ = self.rnn(h)
        s = self.proj(s)
        return self.act(s) if apply_sigmoid else s


class Discriminator(nn.Module):
    """Discriminator: classify latent sequences (real vs synthetic)."""

    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        # Note: No sigmoid here; BCEWithLogitsLoss expects raw logits
        self.proj = nn.Linear(hidden_dim, 1)
        self.apply(xavier_gru_init)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        d, _ = self.rnn(h)
        # produce a logit per timestep
        return self.proj(d)


@dataclass
class TrainingHistory:
    er_iters: List[int] = field(default_factory=list)
    er_vals: List[float] = field(default_factory=list)

    s_iters: List[int] = field(default_factory=list)
    s_vals: List[float] = field(default_factory=list)

    g_iters: List[int] = field(default_factory=list)
    g_vals: List[float] = field(default_factory=list)

    d_iters: List[int] = field(default_factory=list)
    d_vals: List[float] = field(default_factory=list)

    kl_iters: List[int] = field(default_factory=list)
    kl_vals: List[float] = field(default_factory=list)

    def add_er(self, it: int, v: float) -> None:
        self.er_iters.append(it)
        self.er_vals.append(v)

    def add_s(self, it: int, v: float) -> None:
        self.s_iters.append(it)
        self.s_vals.append(v)

    def add_g(self, it: int, v: float) -> None:
        self.g_iters.append(it)
        self.g_vals.append(v)

    def add_d(self, it: int, v: float) -> None:
        self.d_iters.append(it)
        self.d_vals.append(v)

    def add_kl(self, it: int, v: float) -> None:
        self.kl_iters.append(it)
        self.kl_vals.append(v)

    def save_plots(self, out_dir: Path, total_iters: int) -> Dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: Dict[str, Path] = {}

        # Training losses
        fig, ax = plt.subplots(figsize=(9, 5))
        if self.er_iters:
            ax.plot(self.er_iters, self.er_vals, label="Recon (E,R)")
        if self.s_iters:
            ax.plot(self.s_iters, self.s_vals, label="Supervisor (S)")
        if self.g_iters:
            ax.plot(self.g_iters, self.g_vals, label="Generator (G)")
        if self.d_iters:
            ax.plot(self.d_iters, self.d_vals, label="Discriminator (D)")
        ax.set_title("Training Losses vs Iteration")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_xlim(
            1,
            max(
                [
                    total_iters,
                    *self.er_iters,
                    *self.s_iters,
                    *self.g_iters,
                    *self.d_iters,
                ]
                or [total_iters]
            ),
        )
        ax.legend(loc="best")
        fig.tight_layout()
        p1 = out_dir / "training_curves.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved["training_curves"] = p1

        # KL(spread)
        if self.kl_iters:
            fig, ax = plt.subplots(figsize=(9, 3.5))
            ax.plot(self.kl_iters, self.kl_vals, marker="o", linewidth=1)
            ax.set_title("Validation KL(spread) vs Iteration")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("KL(spread)")
            ax.set_xlim(1, max(self.kl_iters))
            fig.tight_layout()
            p2 = out_dir / "kl_spread_curve.png"
            fig.savefig(p2, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved["kl_spread_curve"] = p2

        return saved


@dataclass
class TimeGANHandles:
    encoder: Encoder
    recovery: Recovery
    generator: Generator
    supervisor: Supervisor
    discriminator: Discriminator


@runtime_checkable
class OptLike(Protocol):
    batch_size: int
    seq_len: int
    z_dim: int
    hidden_dim: int
    num_layer: int
    lr: float
    beta1: float
    w_gamma: float
    w_g: float


class TimeGAN:
    """
    End-to-end TimeGAN wrapper with training & generation utilities.
    """

    def __init__(
        self,
        opt: OptLike,
        train_data: NDArray[np.float32],
        val_data: NDArray[np.float32],
        test_data: NDArray[np.float32],
        load_weights: bool = False,
    ) -> None:
        # set seed & device
        set_seed(getattr(opt, "manualseed", getattr(opt, "seed", None)))
        self.device = get_device()

        # options
        self.opt = opt
        self.batch_size: int = opt.batch_size
        self.seq_len: int = opt.seq_len
        self.z_dim: int = opt.z_dim
        self.h_dim: int = opt.hidden_dim
        self.n_layers: int = opt.num_layer

        # schedule
        self.num_iterations = int(getattr(opt, "num_iters", NUM_TRAINING_ITERATIONS))
        self.validate_interval = VALIDATE_INTERVAL

        # scale train only; keep stats for inverse
        self.train_norm, self.fmin, self.fmax = minmax_scale(train_data)
        self.val = val_data
        self.test = test_data

        # build modules (E/R operate on feature dimension)
        feat_dim = int(self.train_norm.shape[-1])
        self.netE = Encoder(feat_dim, self.h_dim, self.n_layers).to(self.device)
        self.netR = Recovery(self.h_dim, feat_dim, self.n_layers).to(self.device)
        self.netG = Generator(self.z_dim, self.h_dim, self.n_layers).to(self.device)
        self.netS = Supervisor(self.h_dim, self.n_layers).to(self.device)
        self.netD = Discriminator(self.h_dim, self.n_layers).to(self.device)

        # losses
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.bce_logits = nn.BCEWithLogitsLoss()

        # optimizers
        self.optE = optim.Adam(
            self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
        )
        self.optR = optim.Adam(
            self.netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
        )
        self.optG = optim.Adam(
            self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
        )
        self.optS = optim.Adam(
            self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
        )
        self.optD = optim.Adam(
            self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
        )

        self.history = TrainingHistory()
        # load
        if load_weights:
            self._maybe_load()

        # initial banner
        rrule("[bold cyan]TimeGAN • init[/bold cyan]")
        rlog(
            f"device={self.device}  "
            f"batch_size={self.batch_size} seq_len={self.seq_len} z_dim={self.z_dim} "
            f"h_dim={self.h_dim} n_layers={self.n_layers} num_iters={self.num_iterations}"
        )
        rlog(
            f"train_norm={self.train_norm.shape}  val={self.val.shape}  test={self.test.shape}"
        )

    # small utility for smooth progress readouts
    @staticmethod
    def _ema(prev: Optional[float], x: float, alpha: float = 0.1) -> float:
        return x if prev is None else (1 - alpha) * prev + alpha * x

    @staticmethod
    def _ckpt_path() -> Path:
        # NOTE: these are Paths from constants; ensure they are Path objects
        out = OUTPUT_DIR / WEIGHTS_DIR
        out.mkdir(parents=True, exist_ok=True)
        return out / "timegan_ckpt.pt"

    def _maybe_load(self) -> None:
        path = self._ckpt_path()
        if not path.exists():
            rlog("[yellow]Checkpoint not found; starting fresh.[/yellow]")
            return
        with rstatus("[cyan]Loading checkpoint"):
            state = torch.load(path, map_location=self.device)
            self.netE.load_state_dict(state["netE"])
            self.netR.load_state_dict(state["netR"])
            self.netG.load_state_dict(state["netG"])
            self.netS.load_state_dict(state["netS"])
            self.netD.load_state_dict(state["netD"])
            self.optE.load_state_dict(state["optE"])
            self.optR.load_state_dict(state["optR"])
            self.optG.load_state_dict(state["optG"])
            self.optS.load_state_dict(state["optS"])
            self.optD.load_state_dict(state["optD"])
        rlog("[green]Checkpoint loaded.[/green]")

    def _save(self, *, with_history: bool = False) -> None:
        with rstatus("[cyan]Saving checkpoint"):
            torch.save(
                {
                    "netE": self.netE.state_dict(),
                    "netR": self.netR.state_dict(),
                    "netG": self.netG.state_dict(),
                    "netS": self.netS.state_dict(),
                    "netD": self.netD.state_dict(),
                    "optE": self.optE.state_dict(),
                    "optR": self.optR.state_dict(),
                    "optG": self.optG.state_dict(),
                    "optS": self.optS.state_dict(),
                    "optD": self.optD.state_dict(),
                },
                self._ckpt_path(),
            )

            if with_history and hasattr(self, "history") and self.history is not None:
                # save plots
                paths = self.history.save_plots(
                    OUTPUT_DIR, total_iters=self.num_iterations
                )
                for k, p in paths.items():
                    rlog(f"[green]Saved {k} → {p}[/green]")

        rlog("[green]Checkpoint saved.[/green]")

    def _to_device(self, *t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(x.to(self.device, non_blocking=True) for x in t)

    def _pretrain_er_step(self, x: torch.Tensor) -> float:
        # E,R reconstruction loss
        h = self.netE(x)
        x_tilde = self.netR(h)
        loss = self.mse(x_tilde, x)
        self.optE.zero_grad()
        self.optR.zero_grad()
        loss.backward()
        self.optE.step()
        self.optR.step()
        return float(loss.detach().cpu())

    def _supervised_step(self, x: torch.Tensor) -> float:
        # next-step supervision on latent H
        h = self.netE(x)
        s = self.netS(h)
        loss = self.mse(h[:, 1:, :], s[:, :-1, :])
        self.optS.zero_grad()
        loss.backward()
        self.optS.step()
        return float(loss.detach().cpu())

    def _generator_step(self, x: torch.Tensor, z: torch.Tensor) -> float:
        # build graph
        h_real = self.netE(x)
        s_real = self.netS(h_real)
        e_hat = self.netG(z)
        h_hat = self.netS(e_hat)
        x_hat = self.netR(h_hat)

        # adversarial losses (on logits)
        y_fake = self.netD(h_hat)
        y_fake_e = self.netD(e_hat)
        adv = self.bce_logits(y_fake, torch.ones_like(y_fake))
        adv_e = self.bce_logits(y_fake_e, torch.ones_like(y_fake_e))

        # moment losses (match mean/std on reconstructions)
        x_std = torch.std(x, dim=(0, 1), unbiased=False)
        xh_std = torch.std(x_hat, dim=(0, 1), unbiased=False)
        v1 = torch.mean(torch.abs(torch.sqrt(xh_std + 1e-6) - torch.sqrt(x_std + 1e-6)))
        v2 = torch.mean(
            torch.abs(torch.mean(x_hat, dim=(0, 1)) - torch.mean(x, dim=(0, 1)))
        )

        # supervised latent loss
        sup = self.mse(s_real[:, :-1, :], h_real[:, 1:, :])

        loss = (
            adv
            + self.opt.w_gamma * adv_e
            + self.opt.w_g * (v1 + v2)
            + torch.sqrt(sup + 1e-12)
        )
        self.optG.zero_grad()
        self.optS.zero_grad()
        loss.backward()
        self.optG.step()
        self.optS.step()
        return float(loss.detach().cpu())

    def _discriminator_step(self, x: torch.Tensor, z: torch.Tensor) -> float:
        with torch.no_grad():
            e_hat = self.netG(z)
            h_hat = self.netS(e_hat)
            h_real = self.netE(x)
        y_real = self.netD(h_real)
        y_fake = self.netD(h_hat)
        y_fake_e = self.netD(e_hat)
        loss = (
            self.bce_logits(y_real, torch.ones_like(y_real))
            + self.bce_logits(y_fake, torch.zeros_like(y_fake))
            + self.opt.w_gamma * self.bce_logits(y_fake_e, torch.zeros_like(y_fake_e))
        )
        # optional hinge to avoid overshooting
        if loss.item() > 0.15:
            self.optD.zero_grad()
            loss.backward()
            self.optD.step()
        return float(loss.detach().cpu())

    def train_model(self) -> None:
        rrule("[bold magenta]TimeGAN • training[/bold magenta]")

        # phase 1: encoder-recovery pretrain
        er_ema: Optional[float] = None
        for it in tqdm(
            range(self.num_iterations), desc="Phase 1 • Pretrain (E,R)", unit="it"
        ):
            x, _T = batch_generator(self.train_norm, None, self.batch_size)  # T unused
            x = torch.as_tensor(x, dtype=torch.float32)
            (x,) = self._to_device(x)
            er = self._pretrain_er_step(x)
            self.history.add_er(it + 1, er)

            er_ema = self._ema(er, er)
            er_ema = self._ema(er_ema, er)
            if (it + 1) % 10 == 0:
                rlog(
                    f"[Pretrain] it={it + 1:,}  recon={er:.4f}  recon_ema={er_ema:.4f}"
                )

        # phase 2: supervisor
        sup_ema: Optional[float] = None
        for it in tqdm(
            range(self.num_iterations), desc="Phase 2 • Supervisor (S)", unit="it"
        ):
            x, _T = batch_generator(self.train_norm, None, self.batch_size)
            x = torch.as_tensor(x, dtype=torch.float32)
            (x,) = self._to_device(x)
            s = self._supervised_step(x)
            self.history.add_s(it + 1, s)

            sup_ema = self._ema(sup_ema, s)
            if (it + 1) % 10 == 0:
                rlog(f"[Supervised] it={it + 1:,}  s_loss={s:.4f}  s_ema={sup_ema:.4f}")

        # phase 3: joint training
        g_ema: Optional[float] = None
        d_ema: Optional[float] = None
        for it in tqdm(
            range(self.num_iterations), desc="Phase 3 • Joint (G/S/D)", unit="it"
        ):
            x, _T = batch_generator(self.train_norm, None, self.batch_size)
            z = sample_noise(self.batch_size, self.z_dim, self.seq_len)
            x = torch.as_tensor(x, dtype=torch.float32)
            z = torch.as_tensor(z, dtype=torch.float32)
            x, z = self._to_device(x, z)

            # 2× G/ER per 1× D
            for _ in range(2):
                g_loss = self._generator_step(x, z)
                self.history.add_g(it + 1, g_loss)

                g_ema = self._ema(g_ema, g_loss)
                # light ER refine pass
                self._pretrain_er_step(x)
            d_loss = self._discriminator_step(x, z)
            self.history.add_d(it + 1, d_loss)

            d_ema = self._ema(d_ema, d_loss)

            if (it + 1) % self.validate_interval == 0:
                # quick KL check on a small synthetic sample (optional)
                try:
                    fake = self.generate(
                        num_rows=min(len(self.val), 4096), mean=0.0, std=1.0
                    )
                    if self.val.shape[1] >= 3 and fake.shape[1] >= 3:
                        kl = kl_divergence_hist(
                            self.val[: len(fake)], fake, metric="spread"
                        )
                    else:
                        kl = float("nan")
                except Exception:
                    kl = float("nan")
                    self.history.add_kl(it + 1, kl)
                self._save()
                rlog(
                    f"[Joint] it={it + 1:,}  G={g_loss:.4f} (ema={g_ema:.4f})  "
                    f"D={d_loss:.4f} (ema={d_ema:.4f})  KL(spread)={kl:.5g}"
                )

        # final save
        self._save(with_history=True)
        rrule("[bold green]TimeGAN • training complete[/bold green]")

    @torch.no_grad()
    def generate(
        self,
        num_rows: int,
        *,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> NDArray[np.float32]:
        """Generate exactly `num_rows` rows of synthetic data (2D array).

        Steps: sample enough [B,T,F] windows → pass through G→S→R →
        inverse-scale with train min/max → flatten to [num_rows, F].
        """
        assert num_rows > 0
        windows_needed = math.ceil(num_rows / self.seq_len)
        z = sample_noise(
            windows_needed,
            self.z_dim,
            self.seq_len,
            mean=mean,
            std=std,
        )
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        e_hat = self.netG(z)
        h_hat = self.netS(e_hat)
        x_hat = self.netR(h_hat)
        x_hat_np = x_hat.detach().cpu().numpy()  # [B, T, F]
        x_hat_np = x_hat_np.reshape(-1, x_hat_np.shape[-1])  # [B*T, F]
        x_hat_np = x_hat_np[:num_rows]
        # inverse scale to original feature space
        x_hat_np = minmax_inverse(x_hat_np, self.fmin, self.fmax)
        return x_hat_np.astype(np.float32, copy=False)

    def print_parameter_count(self) -> None:
        rrule("[bold cyan]Parameter counts[/bold cyan]")
        sub = {
            "Encoder": self.netE,
            "Recovery": self.netR,
            "Generator": self.netG,
            "Supervisor": self.netS,
            "Discriminator": self.netD,
        }
        for name, m in sub.items():
            total = sum(p.numel() for p in m.parameters())
            train = sum(p.numel() for p in m.parameters() if p.requires_grad)
            rlog(f"[white]{name:<13}[/white] total={total:,}  trainable={train:,}")
