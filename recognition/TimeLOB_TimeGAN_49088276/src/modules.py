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
    - Embedder
    - Recovery
    - Generator
    - Supervisor
    - Discriminator
    - TimeGAN
    - TemporalBackboneConfig

Created By: Radhesh Goel (Keys-I)
ID: s49088276

References:
- 
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_seed(seed: Optional[int]):
    if seed is None or seed < 0:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def xavier_gru_init(module: nn.Module) -> None:
    if isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


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
        x_tilde = self.rnn(h)
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
        # note: No sigmoid here; BCEWithLogitsLoss expects raw logits
        self.proj = nn.Linear(hidden_dim, 1)
        self.apply(xavier_gru_init)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        d, _ = self.rnn(h)
        # produce a logit per timestep
        return self.proj(d)

@dataclass
class TimeGANHandles:
    encoder: Encoder
    recovery: Recovery
    generator: Generator
    supervisor: Supervisor
    discriminator: Discriminator
