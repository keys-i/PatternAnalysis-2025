#!/usr/bin/env python3
"""
Train, validate, and test TimeGAN on AMZN LOBSTER level-10 sequences.

This entrypoint wires the unified CLI options, loads data, prepares validation
views, constructs a TimeGAN model, and runs the canonical three-phase schedule:
(1) Encoder–Recovery pretrain, (2) Supervisor pretrain, and (3) joint
adversarial training. It also enables periodic evaluation hooks inside the model
for KL divergence on spread/returns and SSIM on depth heatmaps, with checkpoints
and plots saved by the model.

The model components are defined in ``src.modules`` and the data loader in
``src.dataset``.

Created By: Radhesh Goel (Keys-I)
ID: s49088276
"""

from __future__ import annotations

from src.dataset import load_data
from src.helpers.args import Options
from src.modules import TimeGAN


def train() -> None:
    """Parse options, load data, build the model, and run training.

    Steps:
        1. Parse top-level CLI options using ``Options``.
        2. Load train/val/test splits via dataset options.
        3. Flatten validation and test windows to 2D views when needed for
           metrics that expect shape ``[T', F]``.
        4. Construct ``TimeGAN`` with module/training options and run the
           three-phase training routine.

    Returns:
        None
    """
    # Parse CLI options
    opt = Options().parse()

    # Load data using dataset options
    train_data, val_data, test_data = load_data(opt.dataset)

    # If val/test are windowed [N, T, F], flatten to [T', F] for metric views
    if getattr(val_data, "ndim", None) == 3:
        val_data = val_data.reshape(-1, val_data.shape[-1])
    if getattr(test_data, "ndim", None) == 3:
        test_data = test_data.reshape(-1, test_data.shape[-1])

    # Build model from module options and train
    model = TimeGAN(opt.modules, train_data, val_data, test_data, load_weights=False)
    model.train_model()


if __name__ == "__main__":
    train()
