"""
Train, validate, and test the TimeLOB TimeGAN on LOBSTER sequences.

This module orchestrates the three-phase TimeGAN schedule (autoencoder
pretrain, supervisor pretrain, joint adversarial training), log losses,
computes validation metrics (e.g., KL on spread/returns; SSIM on heatmaps),
and saves model checkpoints and plots. The model is imported from ``modules.py``
and data loaders from ``dataset.py``.

Created By: Radhesh Goel (Keys-I)
ID: s49088276

References:
-
"""

from src.dataset import load_data
from src.helpers.args import Options
from src.modules import TimeGAN


def train() -> None:
    # parse top-level CLI args
    opt = Options().parse()

    # dataset-only args → loader
    train_data, val_data, test_data = load_data(opt.dataset)

    # if val/test are windowed [N, T, F], flatten to [T', F]
    if getattr(val_data, "ndim", None) == 3:
        val_data = val_data.reshape(-1, val_data.shape[-1])
    if getattr(test_data, "ndim", None) == 3:
        test_data = test_data.reshape(-1, test_data.shape[-1])

    # modules-only args → model
    model = TimeGAN(opt.modules, train_data, val_data, test_data, load_weights=False)
    model.train_model()


if __name__ == "__main__":
    train()
