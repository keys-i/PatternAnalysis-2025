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
from dataset import load_data
from modules import TimeGAN
from src.helpers.args import Options


def train() -> None:
    # parse cli args as before
    opt = Options().parse()

    # train_data: [N, T, F]; val/test should be 2D [T, F] for quick metrics
    train_data, val_data, test_data = load_data(opt)
    # if val/test come windowed [N, T, F], flatten to [T', F]
    if getattr(val_data, "ndim", None) == 3:
        val_data = val_data.reshape(-1, val_data.shape[-1])
    if getattr(test_data, "ndim", None) == 3:
        test_data = test_data.reshape(-1, test_data.shape[-1])

    # build and train
    model = TimeGAN(opt, train_data, val_data, test_data, load_weights=False)
    model.train_model()


if __name__ == "__main__":
    train()
