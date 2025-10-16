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
from pathlib import Path

import numpy as np

from dataset import load_data
from helpers.args import Options
from helpers.constants import OUTPUT_DIR
from modules import TimeGAN


def main() -> None:
    # parse CLI args (top-level)
    top = Options().parse()

    # load data using ONLY dataset options
    train_data, val_data, test_data = load_data(top.dataset)

    # build model using ONLY modules/training options
    model = TimeGAN(top.modules, train_data, val_data, test_data, load_weights=True)

    # inference: generate exactly len(test_data) rows (2D array)
    if getattr(test_data, "ndim", None) == 3:
        num_rows = int(test_data.shape[0] * test_data.shape[1])
    else:
        num_rows = int(test_data.shape[0])
    synth = model.generate(num_rows=num_rows, mean=0.0, std=1.0)

    # save
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gen_data.npy"
    np.save(out_path, synth)
    print(f"Saved synthetic data to: {out_path} | shape={synth.shape}")


if __name__ == "__main__":
    main()