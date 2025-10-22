#!/usr/bin/env python3
"""
Generate synthetic LOB sequences with a trained TimeGAN and save results.

This script loads a trained TimeGAN checkpoint, generates a flat 2D array of
synthetic limit order book (LOB) rows that matches the length of the held-out
test split, and writes the array to ``gen_data.npy`` under ``OUTPUT_DIR``.

It consumes command-line options via the unified ``Options`` router:
dataset-related flags are parsed by ``DataOptions`` and model/training flags by
``ModulesOptions``. Only the dataset and modules sections are used here; no
visualization is performed.

Created By: Radhesh Goel (Keys-I)
ID: s49088276
"""

from __future__ import annotations

import numpy as np

from src.dataset import load_data
from src.helpers.args import Options
from src.helpers.constants import OUTPUT_DIR
from src.modules import TimeGAN


def predict() -> None:
    """Load data and model, generate synthetic rows, and save to disk.

    This function:
      1. Parses command-line arguments via ``Options``.
      2. Loads the dataset using only the dataset options.
      3. Builds a TimeGAN model using only the modules/training options and
         restores weights if available.
      4. Generates a flat array of synthetic rows with the same length as the
         test split (flattening windows if needed).
      5. Saves the result to ``OUTPUT_DIR / "gen_data.npy"`` and prints the
         output path and array shape.

    Returns:
        None
    """
    # Parse CLI args (top-level)
    top = Options().parse()

    # Load data using ONLY dataset options
    train_data, val_data, test_data = load_data(top.dataset)

    # Build model using ONLY modules/training options
    model = TimeGAN(top.modules, train_data, val_data, test_data, load_weights=True)

    # Inference: generate exactly len(test_data) rows (2D array)
    if getattr(test_data, "ndim", None) == 3:
        num_rows = int(test_data.shape[0] * test_data.shape[1])
    else:
        num_rows = int(test_data.shape[0])
    synth = model.generate(num_rows=num_rows, mean=0.0, std=1.0)

    # Save
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gen_data.npy"
    np.save(out_path, synth)
    print(f"Saved synthetic data to: {out_path} | shape={synth.shape}")


if __name__ == "__main__":
    predict()
