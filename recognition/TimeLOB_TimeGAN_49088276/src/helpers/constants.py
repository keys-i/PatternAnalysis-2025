"""
Configuration constants for the project.
"""

from __future__ import annotations

import os
import subprocess
from math import isclose
from pathlib import Path


def _repo_root() -> Path:
    env = os.getenv("PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    try:
        out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
        return Path(out).resolve()
    except subprocess.CalledProcessError:
        return Path(__file__).resolve().parents[2]


ROOT_DIR = _repo_root()

OUTPUT_DIR = ROOT_DIR / "outs"
WEIGHTS_DIR = ROOT_DIR / "weights"
DATA_DIR = ROOT_DIR / "data"

ORDERBOOK_FILENAME = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"

# Training hyperparameters for TimeGAN
NUM_TRAINING_ITERATIONS = 25_000
VALIDATE_INTERVAL = 300

TRAIN_TEST_SPLIT = (0.7, 0.15, 0.15)
assert isclose(
    sum(TRAIN_TEST_SPLIT), 1.0, rel_tol=0.0, abs_tol=1e-6
), f"TRAIN_TEST_SPLIT must sum to 1.0 (got {sum(TRAIN_TEST_SPLIT):.8f})"

NUM_LEVELS = 10
