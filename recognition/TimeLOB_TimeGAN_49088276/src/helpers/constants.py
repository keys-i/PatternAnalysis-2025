#!/usr/bin/env python3
"""
Configuration constants for the TimeGAN–LOBSTER project.

This module centralizes project paths and stable defaults used across the codebase:
- Repository root discovery (PROJECT_ROOT env, then git, then file fallback).
- Standard directories for outputs, weights, and data.
- Dataset filename defaults and LOB level count.
- Training schedule defaults and validation cadence.
- Train/val/test split with a strict sanity check.
"""

from __future__ import annotations

import os
import subprocess
from math import isclose
from pathlib import Path


def _repo_root() -> Path:
    """Return the repository root using env, git, or file-based fallback.

    Resolution order:
        1) PROJECT_ROOT environment variable
        2) `git rev-parse --show-toplevel`
        3) Two directories above this file (for non-git environments)

    Returns:
        Absolute Path to the repository root.
    """
    env = os.getenv("PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
        return Path(out).resolve()
    except subprocess.CalledProcessError:
        return Path(__file__).resolve().parents[2]


# Root and standard directories
ROOT_DIR = _repo_root()
OUTPUT_DIR = ROOT_DIR / "outs"
WEIGHTS_DIR = ROOT_DIR / "weights"
DATA_DIR = ROOT_DIR / "data"

# Dataset defaults
ORDERBOOK_FILENAME = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
NUM_LEVELS = 10  # LOBSTER level-10 snapshots

# Training defaults
NUM_TRAINING_ITERATIONS = 25_000
VALIDATE_INTERVAL = 300

# Chronological split fractions (or convert to cumulative in the loader)
TRAIN_TEST_SPLIT = (0.7, 0.15, 0.15)
assert isclose(
    sum(TRAIN_TEST_SPLIT), 1.0, rel_tol=0.0, abs_tol=1e-6
), f"TRAIN_TEST_SPLIT must sum to 1.0 (got {sum(TRAIN_TEST_SPLIT):.8f})"
