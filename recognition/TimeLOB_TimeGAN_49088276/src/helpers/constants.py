"""
Configuration constants for the project.
"""
from math import isclose
from typing import Literal

OUTPUT_DIR = "outs"
WEIGHTS_DIR = "weights"
DATA_DIR = "data"

ORDERBOOK_FILENAME = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"

# Training hyperparameters for TimeGAN
NUM_TRAINING_ITERATIONS = 25_000
VALIDATE_INTERVAL = 300

TRAIN_TEST_SPLIT = (0.7, 0.15, 0.15)
assert isclose(
    sum(TRAIN_TEST_SPLIT), 1.0,
    rel_tol=0.0, abs_tol=1e-6
), (
    f"TRAIN_TEST_SPLIT must sum to 1.0 (got {sum(TRAIN_TEST_SPLIT):.8f})"
)

NUM_LEVELS = 10