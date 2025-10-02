"""
Load and preprocess LOBSTER level-10 order book data for TimeGAN.

This module provides a PyTorch Dataset and DataLoader factory that align,
window, and scale limit order book features (e.g., top-10 bid/ask prices and
size) into fixed-length sequences. Splits should be time-based to avoid
leakage. Tensors are returned in that shape ``(seq_len, feature_dim)``.

Exports:
    - LOBSTERDataset
    - make_dataloader

Created By: Radhesh Goel (Keys-I)
ID: s49088276

References:
- 
"""

import os

def test_dataset_exists():
    data_dir = "data"
    files = os.listdir(data_dir)
    print(f"Files in '{data_dir}': {files}")

if __name__ == "__main__":
    test_dataset_exists()
