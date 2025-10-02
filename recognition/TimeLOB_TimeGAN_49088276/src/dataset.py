"""
Preprocesses LOBSTER Limit Order Book (Level 10) data for TimeGAN training.

Loads paired LOBSTER message/order book CSVs, aligns by event index, windows into fixed-length
sequences, and scales features. Splits are chronological to avoid leakage. Samples are returned as
``(seq_len, num_features)``.

Inputs:
- ``message_10.csv`` and ``orderbook_10.csv`` for the same day (aligned rows; AMZN Level-10).

Outputs:
- NumPy arrays ``(train, val, test)`` with shape ``[num_seq, seq_len, num_features]``.

Features:
- Default ``feature_set="core"`` (5 engineered features):
  1) ``mid_price``            = 0.5 * (ask_price_1 + bid_price_1)
  2) ``spread``               = ask_price_1 - bid_price_1
  3) ``mid_log_return``       = log(mid_price_t) - log(mid_price_{t-1})
  4) ``queue_imbalance_l1``   = (bid_size_1 - ask_size_1) / (bid_size_1 + ask_size_1 + eps)
  5) ``depth_imbalance_l10``  = (Σ_i≤10 bid_size_i - Σ_i≤10 ask_size_i)
                                / (Σ_i≤10 bid_size_i + Σ_i≤10 ask_size_i + eps)

- Alternative ``feature_set="raw10"`` (40 raw LOB columns):
  ask_price_1..10, ask_size_1..10, bid_price_1..10, bid_size_1..10.

Evaluation (for the accompanying report):
- Distribution similarity: KL divergence ≤ 0.1 between generated vs. real spread and mid-price
  return distributions on a held-out test split.
- Visual similarity: SSIM > 0.6 between heatmaps of generated vs. real LOB depth snapshots.
- Also include: model architecture and parameter count, training strategy (full TimeGAN vs.
  adversarial-only or supervised-only variants), GPU type, VRAM, epochs, and total training time.
  Provide 3–5 representative heatmaps with a short error analysis.

Exports:
- ``LOBSTERDataset``  — PyTorch Dataset yielding windowed sequences.
- ``make_dataloader`` — Convenience factory for a configured DataLoader.

Created by: Radhesh Goel (Keys-I) | ID: s49088276
"""


import os


def test_dataset_exists():
    data_dir = "data"
    files = os.listdir(data_dir)
    print(f"Files in '{data_dir}': {files}")


if __name__ == "__main__":
    test_dataset_exists()
