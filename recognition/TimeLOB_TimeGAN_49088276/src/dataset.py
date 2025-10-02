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
from __future__ import annotations

import os
import argparse
from typing import Tuple, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class LOBSTERData:
    """
    Minimal loader -> features -> windows -> splits for LOBSTER L10 data.
    """
    def __init__(
        self,
        data_dir: str,
        message_file: str = "message_10.csv",
        orderbook_file: str = "orderbook_10.csv",
        feature_set: Literal["core", "raw10"] = "core",
        seq_len: int = 64,
        stride: Optional[int] = None,
        splits: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        scaler: Literal["standard", "minmax", "none"] = "standard",
        eps: float = 1e-8,
        headerless_message: bool = False,
        headerless_orderbook: bool = False,
    ):
        self.data_dir = data_dir
        self.message_path = os.path.join(data_dir, message_file)
        self.orderbook_path = os.path.join(data_dir, orderbook_file)
        self.feature_set = feature_set
        self.seq_len = int(seq_len)
        self.stride = int(stride) if stride is not None else self.seq_len
        self.splits = splits
        self.scaler_kind = scaler
        self.eps = eps
        self.headerless_message = headerless_message
        self.headerless_orderbook = headerless_orderbook

        assert abs(sum(splits) - 1.0) < 1e-9, "splits must sum to 1.0"
        assert self.seq_len > 0 and self.stride > 0, "seq_len and stride must be positive"

        self._scaler = None  # fitted on train only
        self._feature_names: List[str] = []

    # ------------------- public API -------------------

    def load_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns train, val, test arrays shaped (num_seq, seq_len, num_features).
        """
        msg_df, ob_df = self._load_csvs()
        self._check_alignment(msg_df, ob_df)
        feats = self._build_features(ob_df)

        train, val, test = self._split_chronologically(feats)
        train_s, val_s, test_s = self._scale_train_only(train, val, test)
        W_train = self._windowize(train_s)
        W_val   = self._windowize(val_s)
        W_test  = self._windowize(test_s)
        return W_train, W_val, W_test

    def get_feature_names(self) -> List[str]:
        return list(self._feature_names)

    def get_scaler(self):
        return self._scaler

    # ------------------- internals --------------------

    def _load_csvs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not os.path.isfile(self.orderbook_path):
            raise FileNotFoundError(f"Missing {self.orderbook_path}")
        if not os.path.isfile(self.message_path):
            raise FileNotFoundError(f"Missing {self.message_path}")

        # Message (6 columns)
        msg_cols = ["time", "type", "order_id", "size", "price", "direction"]
        if self.headerless_message:
            msg_df = pd.read_csv(self.message_path, header=None, names=msg_cols)
        else:
            msg_df = pd.read_csv(self.message_path)
            msg_df.columns = [str(c).strip().lower().replace(" ", "_") for c in msg_df.columns]
            if len(msg_df.columns) == 6 and set(msg_df.columns) != set(msg_cols):
                msg_df.columns = msg_cols

        # Orderbook (40 columns)
        ob_cols = (
            [f"ask_price_{i}" for i in range(1, 11)] +
            [f"ask_size_{i}"  for i in range(1, 11)] +
            [f"bid_price_{i}" for i in range(1, 11)] +
            [f"bid_size_{i}"  for i in range(1, 11)]
        )
        if self.headerless_orderbook:
            ob_df = pd.read_csv(self.orderbook_path, header=None, names=ob_cols)
        else:
            ob_df = pd.read_csv(self.orderbook_path)
            ob_df = self._normalize_orderbook_headers(ob_df, ob_cols)

        return msg_df, ob_df

    def _normalize_orderbook_headers(self, df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
        # Map common LOBSTER styles to snake_case:
        # e.g., AskPrice1 -> ask_price_1, BidSize10 -> bid_size_10
        new_cols = []
        for c in df.columns:
            s = str(c)
            s = s.replace(" ", "").replace("-", "").replace(".", "")
            s = s.replace("AskPrice", "ask_price_").replace("AskSize", "ask_size_") \
                 .replace("BidPrice", "bid_price_").replace("BidSize", "bid_size_")
            s = s.lower()
            s = s.replace("ask_price", "ask_price_").replace("ask_size", "ask_size_") \
                 .replace("bid_price", "bid_price_").replace("bid_size", "bid_size_")
            s = s.replace("__", "_")
            new_cols.append(s)
        df.columns = new_cols

        if set(df.columns) != set(target_cols) and len(df.columns) == len(target_cols):
            df.columns = target_cols
        return df

    def _check_alignment(self, msg_df: pd.DataFrame, ob_df: pd.DataFrame) -> None:
        if len(msg_df) != len(ob_df):
            raise ValueError(f"Message/Orderbook row count mismatch: {len(msg_df)} vs {len(ob_df)}")
        # LOBSTER rows are synchronized by event index; we trust row order.

    def _build_features(self, ob_df: pd.DataFrame) -> np.ndarray:
        # Ensure standard L10 columns exist
        for prefix in ("ask_price_", "ask_size_", "bid_price_", "bid_size_"):
            for L in range(1, 11):
                col = f"{prefix}{L}"
                if col not in ob_df.columns:
                    raise ValueError(f"Expected column missing: {col}")

        if self.feature_set == "raw10":
            cols = (
                [f"ask_price_{i}" for i in range(1, 11)]
                + [f"ask_size_{i}" for i in range(1, 11)]
                + [f"bid_price_{i}" for i in range(1, 11)]
                + [f"bid_size_{i}" for i in range(1, 11)]
            )
            X = ob_df[cols].to_numpy(dtype=np.float64)
            self._feature_names = cols
            return X

        if self.feature_set == "core":
            ap1 = ob_df["ask_price_1"].to_numpy(dtype=np.float64)
            bp1 = ob_df["bid_price_1"].to_numpy(dtype=np.float64)
            as1 = ob_df["ask_size_1"].to_numpy(dtype=np.float64)
            bs1 = ob_df["bid_size_1"].to_numpy(dtype=np.float64)

            # 1) mid_price
            mid_price = 0.5 * (ap1 + bp1)

            # 2) spread
            spread = ap1 - bp1

            # 3) mid_log_return
            mid_log = np.log(np.clip(mid_price, 1e-12, None))
            mid_log_return = np.concatenate([[0.0], np.diff(mid_log)])

            # 4) queue_imbalance_l1
            qi_l1 = (bs1 - as1) / (bs1 + as1 + self.eps)

            # 5) depth_imbalance_l10
            bid_depth = sum(ob_df[f"bid_size_{i}"].to_numpy(dtype=np.float64) for i in range(1, 11))
            ask_depth = sum(ob_df[f"ask_size_{i}"].to_numpy(dtype=np.float64) for i in range(1, 11))
            di_l10 = (bid_depth - ask_depth) / (bid_depth + ask_depth + self.eps)

            X = np.vstack([mid_price, spread, mid_log_return, qi_l1, di_l10]).T
            self._feature_names = [
                "mid_price",
                "spread",
                "mid_log_return",
                "queue_imbalance_l1",
                "depth_imbalance_l10",
            ]
            return X

        raise ValueError("feature_set must be 'core' or 'raw10'")

    def _split_chronologically(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        n_train = int(n * self.splits[0])
        n_val   = int(n * self.splits[1])
        n_test  = n - n_train - n_val
        train = X[:n_train]
        val   = X[n_train : n_train + n_val]
        test  = X[n_train + n_val :]
        return train, val, test

    def _scale_train_only(
        self, train: np.ndarray, val: np.ndarray, test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.scaler_kind == "none":
            return train, val, test

        if self.scaler_kind == "standard":
            scaler = StandardScaler()
        elif self.scaler_kind == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("scaler must be 'standard', 'minmax', or 'none'")

        scaler.fit(train)
        self._scaler = scaler
        return scaler.transform(train), scaler.transform(val), scaler.transform(test)

    def _windowize(self, X: np.ndarray) -> np.ndarray:
        """
        Returns windows shaped (num_seq, seq_len, num_features).
        """
        n, d = X.shape
        if n < self.seq_len:
            return np.empty((0, self.seq_len, d), dtype=np.float64)

        starts = np.arange(0, n - self.seq_len + 1, self.stride, dtype=int)
        W = np.empty((len(starts), self.seq_len, d), dtype=np.float64)
        for i, s in enumerate(starts):
            W[i] = X[s : s + self.seq_len]
        return W


# -------------------------- CLI smoke test ------------------------------------

def _basic_test_cli():
    """
    Run a smoke test ONLY when file names are provided by the user.

    Example:
      python lobster_data.py --data-dir data/AMZN/2014-01-02 \
        --message AMZN_2014-01-02_34200000_57600000_message_10.csv \
        --orderbook AMZN_2014-01-02_34200000_57600000_orderbook_10.csv \
        --headerless-message --headerless-orderbook
    """
    parser = argparse.ArgumentParser(description="LOBSTERData smoke test (filenames required).")
    parser.add_argument("--data-dir", default="data", help="Folder containing the CSVs")
    parser.add_argument("--message", required=True, help="Message CSV file name (e.g., message_10.csv)")
    parser.add_argument("--orderbook", required=True, help="Orderbook CSV file name (e.g., orderbook_10.csv)")
    parser.add_argument("--feature-set", choices=["core", "raw10"], default="core")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--scaler", choices=["standard", "minmax", "none"], default="standard")
    parser.add_argument("--headerless-message", action="store_true", help="Treat message CSV as headerless")
    parser.add_argument("--headerless-orderbook", action="store_true", help="Treat orderbook CSV as headerless")
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"Files in '{data_dir}': {sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else 'MISSING'}")

    try:
        loader = LOBSTERData(
            data_dir=data_dir,
            message_file=args.message,
            orderbook_file=args.orderbook,
            feature_set=args.feature_set,
            seq_len=args.seq_len,
            stride=args.stride,
            splits=(0.7, 0.15, 0.15),
            scaler=args.scaler,
            headerless_message=args.headerless_message,
            headerless_orderbook=args.headerless_orderbook,
        )
        W_train, W_val, W_test = loader.load_arrays()
        print("Feature names:", loader.get_feature_names())
        print("Train windows:", W_train.shape)
        print("Val windows:  ", W_val.shape)
        print("Test windows: ", W_test.shape)
        if W_train.size:
            print("Example window[0] stats -> mean:", float(W_train[0].mean()),
                  "std:", float(W_train[0].std()))
    except Exception as e:
        print("Basic test error:", e)


if __name__ == "__main__":
    _basic_test_cli()

