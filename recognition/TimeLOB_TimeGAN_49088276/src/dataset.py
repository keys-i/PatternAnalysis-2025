"""
Lightweight LOBSTER preprocessing with continuous Min-Max scaling.

This module removes the configuration bloat from the original pipeline and
focuses on the essentials:
    1. Load the raw order book snapshot file (level-10).
    2. Build either the 5-feature "core" representation or the raw 40 columns.
    3. Split chronologically into train/val/test.
    4. Fit a streaming-friendly min-max scaler on the training split only.
    5. Produce sliding windows ready for TimeGAN.

Created By: Radhesh Goel (Keys-I)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

ASK_PRICE_COLS = [f"ask_price_{i}" for i in range(1, 11)]
ASK_SIZE_COLS = [f"ask_size_{i}" for i in range(1, 11)]
BID_PRICE_COLS = [f"bid_price_{i}" for i in range(1, 11)]
BID_SIZE_COLS = [f"bid_size_{i}" for i in range(1, 11)]
ORDERBOOK_COLUMNS = ASK_PRICE_COLS + ASK_SIZE_COLS + BID_PRICE_COLS + BID_SIZE_COLS


@dataclass
class ContinuousMinMaxScaler:
    """
    Simple min-max scaler that keeps track of per-feature extrema and supports
    repeated transforms without relying on sklearn.
    """
    feature_range: Tuple[float, float] = (0.0, 1.0)
    eps: float = 1e-9
    data_min_: Optional[np.ndarray] = field(default=None, init=False)
    data_max_: Optional[np.ndarray] = field(default=None, init=False)

    def fit(self, data: np.ndarray) -> "ContinuousMinMaxScaler":
        arr = np.asarray(data, dtype=np.float64)
        self.data_min_ = arr.min(axis=0)
        self.data_max_ = arr.max(axis=0)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Scaler not fitted.")
        arr = np.asarray(data, dtype=np.float64)
        denom = np.maximum(self.data_max_ - self.data_min_, self.eps)
        scaled = (arr - self.data_min_) / denom
        lo, hi = self.feature_range
        return (scaled * (hi - lo) + lo).astype(arr.dtype, copy=False)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Scaler not fitted.")
        lo, hi = self.feature_range
        arr = np.asarray(data, dtype=np.float64)
        base = (arr - lo) / (hi - lo + self.eps)
        return base * (self.data_max_ - self.data_min_) + self.data_min_


class LOBSTERData:
    """
    Minimal LOBSTER loader (orderbook only) with continuous min-max scaling.

    Parameters
    ----------
    data_dir : str
        Folder containing orderbook_10.csv (and optionally message_10.csv).
    feature_set : {"core", "raw10"}
        Representation to build.
    seq_len : int
        Window length fed to TimeGAN.
    stride : int, optional
        Step between consecutive windows (defaults to seq_len for non-overlap).
    splits : tuple
        Train/val/test fractions; must sum to 1.0.
    """

    def __init__(
        self,
        data_dir: str,
        message_file: str = "message_10.csv",  # kept for compatibility; unused
        orderbook_file: str = "orderbook_10.csv",
        feature_set: Literal["core", "raw10"] = "core",
        seq_len: int = 128,
        stride: Optional[int] = None,
        splits: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        feature_range: Tuple[float, float] = (0.0, 1.0),
        dtype: Literal["float32", "float64"] = "float32",
        save_dir: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.message_file = message_file  # placeholder for potential alignment checks
        self.orderbook_path = os.path.join(data_dir, orderbook_file)
        self.feature_set = feature_set
        self.seq_len = int(seq_len)
        self.stride = int(stride) if stride is not None else self.seq_len
        self.splits = splits
        self.scaler = ContinuousMinMaxScaler(feature_range=feature_range)
        self._dtype_name = dtype
        self.dtype = np.float32 if dtype == "float32" else np.float64
        self.save_dir = save_dir
        self.eps = 1e-8
        self._validate_inputs()

    # ------------------- public API -------------------

    def load_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        orderbook = self._load_orderbook()
        features = self._build_features(orderbook)
        features = features[~np.isnan(features).any(axis=1)]
        train, val, test = self._split(features)

        self.scaler.fit(train)
        train = self.scaler.transform(train)
        val = self.scaler.transform(val)
        test = self.scaler.transform(test)

        W_train = self._windowize(train)
        W_val = self._windowize(val)
        W_test = self._windowize(test)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(self.save_dir, "windows.npz"),
                train=W_train, val=W_val, test=W_test
            )
            with open(os.path.join(self.save_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(self.get_meta(), f, indent=2)

        return W_train, W_val, W_test

    def get_meta(self) -> dict:
        return {
            "feature_set": self.feature_set,
            "seq_len": self.seq_len,
            "stride": self.stride,
            "splits": self.splits,
            "feature_range": self.scaler.feature_range,
            "dtype": self._dtype_name,
        }

    # ------------------- helpers ---------------------

    def _validate_inputs(self) -> None:
        if not os.path.exists(self.orderbook_path):
            raise FileNotFoundError(self.orderbook_path)
        if self.seq_len <= 0 or self.stride <= 0:
            raise ValueError("seq_len and stride must be positive.")
        total = sum(self.splits)
        if not np.isclose(total, 1.0):
            raise ValueError(f"splits must sum to 1.0, got {self.splits} (sum={total}).")
        if any(x <= 0 for x in self.splits):
            raise ValueError("splits must be positive.")
        lo, hi = self.scaler.feature_range
        if hi <= lo:
            raise ValueError("feature_range must satisfy min < max.")

    def _load_orderbook(self) -> pd.DataFrame:
        df = pd.read_csv(self.orderbook_path, header=None)
        if df.shape[1] < len(ORDERBOOK_COLUMNS):
            raise ValueError(f"Expected >= {len(ORDERBOOK_COLUMNS)} columns, found {df.shape[1]}.")
        df = df.iloc[:, :len(ORDERBOOK_COLUMNS)]
        numeric_ratio = pd.to_numeric(df.iloc[0], errors="coerce").notna().mean()
        if numeric_ratio < 0.5:
            df = df.iloc[1:].reset_index(drop=True)
        df.columns = ORDERBOOK_COLUMNS
        df = df.apply(pd.to_numeric, errors="coerce")
        return df

    def _build_features(self, ob_df: pd.DataFrame) -> np.ndarray:
        data = ob_df.to_numpy(dtype=np.float64)
        if self.feature_set == "raw10":
            return data
        ask_prices = data[:, :10]
        ask_sizes = data[:, 10:20]
        bid_prices = data[:, 20:30]
        bid_sizes = data[:, 30:40]

        mid_price = 0.5 * (ask_prices[:, 0] + bid_prices[:, 0])
        spread = ask_prices[:, 0] - bid_prices[:, 0]
        log_mid = np.log(np.clip(mid_price, self.eps, None))
        mid_log_return = np.concatenate([[0.0], np.diff(log_mid)])
        queue_imbalance = (
            (bid_sizes[:, 0] - ask_sizes[:, 0]) /
            (bid_sizes[:, 0] + ask_sizes[:, 0] + self.eps)
        )
        depth_imbalance = (
            (bid_sizes.sum(axis=1) - ask_sizes.sum(axis=1)) /
            (bid_sizes.sum(axis=1) + ask_sizes.sum(axis=1) + self.eps)
        )

        feats = np.stack(
            [mid_price, spread, mid_log_return, queue_imbalance, depth_imbalance],
            axis=1,
        )
        return feats

    def _split(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(feats)
        n_train = int(n * self.splits[0])
        n_val = int(n * self.splits[1])
        n_test = n - n_train - n_val
        if n_train < self.seq_len or n_val < self.seq_len or n_test < self.seq_len:
            raise ValueError(
                "Not enough rows for the requested seq_len/splits combination. "
                f"Have {n} rows with splits {self.splits}."
            )
        train = feats[:n_train]
        val = feats[n_train:n_train + n_val]
        test = feats[n_train + n_val:]
        return train, val, test

    def _windowize(self, arr: np.ndarray) -> np.ndarray:
        windows = []
        limit = len(arr) - self.seq_len + 1
        for start in range(0, limit, self.stride):
            window = arr[start:start + self.seq_len]
            if window.shape[0] == self.seq_len:
                windows.append(window)
        if not windows:
            raise ValueError("Not enough rows to create even a single window.")
        stacked = np.stack(windows).astype(self.dtype, copy=False)
        return stacked
