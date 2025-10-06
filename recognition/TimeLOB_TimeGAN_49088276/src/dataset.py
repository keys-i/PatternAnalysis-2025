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

from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.helpers.constants import DATA_DIR, ORDERBOOK_FILENAME, TRAIN_TEST_SPLIT


class MinMaxScaler:
    """
    Feature-wise min–max scaler with a scikit-learn-like API.
    """

    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon
        self._min: Optional[NDArray[np.floating]] = None
        self._max: Optional[NDArray[np.floating]] = None

    def fit(self, data: NDArray[np.floating]) -> "MinMaxScaler":
        self._min = np.min(data, axis=0)
        self._max = np.max(data, axis=0)
        return self

    def transform(
            self, data: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        if self._min is None or self._max is None:
            raise RuntimeError("Scaler must be fitted before transform.")
        numerator = data - self._min
        denominator = (self._max - self._min) + self.epsilon
        return numerator / denominator

    def fit_transform(self, data: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: NDArray[np.floating]) -> NDArray[np.floating]:
        if self._min is None or self._max is None:
            raise RuntimeError("Scaler must be fitted before inverse_transform.")
        return data * ((self._max - self._min) + self.epsilon) + self._min


@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuration for loading and preprocessing order-book data.
    """
    seq_len: int
    data_dir: Path = field(default_factory=lambda: Path(DATA_DIR))
    filename: str = ORDERBOOK_FILENAME
    splits: Tuple[float, float, float] = TRAIN_TEST_SPLIT
    shuffle: bool = True
    dtype: type = np.float32
    filter_zero_rows: bool = True

    @classmethod
    def from_namespace(cls, arg: Namespace) -> "DatasetConfig":
        return cls(
            seq_len=getattr(arg, "seq_len", 128),
            data_dir=Path(getattr(arg, "data_dir", DATA_DIR)),
            filename=getattr(arg, "filename", ORDERBOOK_FILENAME),
            shuffle=getattr(arg, "shuffle", True),
            dtype=getattr(arg, "dtype", np.float32),
            filter_zero_rows=getattr(arg, "filter_zero_rows", True),
        )


class LOBDataset:
    """
    End-to-end loader for a single LOBSTER orderbook file
    """

    def __init__(
            self, cfg: DatasetConfig,
            scaler: Optional[MinMaxScaler] = None
    ):
        self.cfg = cfg
        self.scaler = scaler or MinMaxScaler()

        self._raw: Optional[NDArray[np.int64]] = None
        self._filtered: Optional[NDArray[np.floating]] = None
        self._train: Optional[NDArray[np.floating]] = None
        self._val: Optional[NDArray[np.floating]] = None
        self._test: Optional[NDArray[np.floating]] = None

    def load(self) -> "LOBDataset":
        print("Loading and preprocessing LOBSTER orderbook dataset...")
        data = self._read_raw()
        data = self._filter_unoccupied(data) if self.cfg.filter_zero_rows else data.astype(self.cfg.dtype)
        self._filtered = data.astype(self.cfg.dtype)

        self._split_chronological()
        self._scale_train_only()
        print("Dataset loaded, split, and scaled.")
        return self

    def make_windows(
            self,
            split: str = "train"
    ) -> NDArray[np.float32]:
        """
        Window the selected split into shape (num_windows, seq_len, num_features).
        """
        data = self._select_split(split)
        return self._windowize(data, self.cfg.seq_len, self.cfg.shuffle)

    def dataset_windowed(
            self
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """
            Return (train_w, val_w, test_w) as windowed arrays.
        """
        train_w = self.make_windows(split="train")
        val_w = self.make_windows(split="val")
        test_w = self.make_windows(split="test")
        return train_w, val_w, test_w

    def _read_raw(self) -> NDArray[np.int64]:
        path = Path(self.cfg.data_dir, self.cfg.filename)
        if not path.exists():
            msg = (
                f"{path} not found.\n"
                "Download AMZN level-10 sample from:\n"
                "https://lobsterdata.com/info/sample/LOBSTER_SampleFile_AMZN_2012-06-21_10.zip\n"
                "and place the '..._orderbook_10' file in the data directory."
            )
            raise FileNotFoundError(msg)
        print("Reading orderbook file...", path)
        raw = np.loadtxt(path, delimiter=",", skiprows=0, dtype=np.int64)
        print("Raw shape:", raw.shape)
        self._raw = raw
        return raw

    def _filter_unoccupied(self, data: NDArray[np.int64]) -> NDArray[np.float32]:
        """
        Remove rows containing zeros (dummy volumes) to avoid invalid states
        """
        mask = ~(data == 0).any(axis=1)
        filtered = data[mask].astype(np.float32)
        print("Filtered rows (no zeros). Shape", filtered.shape)
        return filtered

    def _split_chronological(self) -> None:
        assert self._filtered is not None, "Call load() first."
        n = len(self._filtered)
        t_frac, v_frac, _ = self.cfg.splits
        t_cutoff = int(n * t_frac)
        v_cutoff = int(n * v_frac)
        self._train = self._filtered[:t_cutoff]
        self._val = self._filtered[t_cutoff:v_cutoff]
        self._test = self._filtered[v_cutoff:]
        assert all(
            len(d) > 5 for d in (self._train, self._val, self._test)
        ), "Each split must have at least 5 windows."
        print("Split sizes - train: %d, val: %d, test: %d", len(self._train), len(self._val), len(self._test))

    def _scale_train_only(self) -> None:
        assert (
                self._train is not None
                and self._val is not None
                and self._test is not None
        )
        print("Fitting MinMaxScaler on train split.")
        self._train = self.scaler.fit_transform(self._train)
        self._val = self.scaler.transform(self._val)
        self._test = self.scaler.transform(self._test)

    def _windowize(
            self,
            data: NDArray[np.float32],
            seq_len: int,
            shuffle: bool
    ) -> NDArray[np.float32]:
        n_samples, n_features = data.shape
        n_windows = n_samples - seq_len + 1
        if n_windows <= 0:
            raise ValueError(f"seq_len={seq_len} is too large for data of length {n_samples}.")

        out = np.empty((n_windows, seq_len, n_features), dtype=self.cfg.dtype)
        for i in range(n_windows):
            out[i] = data[i: i + seq_len]
        if shuffle:
            np.random.shuffle(out)
        return out

    def _select_split(self, split: str) -> NDArray[np.float32]:
        if split == "train": return self._train
        if split == "val": return self._val
        if split == "test": return self._test
        raise ValueError("split must be 'train', 'val' or 'test'")


def batch_generator(
        data: NDArray[np.float32],
        time: Optional[NDArray[np.float32]],
        batch_size: int,
):
    """
    Random mini-batch generator
    if `time` is None, uses a constant length equal to data.shape[1] (seq_len).
    """
    n = len(data)
    idx = np.random.randint(n)[:batch_size]
    data_mb = data[idx].astype(np.float32)
    if time is not None:
        T_mb = np.full((batch_size,), data_mb.shape[1], dtype=np.int32)
    else:
        T_mb = time[idx].astype(np.int32)
    return data_mb, T_mb


def load_data(arg: Namespace) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """
    Backwards-compatible wrapper.
    """
    cfg = DatasetConfig.from_namespace(arg)
    loader = LOBDataset(cfg).load()
    train_w = loader.make_windows("train")
    val = loader._val
    test = loader._test
    print("Stock dataset has been loaded and preprocessed.")
    return train_w, val, test

