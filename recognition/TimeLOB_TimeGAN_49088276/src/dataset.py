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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.helpers.constants import DATA_DIR, ORDERBOOK_FILENAME, TRAIN_TEST_SPLIT
from src.helpers.richie import dataset_summary
from src.helpers.richie import log as rlog
from src.helpers.richie import status as rstatus


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

    def transform(self, data: NDArray[np.floating]) -> NDArray[np.floating]:
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
    data_dir: Path = DATA_DIR
    orderbook_filename: str = ORDERBOOK_FILENAME
    splits: Tuple[float, float, float] = TRAIN_TEST_SPLIT
    shuffle_windows: bool = True
    dtype: type = np.float32
    filter_zero_rows: bool = True

    @classmethod
    def from_namespace(cls, arg: Namespace) -> "DatasetConfig":
        return cls(
            seq_len=getattr(arg, "seq_len", 128),
            data_dir=Path(getattr(arg, "data_dir", DATA_DIR)),
            orderbook_filename=getattr(arg, "orderbook_filename", ORDERBOOK_FILENAME),
            shuffle_windows=getattr(arg, "shuffle_windows", True),
            dtype=getattr(arg, "dtype", np.float32),
            filter_zero_rows=getattr(arg, "filter_zero_rows", True),
        )


class LOBDataset:
    """
    End-to-end loader for a single LOBSTER orderbook file
    """

    def __init__(self, cfg: DatasetConfig, scaler: Optional[MinMaxScaler] = None):
        self.cfg = cfg
        self.scaler = scaler or MinMaxScaler()

        self._raw: Optional[NDArray[np.int64]] = None
        self._filtered: Optional[NDArray[np.floating]] = None
        self._train: Optional[NDArray[np.floating]] = None
        self._val: Optional[NDArray[np.floating]] = None
        self._test: Optional[NDArray[np.floating]] = None

    def load(self) -> "LOBDataset":
        with rstatus("[bold cyan]Loading and preprocessing LOBSTER orderbook dataset..."):
            data = self._read_raw()
            data = (
                self._filter_unoccupied(data)
                if self.cfg.filter_zero_rows
                else data.astype(self.cfg.dtype)
            )
            self._filtered = data.astype(self.cfg.dtype)

            self._split_chronological()
            self._scale_train_only()

        self._render_summary()
        rlog("[green]Dataset loaded, split, and scaled.[/green]")
        return self

    def make_windows(self, split: str = "train") -> NDArray[np.float32]:
        """
        Window the selected split into shape (num_windows, seq_len, num_features).
        """
        data = self._select_split(split)
        return self._windowize(data, self.cfg.seq_len, self.cfg.shuffle_windows)

    def dataset_windowed(
        self,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """
        Return (train_w, val_w, test_w) as windowed arrays.
        """
        train_w = self.make_windows(split="train")
        val_w = self.make_windows(split="val")
        test_w = self.make_windows(split="test")
        return train_w, val_w, test_w

    def _read_raw(self) -> NDArray[np.int64]:
        path = Path(self.cfg.data_dir, self.cfg.orderbook_filename)
        if not path.exists():
            msg = (
                f"{path} not found.\n"
                "Download AMZN level-10 sample from:\n"
                "https://lobsterdata.com/info/sample/LOBSTER_SampleFile_AMZN_2012-06-21_10.zip\n"
                "and place the '..._orderbook_10' file in the data directory."
            )
            raise FileNotFoundError(msg)
        rlog(f"[bold]Reading orderbook file[/bold]: {path}")
        raw = np.loadtxt(path, delimiter=",", skiprows=0, dtype=np.int64)
        rlog(f"Raw shape: {raw.shape}")
        self._raw = raw
        return raw

    def _filter_unoccupied(self, data: NDArray[np.int64]) -> NDArray[np.float32]:
        """
        Remove rows containing zeros (dummy volumes) to avoid invalid states
        """
        mask = ~(data == 0).any(axis=1)
        filtered = data[mask].astype(np.float32)
        rlog(f"Filtered rows (no zeros). Shape {filtered.shape}")
        return filtered

    def _split_chronological(self) -> None:
        assert self._filtered is not None, "Call load() first."
        n = len(self._filtered)
        a, b, c = self.cfg.splits

        # proportions if they sum to ~1.0; otherwise treat as cumulative cutoffs
        if abs((a + b + c) - 1.0) < 1e-6:
            # proportions → cumulative
            t_cut = int(n * a)
            v_cut = int(n * (a + b))
        else:
            # cumulative; require 0 < a < b <= 1.0
            if not (0.0 < a < b <= 1.0 + 1e-9):
                raise ValueError(
                    f"Invalid cumulative splits {self.cfg.splits}; expected 0 < TRAIN < VAL ≤ 1."
                )
            t_cut = int(n * a)
            v_cut = int(n * b)

        self._train = self._filtered[:t_cut]
        self._val = self._filtered[t_cut:v_cut]
        self._test = self._filtered[v_cut:]

        # window-aware sanity check
        L = self.cfg.seq_len

        def nwin(x: Optional[NDArray[np.floating]]) -> int:
            if x is None:
                return 0
            return len(x) - L + 1

        min_w = 5
        if any(nwin(x) < min_w for x in (self._train, self._val, self._test)):
            raise ValueError(
                f"Not enough windows with seq_len={L} (need ≥{min_w}): "
                f"train={nwin(self._train)}, val={nwin(self._val)}, test={nwin(self._test)}. "
                "Try smaller --seq-len, different --splits, or --keep_zero_rows."
            )

    def _scale_train_only(self) -> None:
        assert self._train is not None and self._val is not None and self._test is not None
        rlog("[bold magenta]Fitting MinMaxScaler on train split.[/bold magenta]")
        self._train = self.scaler.fit_transform(self._train)
        self._val = self.scaler.transform(self._val)
        self._test = self.scaler.transform(self._test)

    def _windowize(
        self, data: NDArray[np.float32], seq_len: int, shuffle_windows: bool
    ) -> NDArray[np.float32]:
        n_samples, n_features = data.shape
        n_windows = n_samples - seq_len + 1
        if n_windows <= 0:
            raise ValueError(f"seq_len={seq_len} is too large for data of length {n_samples}.")

        out = np.empty((n_windows, seq_len, n_features), dtype=self.cfg.dtype)
        for i in range(n_windows):
            out[i] = data[i : i + seq_len]
        if shuffle_windows:
            np.random.shuffle(out)
        return out

    def _select_split(self, split: str) -> NDArray[np.float32]:
        if split == "train":
            return self._train  # type: ignore[return-value]
        if split == "val":
            return self._val  # type: ignore[return-value]
        if split == "test":
            return self._test  # type: ignore[return-value]
        raise ValueError("split must be 'train', 'val' or 'test'")

    def _render_summary(self) -> None:
        # compute rows/windows
        L = self.cfg.seq_len

        def counts(arr: Optional[NDArray[np.floating]]) -> tuple[int, int]:
            rows = 0 if arr is None else int(arr.shape[0])
            wins = max(0, rows - L + 1)
            return rows, wins

        splits_for_view = [
            ("train", counts(self._train)),
            ("val", counts(self._val)),
            ("test", counts(self._test)),
        ]

        dataset_summary(
            file_path=Path(self.cfg.data_dir, self.cfg.orderbook_filename),
            seq_len=self.cfg.seq_len,
            dtype_name=self.cfg.dtype.__name__,
            filter_zero_rows=self.cfg.filter_zero_rows,
            splits=splits_for_view,
        )


def batch_generator(
    data: NDArray[np.float32],
    time: Optional[NDArray[np.int32]],
    batch_size: int,
) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
    """
    Random mini-batch generator for windowed sequences.

    Args:
        data: Array of shape [N, T, F] (windowed sequences).
        time: Optional array of shape [N] giving per-window lengths (T_i).
              If None, returns a constant length vector == data.shape[1].
        batch_size: Number of windows to sample (with replacement).

    Returns:
        data_mb: [batch_size, T, F] float32 mini-batch.
        T_mb:    [batch_size] int32 vector of sequence lengths.
    """
    if data.ndim != 3:
        raise ValueError(f"`data` must be [N, T, F]; got shape {data.shape}")

    n = data.shape[0]
    if n == 0:
        raise ValueError("Cannot sample mini-batch from empty data.")

    rng = np.random.default_rng()
    idx = rng.integers(0, n, size=batch_size)  # with replacement

    data_mb = data[idx].astype(np.float32, copy=False)

    if time is None:
        T_mb = np.full((batch_size,), data_mb.shape[1], dtype=np.int32)
    else:
        if time.shape[0] != n:
            raise ValueError(f"`time` length {time.shape[0]} does not match N={n}.")
        T_mb = time[idx].astype(np.int32, copy=False)

    return data_mb, T_mb


def load_data(
    arg: Namespace,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """
    Backwards-compatible wrapper.
    Returns:
        train_w: [Nw, T, F] windowed training sequences
        val:     [Tv, F]    validation rows (scaled)
        test:    [Ts, F]    test rows (scaled)
    """
    cfg = DatasetConfig.from_namespace(arg)
    loader = LOBDataset(cfg).load()
    train_w = loader.make_windows("train")
    val = loader._val
    test = loader._test
    rlog("[bold green]Stock dataset has been loaded and preprocessed.[/bold green]")
    return train_w, val, test
