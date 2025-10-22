#!/usr/bin/env python3
"""Lightweight LOBSTER preprocessing with continuous min–max scaling.

This module provides a minimal data pipeline for AMZN Level-10 LOBSTER snapshots:

1) Load the raw order book file (level-10, 40 columns).
2) Optionally filter out rows containing zeros.
3) Chronologically split into train, validation, and test.
4) Fit a train-only min–max scaler and transform all splits.
5) Produce sliding windows for TimeGAN training.

It also includes a batch generator for windowed sequences and a convenience
``load_data`` wrapper that returns windowed train data and 2D val/test views.

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
    """Feature-wise min–max scaler with a scikit-learn-like API.

    The scaler computes per-feature minima and maxima on the training split and
    applies an epsilon-protected min–max transform. The inverse transform uses
    the stored min and max.
    """

    def __init__(self, epsilon: float = 1e-7) -> None:
        """Initialize the scaler.

        Args:
            epsilon: Small constant added to the denominator to avoid division by zero.
        """
        self.epsilon = epsilon
        self._min: Optional[NDArray[np.floating]] = None
        self._max: Optional[NDArray[np.floating]] = None

    def fit(self, data: NDArray[np.floating]) -> "MinMaxScaler":
        """Compute per-feature minima and maxima.

        Args:
            data: Array with features along the last dimension.

        Returns:
            Self, for chaining.
        """
        self._min = np.min(data, axis=0)
        self._max = np.max(data, axis=0)
        return self

    def transform(self, data: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply min–max scaling using fitted statistics.

        Args:
            data: Array to scale.

        Returns:
            Scaled array of the same shape.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if self._min is None or self._max is None:
            raise RuntimeError("Scaler must be fitted before transform.")
        numerator = data - self._min
        denominator = (self._max - self._min) + self.epsilon
        return numerator / denominator

    def fit_transform(self, data: NDArray[np.floating]) -> NDArray[np.floating]:
        """Fit the scaler and transform the data in one call.

        Args:
            data: Array to fit and transform.

        Returns:
            Scaled array of the same shape.
        """
        return self.fit(data).transform(data)

    def inverse_transform(self, data: NDArray[np.floating]) -> NDArray[np.floating]:
        """Invert the scaling back to the original feature space.

        Args:
            data: Scaled array to invert.

        Returns:
            Array in the original feature scale.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if self._min is None or self._max is None:
            raise RuntimeError("Scaler must be fitted before inverse_transform.")
        return data * ((self._max - self._min) + self.epsilon) + self._min


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for loading and preprocessing order book data.

    Attributes:
        seq_len: Window length (time steps).
        data_dir: Directory containing the order book file.
        orderbook_filename: Filename of the Level-10 order book CSV.
        splits: Train/validation/test split fractions or cumulative cutoffs.
        shuffle_windows: Shuffle windows after windowing the chosen split.
        dtype: Target dtype for arrays.
        filter_zero_rows: Whether to drop rows containing zeros.
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
        """Build a configuration from an argparse namespace.

        Args:
            arg: Namespace carrying dataset-related flags.

        Returns:
            A populated ``DatasetConfig`` instance.
        """
        return cls(
            seq_len=getattr(arg, "seq_len", 128),
            data_dir=Path(getattr(arg, "data_dir", DATA_DIR)),
            orderbook_filename=getattr(arg, "orderbook_filename", ORDERBOOK_FILENAME),
            shuffle_windows=getattr(arg, "shuffle_windows", True),
            dtype=getattr(arg, "dtype", np.float32),
            filter_zero_rows=getattr(arg, "filter_zero_rows", True),
        )


class LOBDataset:
    """End-to-end loader for a single LOBSTER Level-10 order book file."""

    def __init__(
        self, cfg: DatasetConfig, scaler: Optional[MinMaxScaler] = None
    ) -> None:
        """Initialize the loader.

        Args:
            cfg: Dataset configuration.
            scaler: Optional external scaler; a new ``MinMaxScaler`` is created if None.
        """
        self.cfg = cfg
        self.scaler = scaler or MinMaxScaler()

        self._raw: Optional[NDArray[np.int64]] = None
        self._filtered: Optional[NDArray[np.floating]] = None
        self._train: Optional[NDArray[np.floating]] = None
        self._val: Optional[NDArray[np.floating]] = None
        self._test: Optional[NDArray[np.floating]] = None

    def load(self) -> "LOBDataset":
        """Load, split, scale, and summarize the dataset.

        Returns:
            Self, for chaining.
        """
        with rstatus(
            "[bold cyan]Loading and preprocessing LOBSTER orderbook dataset..."
        ):
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
        """Window a selected split into shape ``[num_windows, seq_len, num_features]``.

        Args:
            split: One of {'train', 'val', 'test'}.

        Returns:
            Windowed array for the chosen split.
        """
        data = self._select_split(split)
        return self._windowize(data, self.cfg.seq_len, self.cfg.shuffle_windows)

    def dataset_windowed(
        self,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Return windowed train, val, and test arrays.

        Returns:
            (train_w, val_w, test_w) where each is a 3D array of windows.
        """
        train_w = self.make_windows(split="train")
        val_w = self.make_windows(split="val")
        test_w = self.make_windows(split="test")
        return train_w, val_w, test_w

    def _read_raw(self) -> NDArray[np.int64]:
        """Read the raw CSV into a 2D array of type int64.

        Returns:
            Raw ndarray with shape [T, 40] for Level-10.

        Raises:
            FileNotFoundError: If the CSV is not present at ``data_dir/orderbook_filename``.
        """
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

    @staticmethod
    def _filter_unoccupied(data: NDArray[np.int64]) -> NDArray[np.float32]:
        """Remove rows containing any zero to avoid invalid or dummy volumes.

        Args:
            data: Raw order book rows [T, 40].

        Returns:
            Filtered float32 array.
        """
        mask = ~(data == 0).any(axis=1)
        filtered = data[mask].astype(np.float32)
        rlog(f"Filtered rows (no zeros). Shape {filtered.shape}")
        return filtered

    def _split_chronological(self) -> None:
        """Split the filtered data chronologically into train, val, and test.

        Supports both proportion splits that sum to 1.0 and cumulative cutoffs
        (e.g., 0.7, 0.85, 1.0). Ensures each split yields at least a minimum
        number of windows for the configured sequence length.
        """
        assert self._filtered is not None, "Call load() first."
        n = len(self._filtered)
        a, b, c = self.cfg.splits

        # proportions if they sum to ~1.0; otherwise treat as cumulative cutoffs
        if abs((a + b + c) - 1.0) < 1e-6:
            t_cut = int(n * a)
            v_cut = int(n * (a + b))
        else:
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
        l = self.cfg.seq_len

        def nwin(x: Optional[NDArray[np.floating]]) -> int:
            if x is None:
                return 0
            return len(x) - l + 1

        min_w = 5
        if any(nwin(x) < min_w for x in (self._train, self._val, self._test)):
            raise ValueError(
                f"Not enough windows with seq_len={l} (need ≥{min_w}): "
                f"train={nwin(self._train)}, val={nwin(self._val)}, test={nwin(self._test)}. "
                "Try smaller --seq-len, different --splits, or --keep_zero_rows."
            )

    def _scale_train_only(self) -> None:
        """Fit min–max on train and transform train, val, and test in place."""
        assert (
            self._train is not None and self._val is not None and self._test is not None
        )
        rlog("[bold magenta]Fitting MinMaxScaler on train split.[/bold magenta]")
        self._train = self.scaler.fit_transform(self._train)
        self._val = self.scaler.transform(self._val)
        self._test = self.scaler.transform(self._test)

    def _windowize(
        self,
        data: NDArray[np.float32],
        seq_len: int,
        shuffle_windows: bool,
    ) -> NDArray[np.float32]:
        """Slice a [T, F] split into overlapping windows of length ``seq_len``.

        Args:
            data: 2D array [T, F].
            seq_len: Window length.
            shuffle_windows: Shuffle windows after creation.

        Returns:
            3D array of windows [Nw, seq_len, F].

        Raises:
            ValueError: If ``seq_len`` exceeds the number of rows.
        """
        n_samples, n_features = data.shape
        n_windows = n_samples - seq_len + 1
        if n_windows <= 0:
            raise ValueError(
                f"seq_len={seq_len} is too large for data of length {n_samples}."
            )

        out = np.empty((n_windows, seq_len, n_features), dtype=self.cfg.dtype)
        for i in range(n_windows):
            out[i] = data[i : i + seq_len]
        if shuffle_windows:
            np.random.shuffle(out)
        return out

    def _select_split(self, split: str) -> NDArray[np.float32]:
        """Return the requested split array."""
        if split == "train":
            return self._train  # type: ignore[return-value]
        if split == "val":
            return self._val  # type: ignore[return-value]
        if split == "test":
            return self._test  # type: ignore[return-value]
        raise ValueError("split must be 'train', 'val' or 'test'")

    def _render_summary(self) -> None:
        """Print a dataset summary using Rich (or plain text fallback)."""
        l = self.cfg.seq_len

        def counts(arr: Optional[NDArray[np.floating]]) -> tuple[int, int]:
            rows = 0 if arr is None else int(arr.shape[0])
            wins = max(0, rows - l + 1)
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
    """Random mini-batch generator for windowed sequences.

    Args:
        data: Array of shape [N, T, F] (windowed sequences).
        time: Optional array of shape [N] giving per-window lengths (T_i).
              If None, returns a constant length vector equal to data.shape[1].
        batch_size: Number of windows to sample (with replacement).

    Returns:
        data_mb: Mini-batch of windows [batch_size, T, F] (float32).
        t_mb:    Vector of sequence lengths [batch_size] (int32).

    Raises:
        ValueError: If ``data`` is not 3D or has zero windows.
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
        t_mb = np.full((batch_size,), data_mb.shape[1], dtype=np.int32)
    else:
        if time.shape[0] != n:
            raise ValueError(f"`time` length {time.shape[0]} does not match N={n}.")
        t_mb = time[idx].astype(np.int32, copy=False)

    return data_mb, t_mb


def load_data(
    arg: Namespace,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Load, preprocess, window train, and return val/test 2D views.

    Args:
        arg: Namespace containing dataset flags (see DataOptions).

    Returns:
        train_w: Windowed training sequences [Nw, T, F].
        val:     Validation rows [Tv, F] (scaled).
        test:    Test rows [Ts, F] (scaled).
    """
    cfg = DatasetConfig.from_namespace(arg)
    loader = LOBDataset(cfg).load()
    train_w = loader.make_windows("train")
    val = loader._val
    test = loader._test
    rlog("[bold green]Stock dataset has been loaded and preprocessed.[/bold green]")
    return train_w, val, test
