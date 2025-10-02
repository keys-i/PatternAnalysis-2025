"""
LOBSTERData: load, featurize, window, split (TimeGAN-ready) + CSV summaries.

- Works with headerless LOBSTER CSVs (message_10.csv, orderbook_10.csv).
- Engineered 5-feature "core" set or raw level-10 (40 columns).
- Chronological train/val/test split; scaler fit on train only.
- Windows shape: (num_seq, seq_len, num_features).
- Extras: NaN/inf cleaning, dtype control, meta, inverse_transform, NPZ export.
- NEW: summarize() and --summary CLI to inspect both message & orderbook tables.

Created by: Radhesh Goel (Keys-I) | ID: s49088276
"""

from __future__ import annotations

import os
import argparse
from typing import Tuple, List, Literal, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ------------------------------ utilities ------------------------------------ #

def _summarize_df(df: pd.DataFrame, name: str, peek: int = 5) -> str:
    lines = []
    lines.append(f"=== {name} ===")
    lines.append(f"shape: {df.shape[0]} rows × {df.shape[1]} cols")
    lines.append(f"columns: {list(df.columns)}")
    dtypes = df.dtypes.astype(str).to_dict()
    lines.append(f"dtypes: {dtypes}")
    na_counts = df.isna().sum().to_dict()
    lines.append(f"na_counts: {na_counts}")
    # time range if a 'time' column exists
    if "time" in df.columns:
        try:
            t = pd.to_datetime(df["time"], errors="coerce", unit=None)
            lines.append(f"time: min={t.min()} max={t.max()}")
        except Exception:
            pass
    # numeric quick stats
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        desc = df[num_cols].describe().to_dict()
        # ensure json-like floats, not numpy types
        desc = {k: {m: float(v) for m, v in stats.items()} for k, stats in desc.items()}
        lines.append("numeric.describe():")
        lines.append(str(desc))
    # head/tail
    lines.append("head:")
    lines.append(df.head(peek).to_string(index=False))
    lines.append("tail:")
    lines.append(df.tail(peek).to_string(index=False))
    return "\n".join(lines)


# ------------------------------- core class ---------------------------------- #

class LOBSTERData:
    """
    Loader -> features -> windows -> splits for LOBSTER L10 data.
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
        feature_range: Tuple[float, float] = (0.0, 1.0),  # for minmax
        eps: float = 1e-8,
        headerless_message: bool = False,
        headerless_orderbook: bool = False,
        dropna: bool = True,
        output_dtype: Literal["float32", "float64"] = "float32",
    ):
        self.data_dir = data_dir
        self.message_path = os.path.join(data_dir, message_file)
        self.orderbook_path = os.path.join(data_dir, orderbook_file)
        self.feature_set = feature_set
        self.seq_len = int(seq_len)
        self.stride = int(stride) if stride is not None else self.seq_len
        self.splits = splits
        self.scaler_kind = scaler
        self.feature_range = feature_range
        self.eps = eps
        self.headerless_message = headerless_message
        self.headerless_orderbook = headerless_orderbook
        self.dropna = dropna
        self.output_dtype = np.float32 if output_dtype == "float32" else np.float64

        self._validate_splits()
        if not (self.seq_len > 0 and self.stride > 0):
            raise ValueError("seq_len and stride must be positive")

        self._scaler = None  # fitted on train only
        self._feature_names: List[str] = []
        self._row_counts: Dict[str, int] = {}

    # ------------------- public API -------------------

    def load_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns train, val, test arrays shaped (num_seq, seq_len, num_features).
        """
        msg_df, ob_df = self._load_csvs()
        self._check_alignment(msg_df, ob_df)
        feats = self._build_features(ob_df)

        # hygiene
        if self.dropna:
            feats = feats[~np.isnan(feats).any(axis=1)]
        feats = feats[np.isfinite(feats).all(axis=1)]
        self._row_counts["post_clean"] = int(feats.shape[0])

        train, val, test = self._split_chronologically(feats)
        self._row_counts.update(train=len(train), val=len(val), test=len(test))

        train_s, val_s, test_s = self._scale_train_only(train, val, test)
        W_train = self._windowize(train_s)
        W_val   = self._windowize(val_s)
        W_test  = self._windowize(test_s)

        # final dtype cast
        W_train = W_train.astype(self.output_dtype, copy=False)
        W_val   = W_val.astype(self.output_dtype, copy=False)
        W_test  = W_test.astype(self.output_dtype, copy=False)
        return W_train, W_val, W_test

    def summarize(self, peek: int = 5) -> str:
        """Human-readable summary of both message and orderbook CSVs."""
        msg_df, ob_df = self._load_csvs()
        # ensure normalized headers for orderbook are visible
        _ = self._normalize_orderbook_headers(
            ob_df,
            [f"ask_price_{i}" for i in range(1, 11)]
            + [f"ask_size_{i}" for i in range(1, 11)]
            + [f"bid_price_{i}" for i in range(1, 11)]
            + [f"bid_size_{i}" for i in range(1, 11)]
        )
        parts = [
            _summarize_df(msg_df, "message_10.csv", peek=peek),
            _summarize_df(ob_df, "orderbook_10.csv", peek=peek),
        ]
        return "\n\n".join(parts)

    def get_feature_names(self) -> List[str]:
        return list(self._feature_names)

    def get_scaler(self):
        return self._scaler

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """Inverse-transform features (per time-step) using the fitted scaler."""
        if self._scaler is None:
            raise RuntimeError("Scaler not fitted; call load_arrays() first or use scaler='none'.")
        orig_shape = arr.shape
        flat = arr.reshape(-1, arr.shape[-1])
        inv = self._scaler.inverse_transform(flat)
        return inv.reshape(orig_shape)

    def get_meta(self) -> Dict[str, object]:
        return {
            "feature_set": self.feature_set,
            "feature_names": self.get_feature_names(),
            "seq_len": self.seq_len,
            "stride": self.stride,
            "splits": self.splits,
            "scaler": type(self._scaler).__name__ if self._scaler is not None else "None",
            "row_counts": self._row_counts,
        }

    # ------------------- internals --------------------

    def _validate_splits(self) -> None:
        s = sum(self.splits)
        if not (abs(s - 1.0) < 1e-12):
            raise ValueError(f"splits must sum to 1.0, got {self.splits} (sum={s})")
        if any(x < 0 for x in self.splits):
            raise ValueError("splits cannot be negative")

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
            # if columns are 6 but non-standard, coerce to canonical names
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

        # If still mismatched but counts align, force target order.
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

            # 3) mid_log_return (first element 0.0 to preserve length)
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
        if n < self.seq_len:
            raise ValueError(
                f"Not enough rows ({n}) for seq_len={self.seq_len}. "
                "Consider reducing seq_len or collecting more data."
            )
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
            scaler = MinMaxScaler(feature_range=self.feature_range)
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
        if starts.size == 0:
            return np.empty((0, self.seq_len, d), dtype=np.float64)

        W = np.empty((len(starts), self.seq_len, d), dtype=np.float64)
        for i, s in enumerate(starts):
            W[i] = X[s : s + self.seq_len]
        return W


# -------------------------- CLI: smoke test & summary ------------------------- #

def _main_cli():
    parser = argparse.ArgumentParser(description="LOBSTERData (preprocess + summarize).")
    parser.add_argument("--data-dir", default="data", help="Folder containing the CSVs")
    parser.add_argument("--message", required=True, help="Message CSV file name (e.g., message_10.csv)")
    parser.add_argument("--orderbook", required=True, help="Orderbook CSV file name (e.g., orderbook_10.csv)")
    parser.add_argument("--feature-set", choices=["core", "raw10"], default="core")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--splits", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
                        default=(0.7, 0.15, 0.15), help="Fractions that must sum to 1.0")
    parser.add_argument("--scaler", choices=["standard", "minmax", "none"], default="standard")
    parser.add_argument("--feature-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=(0.0, 1.0))
    parser.add_argument("--headerless-message", action="store_true", help="Treat message CSV as headerless")
    parser.add_argument("--headerless-orderbook", action="store_true", help="Treat orderbook CSV as headerless")
    parser.add_argument("--no-dropna", action="store_true", help="Disable row drop for NaN")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--save-npz", type=str, default=None, help="If set, save windows to this .npz path")
    parser.add_argument("--summary", action="store_true", help="Print a summary of both CSVs and exit")
    parser.add_argument("--peek", type=int, default=5, help="Rows to show in head/tail for summary")
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"Files in '{data_dir}': {sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else 'MISSING'}")

    loader = LOBSTERData(
        data_dir=data_dir,
        message_file=args.message,
        orderbook_file=args.orderbook,
        feature_set=args.feature_set,
        seq_len=args.seq_len,
        stride=args.stride,
        splits=tuple(args.splits),
        scaler=args.scaler,
        feature_range=tuple(args.feature_range),
        headerless_message=args.headerless_message,
        headerless_orderbook=args.headerless_orderbook,
        dropna=not args.no_dropna,
        output_dtype=args.dtype,
    )

    if args.summary:
        print(loader.summarize(peek=args.peek))
        return

    # Build windows
    W_train, W_val, W_test = loader.load_arrays()
    meta = loader.get_meta()

    print("Feature names:", loader.get_feature_names())
    print("Meta:", meta)
    print("Train windows:", W_train.shape)
    print("Val windows:  ", W_val.shape)
    print("Test windows: ", W_test.shape)
    if W_train.size:
        print("Example window[0] stats -> mean:", float(W_train[0].mean()),
              "std:", float(W_train[0].std()))

    if args.save_npz:
        np.savez_compressed(
            args.save_npz,
            train=W_train, val=W_val, test=W_test,
            feature_names=np.array(loader.get_feature_names(), dtype=object),
            meta=np.array([str(meta)], dtype=object),
        )
        print(f"Saved windows to: {args.save_npz}")


if __name__ == "__main__":
    _main_cli()
