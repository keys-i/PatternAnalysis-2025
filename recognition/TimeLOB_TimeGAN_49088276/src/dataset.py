"""
LOBSTER (Level-10) preprocessing for TimeGAN.

- Loads paired LOBSTER CSVs (message_10.csv, orderbook_10.csv), aligned by event index.
- Builds either a compact engineered 5-feature set ("core") or raw level-10 depth ("raw10").
- Chronological train/val/test split (prevents leakage), train-only scaling.
- Sliding-window sequences shaped (num_seq, seq_len, num_features).

Inputs (per trading session):
  message_10.csv, orderbook_10.csv
    - If headers are missing, pass --headerless-message / --headerless-orderbook (CLI).

Outputs:
  train, val, test  — NumPy arrays with shape [num_seq, seq_len, num_features]

Feature sets:
  feature_set="core"  (5 engineered features)
    1) mid_price            = 0.5 * (ask_price_1 + bid_price_1)
    2) spread               = ask_price_1 - bid_price_1
    3) mid_log_return       = log(mid_price_t) - log(mid_price_{t-1})
    4) queue_imbalance_l1   = (bid_size_1 - ask_size_1) / (bid_size_1 + ask_size_1 + eps)
    5) depth_imbalance_l10  = (Σ_i≤10 bid_size_i - Σ_i≤10 ask_size_i) /
                              (Σ_i≤10 bid_size_i + Σ_i≤10 ask_size_i + eps)

  feature_set="raw10" (40 raw columns)
    ask_price_1..10, ask_size_1..10, bid_price_1..10, bid_size_1..10

Notes:
- Scaling is fit on TRAIN only (Standard/MinMax/None).
- Windows default to non-overlapping (stride=seq_len); set stride<seq_len for overlap.
- If your CSV headers use camel-case (e.g., AskPrice1), they’re auto-normalized.

Created by: Radhesh Goel (Keys-I) | ID: s49088276
"""
from __future__ import annotations

import os
import argparse
import shutil
from typing import Tuple, List, Literal, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ============================== Pretty printing ===============================

def _supports_color(no_color_flag: bool) -> bool:
    if no_color_flag:
        return False
    try:
        return os.isatty(1)
    except Exception:
        return False

class _C:
    def __init__(self, enabled: bool):
        n = "" if enabled else ""
        self.RESET = n
        self.DIM = "\033[2m" if enabled else ""
        self.BOLD = "\033[1m" if enabled else ""
        self.CYAN = "\033[36m" if enabled else ""
        self.YELLOW = "\033[33m" if enabled else ""
        self.GREEN = "\033[32m" if enabled else ""
        self.MAGENTA = "\033[35m" if enabled else ""
        self.BLUE = "\033[34m" if enabled else ""

def _term_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default

def _hr(width: int, c: _C) -> str:
    return f"{c.DIM}{'─'*width}{c.RESET}"

def _box(title: str, body_lines: List[str], c: _C, width: int | None = None) -> str:
    width = width or _term_width()
    border = "─" * (width - 2)
    out = [f"{c.CYAN}┌{border}┐{c.RESET}"]
    title_line = f" {title} "
    pad = max(0, width - 2 - len(title_line))
    out.append(f"{c.CYAN}│{c.RESET}{c.BOLD}{title_line}{c.RESET}{' '*pad}{c.CYAN}│{c.RESET}")
    out.append(f"{c.CYAN}├{border}┤{c.RESET}")
    for ln in body_lines:
        for sub in _wrap(ln, width - 4):
            pad = max(0, width - 4 - len(sub))
            out.append(f"{c.CYAN}│{c.RESET} {sub}{' '*pad} {c.CYAN}│{c.RESET}")
    out.append(f"{c.CYAN}└{border}┘{c.RESET}")
    return "\n".join(out)

def _wrap(s: str, width: int) -> List[str]:
    if len(s) <= width:
        return [s]
    out, cur = [], ""
    for tok in s.split(" "):
        if not cur:
            cur = tok
        elif len(cur) + 1 + len(tok) <= width:
            cur += " " + tok
        else:
            out.append(cur)
            cur = tok
    if cur:
        out.append(cur)
    return out

def _fmt_shape(arr: tuple | list | np.ndarray) -> str:
    if isinstance(arr, np.ndarray):
        return "×".join(map(str, arr.shape))
    if isinstance(arr, (tuple, list)):
        return "×".join(map(str, arr))
    return str(arr)

def _kv_lines(d: Dict[str, object]) -> List[str]:
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for sk, sv in v.items():
                lines.append(f"  {sk}: {sv}")
        else:
            lines.append(f"{k}: {v}")
    return lines


# ================================ Summaries ===================================

def _summarize_df(df: pd.DataFrame, name: str, peek: int = 5) -> List[str]:
    lines: List[str] = []
    lines.append(f"{name}")
    lines.append(f"shape: {df.shape[0]} rows × {df.shape[1]} cols")
    # columns (trim if very long)
    cols = list(df.columns)
    col_str = ", ".join(cols)
    lines.append("columns: " + col_str if len(col_str) < 160 else "columns: " + ", ".join(cols[:12]) + ", ...")
    # dtypes / NA counts (only non-zero NA counts shown)
    dtypes = df.dtypes.astype(str).to_dict()
    na_counts = {k: int(v) for k, v in df.isna().sum().items() if int(v) > 0}
    lines.append("dtypes: " + ", ".join([f"{k}:{v}" for k, v in dtypes.items()]))
    lines.append("na_counts: " + (str(na_counts) if na_counts else "{}"))
    # value counts of common message fields
    for col in ("type", "direction"):
        if col in df.columns:
            try:
                vc = df[col].value_counts(dropna=False).to_dict()
                lines.append(f"value_counts[{col}]: {vc}")
            except Exception:
                pass
    # time range + monotonic check
    if "time" in df.columns:
        try:
            t = pd.to_datetime(df["time"], errors="coerce", unit=None)
            lines.append(f"time: min={t.min()} max={t.max()}")
            if t.notna().all():
                is_mono = bool((t.diff().dropna() >= pd.Timedelta(0)).all())
                lines.append(f"time monotonic nondecreasing: {is_mono}")
        except Exception:
            pass
    # numeric quick stats (only a few cols to keep output tidy)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        sample_cols = num_cols[:6]
        desc = df[sample_cols].describe().to_dict()
        desc = {k: {m: float(v) for m, v in stats.items()} for k, stats in desc.items()}
        lines.append("describe(sample of numeric cols):")
        for k, stats in desc.items():
            stats_str = ", ".join([f"{m}={val:.4g}" for m, val in stats.items()])
            lines.append(f"  {k}: {stats_str}")
    # head / tail
    if peek > 0:
        lines.append("head:")
        lines.append(df.head(peek).to_string(index=False))
        lines.append("tail:")
        lines.append(df.tail(peek).to_string(index=False))
    return lines


# =============================== Core class ===================================

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
        feature_range: Tuple[float, float] = (0.0, 1.0),
        eps: float = 1e-8,
        headerless_message: bool = False,
        headerless_orderbook: bool = False,
        dropna: bool = True,
        output_dtype: Literal["float32", "float64"] = "float32",
        sort_by_time: bool = False,
        every: int = 1,
        clip_quantiles: Optional[Tuple[float, float]] = None,
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

        self.sort_by_time = bool(sort_by_time)
        self.every = max(1, int(every))
        self.clip_quantiles = clip_quantiles

        self._validate_splits()
        if not (self.seq_len > 0 and self.stride > 0):
            raise ValueError("seq_len and stride must be positive")

        self._scaler = None
        self._feature_names: List[str] = []
        self._row_counts: Dict[str, int] = {}
        self._clip_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (lo, hi)

    # ------------------- public API -------------------

    def load_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        msg_df, ob_df = self._load_csvs()

        if self.sort_by_time and "time" in msg_df.columns:
            order = msg_df["time"].reset_index(drop=True).sort_values().index
            msg_df = msg_df.iloc[order].reset_index(drop=True)
            ob_df = ob_df.iloc[order].reset_index(drop=True)

        self._check_alignment(msg_df, ob_df)
        feats = self._build_features(ob_df)

        if self.every > 1:
            feats = feats[::self.every]
            self._row_counts["decimated_every"] = self.every

        if self.dropna:
            feats = feats[~np.isnan(feats).any(axis=1)]
        feats = feats[np.isfinite(feats).all(axis=1)]
        self._row_counts["post_clean"] = int(feats.shape[0])

        train, val, test = self._split_chronologically(feats)
        self._row_counts.update(train=len(train), val=len(val), test=len(test))

        if self.clip_quantiles is not None:
            qmin, qmax = self.clip_quantiles
            if not (0.0 <= qmin < qmax <= 1.0):
                raise ValueError("clip_quantiles must satisfy 0 <= qmin < qmax <= 1")
            lo = np.quantile(train, qmin, axis=0)
            hi = np.quantile(train, qmax, axis=0)
            self._clip_bounds = (lo, hi)
            train = np.clip(train, lo, hi)
            val   = np.clip(val,   lo, hi)
            test  = np.clip(test,  lo, hi)

        train_s, val_s, test_s = self._scale_train_only(train, val, test)
        W_train = self._windowize(train_s)
        W_val   = self._windowize(val_s)
        W_test  = self._windowize(test_s)

        W_train = W_train.astype(self.output_dtype, copy=False)
        W_val   = W_val.astype(self.output_dtype, copy=False)
        W_test  = W_test.astype(self.output_dtype, copy=False)
        return W_train, W_val, W_test

    def summarize(self, peek: int = 5) -> List[str]:
        msg_df, ob_df = self._load_csvs()
        _ = self._normalize_orderbook_headers(
            ob_df,
            [f"ask_price_{i}" for i in range(1, 11)]
            + [f"ask_size_{i}" for i in range(1, 11)]
            + [f"bid_price_{i}" for i in range(1, 11)]
            + [f"bid_size_{i}" for i in range(1, 11)]
        )
        lines = []
        lines += _summarize_df(msg_df, "message_10.csv", peek=peek)
        lines.append("")  # spacer
        lines += _summarize_df(ob_df, "orderbook_10.csv", peek=peek)
        return lines

    def get_feature_names(self) -> List[str]:
        return list(self._feature_names)

    def get_scaler(self):
        return self._scaler

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
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
            "clip_bounds": None if self._clip_bounds is None else {
                "lo": self._clip_bounds[0].tolist(),
                "hi": self._clip_bounds[1].tolist(),
            },
            "every": self.every,
            "sorted_by_time": self.sort_by_time,
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

    def _build_features(self, ob_df: pd.DataFrame) -> np.ndarray:
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

            mid_price = 0.5 * (ap1 + bp1)
            spread = ap1 - bp1
            mid_log = np.log(np.clip(mid_price, 1e-12, None))
            mid_log_return = np.concatenate([[0.0], np.diff(mid_log)])
            qi_l1 = (bs1 - as1) / (bs1 + as1 + self.eps)
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
                "Reduce seq_len or use a longer session."
            )
        n_train = int(n * self.splits[0])
        n_val   = int(n * self.splits[1])
        n_test  = n - n_train - n_val
        if n_train < self.seq_len:
            raise ValueError(f"Train split too small ({n_train} rows) for seq_len={self.seq_len}")
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


# ============================ CLI and nice output =============================

def _print_dir_listing(path: str, c: _C) -> None:
    if os.path.isdir(path):
        files = sorted(os.listdir(path))
        lines = [f"path: {path}", f"files: {len(files)}"]
        lines += [f"  - {f}" for f in files[:12]]
        if len(files) > 12:
            lines.append(f"  ... (+{len(files)-12} more)")
    else:
        lines = [f"path: {path}", "files: (missing)"]
    print(_box("Data directory", lines, c))

def _print_summary(lines: List[str], c: _C) -> None:
    print(_box("CSV Summary", lines, c))

def _print_report(W_train, W_val, W_test, meta: Dict[str, object], c: _C) -> None:
    shapes = {
        "train windows": _fmt_shape(W_train.shape),
        "val windows": _fmt_shape(W_val.shape),
        "test windows": _fmt_shape(W_test.shape),
        "seq_len": meta.get("seq_len"),
        "stride": meta.get("stride"),
        "feature_set": meta.get("feature_set"),
        "features": len(meta.get("feature_names", [])),
        "scaler": meta.get("scaler"),
        "sorted_by_time": meta.get("sorted_by_time"),
        "every": meta.get("every"),
    }
    lines = _kv_lines(shapes)
    rc = meta.get("row_counts", {})
    if rc:
        lines.append("")
        lines.append("row_counts:")
        for k, v in rc.items():
            lines.append(f"  {k}: {v}")
    print(_box("Preprocessing Report", lines, c))

    # quick sample stats on first window (if exists)
    if getattr(W_train, "size", 0):
        win = W_train[0]
        stats = {
            "window[0] mean": f"{float(win.mean()):.5f}",
            "window[0] std": f"{float(win.std()):.5f}",
            "feature_names (first 8)": ", ".join(meta.get("feature_names", [])[:8]) + ("..." if len(meta.get("feature_names", [])) > 8 else "")
        }
        print(_box("Sample Window Stats", _kv_lines(stats), c))

def _main_cli():
    parser = argparse.ArgumentParser(description="LOBSTERData (preprocess + summarize).")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--message", required=True)
    parser.add_argument("--orderbook", required=True)
    parser.add_argument("--feature-set", choices=["core", "raw10"], default="core")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--splits", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
                        default=(0.7, 0.15, 0.15))
    parser.add_argument("--scaler", choices=["standard", "minmax", "none"], default="standard")
    parser.add_argument("--feature-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=(0.0, 1.0))
    parser.add_argument("--headerless-message", action="store_true")
    parser.add_argument("--headerless-orderbook", action="store_true")
    parser.add_argument("--no-dropna", action="store_true")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--save-npz", type=str, default=None)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--peek", type=int, default=5)
    parser.add_argument("--sort-by-time", action="store_true")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--clip-quantiles", type=float, nargs=2, metavar=("QMIN", "QMAX"), default=None)
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors in output")
    args = parser.parse_args()

    c = _C(_supports_color(args.no_color))
    _print_dir_listing(args.data_dir, c)

    loader = LOBSTERData(
        data_dir=args.data_dir,
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
        sort_by_time=args.sort_by_time,
        every=args.every,
        clip_quantiles=tuple(args.clip_quantiles) if args.clip_quantiles else None,
    )

    if args.summary:
        lines = loader.summarize(peek=args.peek)
        _print_summary(lines, c)
        return

    W_train, W_val, W_test = loader.load_arrays()
    meta = loader.get_meta()
    _print_report(W_train, W_val, W_test, meta, c)

    if args.save_npz:
        np.savez_compressed(
            args.save_npz,
            train=W_train, val=W_val, test=W_test,
            feature_names=np.array(loader.get_feature_names(), dtype=object),
            meta=np.array([str(meta)], dtype=object),
        )
        print(_box("Saved", [f"path: {args.save_npz}"], c))


if __name__ == "__main__":
    _main_cli()