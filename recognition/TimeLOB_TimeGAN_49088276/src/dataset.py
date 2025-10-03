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
import re
import argparse
import shutil
from datetime import datetime
from typing import Tuple, List, Literal, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tabulate import tabulate


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
        self.enabled = enabled
        self.RESET  = "\033[0m"  if enabled else ""
        self.DIM    = "\033[2m"  if enabled else ""
        self.BOLD   = "\033[1m"  if enabled else ""
        self.CYAN   = "\033[36m" if enabled else ""
        self.YELLOW = "\033[33m" if enabled else ""
        self.GREEN  = "\033[32m" if enabled else ""
        self.MAGENTA= "\033[35m" if enabled else ""
        self.BLUE   = "\033[34m" if enabled else ""

def _term_width(default: int = 96) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default

def _wrap(s: str, width: int) -> list[str]:
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

# ---- detect and preserve tabulate tables inside panels/bubbles ----

_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def _visible_len(s: str) -> int:
    """Visible length without ANSI codes (so width calc matches terminal)."""
    return len(_ANSI_RE.sub("", s))

def _is_table_line(s: str) -> bool:
    """
    Heuristic for tabulate-like lines we should not wrap:
    - GitHub style: lines starting with '|' and having columns separated by '|'
    - Grid style: rule lines with '+' borders
    - Simple header/rule lines made of '-:|+ '
    """
    t = s.strip()
    if not t:
        return False
    if t.startswith("|") and "|" in t[1:]:
        return True
    if t.startswith("+") and t.endswith("+"):
        return True
    if set(t) <= set("-:|+ "):
        return True
    return False

def _kv_table(rows: list[tuple[str, str]], width: int, pad: int = 2) -> list[str]:
    """
    Render key–value rows as a compact 2-col table using tabulate.
    Returns a list of lines to embed inside bubbles/boxes.
    """
    if not rows:
        return []
    table = tabulate(rows, headers=["key", "value"], tablefmt="github", stralign="left")
    return table.splitlines()

def _bubble(title: str, body_lines: list[str], c: _C, align: str = "left", width: int | None = None) -> str:
    """
    Render a chat-style message bubble.
    - Does NOT wrap lines that look like preformatted tables.
    - Auto-fits inner width to the widest table line (within terminal limit).
    """
    termw = _term_width()
    width = min(termw, width or termw)

    # Baseline inner width
    base_inner = max(24, width - 10)

    # If there are preformatted table lines, fit to the widest visible line
    widest_tbl = 0
    for ln in body_lines:
        if _is_table_line(ln):
            widest_tbl = max(widest_tbl, _visible_len(ln))
    max_inner = min(max(base_inner, widest_tbl), width - 10)

    # Left/right alignment
    indent = 2 if align == "left" else max(2, width - (max_inner + 8))
    pad = " " * indent

    # Header
    ts = datetime.now().strftime("%H:%M")
    head = f"{c.BOLD}{title}{c.RESET}  {c.DIM}{ts}{c.RESET}"
    head_lines = _wrap(head, max_inner)
    lines = [pad + " " + head_lines[0]]
    for hl in head_lines[1:]:
        lines.append(pad + " " + hl)

    # Bubble top border
    lines.append(pad + "  " + ("╭" + "─" * (max_inner + 2) + "╮"))

    # Body: keep table lines intact; wrap normal text
    for ln in body_lines:
        if _is_table_line(ln):
            vis = _visible_len(ln)
            if vis <= max_inner:
                out = ln + " " * (max_inner - vis)
            else:
                out = ln[:max_inner]
            lines.append(pad + "  " + "│ " + out + " │")
        else:
            for wln in _wrap(ln, max_inner):
                lines.append(pad + "  " + "│ " + wln.ljust(max_inner) + " │")

    # Bubble bottom + tail
    tail_left  = pad + "  " + "╰" + "─" * (max_inner + 2) + "╯" + "⟋"
    tail_right = pad + " "  + "⟍" + "╰" + "─" * (max_inner + 2) + "╯"
    lines.append(tail_left if align == "left" else tail_right)
    return "\n".join(lines)

def _panel(title: str, body_lines: list[str], c: _C, width: int | None = None) -> str:
    """Box panel; does not wrap tabulated lines; auto-fits to widest table row."""
    termw = _term_width()
    width = width or termw
    inner = width - 4  # borders + spaces

    # Fit inner width to widest table line if present (within terminal width)
    widest_tbl = 0
    for ln in body_lines:
        if _is_table_line(ln):
            widest_tbl = max(widest_tbl, _visible_len(ln))
    inner = min(max(inner, widest_tbl), termw - 4)
    width = inner + 4

    border = "─" * (width - 2)
    out = [f"{c.CYAN}┌{border}┐{c.RESET}"]
    title_line = f" {title} "
    pad = max(0, width - 2 - len(title_line))
    out.append(f"{c.CYAN}│{c.RESET}{c.BOLD}{title_line}{c.RESET}{' '*pad}{c.CYAN}│{c.RESET}")
    out.append(f"{c.CYAN}├{border}┤{c.RESET}")

    for ln in body_lines:
        if _is_table_line(ln):
            vis = _visible_len(ln)
            # inner-2 for side spaces inside the box content
            width_ok = inner - 2
            if vis <= width_ok:
                body = ln + " " * (width_ok - vis)
            else:
                body = ln[:width_ok]
            out.append(f"{c.CYAN}│{c.RESET} {body} {c.CYAN}│{c.RESET}")
        else:
            for sub in _wrap(ln, inner - 2):
                padlen = max(0, (inner - 2) - len(sub))
                out.append(f"{c.CYAN}│{c.RESET} {sub}{' '*padlen} {c.CYAN}│{c.RESET}")

    out.append(f"{c.CYAN}└{border}┘{c.RESET}")
    return "\n".join(out)

def _render_card(title: str, body_lines: list[str], c: _C, style: str = "chat", align: str = "left") -> str:
    return _bubble(title, body_lines, c, align=align) if style == "chat" else _panel(title, body_lines, c)


# ============================== Verbose helpers ===============================

def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"

def _first_last_time(msg_df: pd.DataFrame) -> tuple[str, str]:
    if "time" not in msg_df.columns:
        return ("", "")
    try:
        t = pd.to_datetime(msg_df["time"], errors="coerce", unit=None)
        return (str(t.min()), str(t.max()))
    except Exception:
        return ("", "")


# ================================ Summaries ===================================

def _summarize_df(df: pd.DataFrame, name: str, peek: int = 5) -> List[str]:
    lines: List[str] = []
    lines.append(f"{name}")
    lines.append(f"shape: {df.shape[0]} rows × {df.shape[1]} cols")
    cols = list(df.columns)
    col_str = ", ".join(cols)
    lines.append("columns: " + col_str if len(col_str) < 160 else "columns: " + ", ".join(cols[:12]) + ", …")
    dtypes = df.dtypes.astype(str).to_dict()
    na_counts = {k: int(v) for k, v in df.isna().sum().items() if int(v) > 0}
    lines.append("dtypes: " + ", ".join([f"{k}:{v}" for k, v in dtypes.items()]))
    lines.append("na_counts: " + (str(na_counts) if na_counts else "{}"))
    for col in ("type", "direction"):
        if col in df.columns:
            try:
                vc = df[col].value_counts(dropna=False).to_dict()
                lines.append(f"value_counts[{col}]: {vc}")
            except Exception:
                pass
    if "time" in df.columns:
        try:
            t = pd.to_datetime(df["time"], errors="coerce", unit=None)
            lines.append(f"time: min={t.min()} max={t.max()}")
            if t.notna().all():
                is_mono = bool((t.diff().dropna() >= pd.Timedelta(0)).all())
                lines.append(f"time monotonic nondecreasing: {is_mono}")
        except Exception:
            pass

    # numeric quick stats (pretty table)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        sample_cols = num_cols[: min(8, len(num_cols))]
        desc_df = df[sample_cols].describe().round(6)
        lines.append("describe(sample numeric cols):")
        lines.extend(tabulate(desc_df, headers="keys", tablefmt="github").splitlines())

    # head / tail (pretty tables)
    if peek > 0:
        lines.append("head:")
        head_tbl = tabulate(df.head(peek), headers="keys", tablefmt="github", showindex=False)
        lines.extend(head_tbl.splitlines())
        lines.append("tail:")
        tail_tbl = tabulate(df.tail(peek), headers="keys", tablefmt="github", showindex=False)
        lines.extend(tail_tbl.splitlines())

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
        lines.append("")  # spacer between the two tables
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


# ============================ CLI and message output ==========================

def _print_dir_listing(path: str, c: _C, style: str) -> None:
    if os.path.isdir(path):
        files = sorted(os.listdir(path))
        body = [f"path: {path}", f"files: {len(files)}"]
        body += [f"• {f}" for f in files[:10]]
        if len(files) > 10:
            body.append(f"• (+{len(files)-10} more)")
    else:
        body = [f"path: {path}", "files: (missing)"]
    print(_render_card("Data directory", body, c, style=style, align="left"))

def _print_summary(lines: list[str], c: _C, style: str) -> None:
    # split into two bubbles by blank line
    if "" in lines:
        idx = lines.index("")
        msg_part = lines[:idx]
        ob_part  = lines[idx+1:]
    else:
        msg_part, ob_part = lines, []

    def split_title(block: list[str]) -> tuple[str, list[str]]:
        if not block:
            return ("", [])
        title, body = block[0], block[1:]
        return (title, body)

    t1, b1 = split_title(msg_part)
    if t1:
        print(_render_card(f"{t1}", b1, c, style=style, align="left"))
    t2, b2 = split_title(ob_part)
    if t2:
        print(_render_card(f"{t2}", b2, c, style=style, align="left"))

def _print_report(W_train, W_val, W_test, meta: dict, c: _C, style: str, *,
                  verbose: bool = False,
                  scaler_obj = None,
                  clip_bounds = None,
                  time_coverage: tuple[str, str] = ("","")) -> None:
    # Basic block
    block1 = [
        ("train windows", "×".join(map(str, W_train.shape))),
        ("val windows",   "×".join(map(str, W_val.shape))),
        ("test windows",  "×".join(map(str, W_test.shape))),
        ("seq_len",       str(meta.get("seq_len"))),
        ("stride",        str(meta.get("stride"))),
        ("feature_set",   str(meta.get("feature_set"))),
        ("#features",     str(len(meta.get("feature_names", [])))),
        ("scaler",        str(meta.get("scaler"))),
        ("sorted_by_time",str(meta.get("sorted_by_time"))),
        ("every",         str(meta.get("every"))),
    ]
    lines1 = _kv_table(block1, width=min(_term_width(), 84))
    print(_render_card("Preprocessing report", lines1, c, style=style, align="right"))

    # Row counts
    rc = meta.get("row_counts", {})
    if rc:
        block2 = [(k, str(v)) for k, v in rc.items()]
        lines2 = _kv_table(block2, width=min(_term_width(), 84))
        print(_render_card("Row counts", lines2, c, style=style, align="right"))

    # Sample window stats
    if getattr(W_train, "size", 0):
        win = W_train[0]
        block3 = [
            ("window[0] mean", f"{float(win.mean()):.6f}"),
            ("window[0] std",  f"{float(win.std()):.6f}"),
            ("features", ", ".join(meta.get("feature_names", [])[:8]) + ("…" if len(meta.get("feature_names", []))>8 else "")),
        ]
        lines3 = _kv_table(block3, width=min(_term_width(), 84))
        print(_render_card("Sample window", lines3, c, style=style, align="right"))

    if not verbose:
        return

    # Verbose extras
    vlines: list[str] = []
    # Memory footprint
    total_bytes = (W_train.nbytes if hasattr(W_train, "nbytes") else 0) + \
                  (W_val.nbytes   if hasattr(W_val, "nbytes")   else 0) + \
                  (W_test.nbytes  if hasattr(W_test, "nbytes")  else 0)
    vlines.append(f"memory total: {_fmt_bytes(total_bytes)}")
    vlines.append(f"train bytes: {_fmt_bytes(getattr(W_train, 'nbytes', 0))}")
    vlines.append(f"val bytes:   {_fmt_bytes(getattr(W_val, 'nbytes', 0))}")
    vlines.append(f"test bytes:  {_fmt_bytes(getattr(W_test, 'nbytes', 0))}")

    # Time coverage if available
    tmin, tmax = time_coverage
    if tmin or tmax:
        vlines.append(f"time coverage: {tmin}  →  {tmax}")

    print(_render_card("Resources & coverage", vlines, c, style=style, align="right"))

    # Scaler params
    if scaler_obj is not None:
        s_lines = []
        if hasattr(scaler_obj, "mean_") and hasattr(scaler_obj, "scale_"):
            # StandardScaler
            means = scaler_obj.mean_
            scales = scaler_obj.scale_
            s_lines += _kv_table([
                ("type", "StandardScaler"),
                ("mean[0:8]",  np.array2string(means[:8], precision=4, separator=", ")),
                ("scale[0:8]", np.array2string(scales[:8], precision=4, separator=", ")),
            ], width=min(_term_width(), 84))
        elif hasattr(scaler_obj, "data_min_") and hasattr(scaler_obj, "data_max_"):
            # MinMaxScaler
            s_lines += _kv_table([
                ("type", "MinMaxScaler"),
                ("data_min[0:8]", np.array2string(scaler_obj.data_min_[:8], precision=4, separator=", ")),
                ("data_max[0:8]", np.array2string(scaler_obj.data_max_[:8], precision=4, separator=", ")),
                ("feature_range", str(getattr(scaler_obj, "feature_range", None))),
            ], width=min(_term_width(), 84))
        if s_lines:
            print(_render_card("Scaler parameters", s_lines, c, style=style, align="right"))

    # Clip bounds preview
    if clip_bounds is not None:
        lo, hi = clip_bounds
        cb_lines = _kv_table([
            ("q-lo[0:8]", np.array2string(lo[:8], precision=4, separator=", ")),
            ("q-hi[0:8]", np.array2string(hi[:8], precision=4, separator=", ")),
        ], width=min(_term_width(), 84))
        print(_render_card("Clip bounds (preview)", cb_lines, c, style=style, align="right"))

    # Per-split window counts and overlap ratio
    def _count_windows(n_rows: int, seq_len: int, stride: int) -> int:
        if n_rows < seq_len:
            return 0
        return 1 + (n_rows - seq_len) // stride

    rc_train = rc.get("train", 0)
    rc_val   = rc.get("val", 0)
    rc_test  = rc.get("test", 0)
    overlap = 1.0 - (meta.get("stride", 1) / max(1, meta.get("seq_len", 1)))
    perf = _kv_table([
        ("expected train windows", str(_count_windows(rc_train, meta.get("seq_len", 0), meta.get("stride", 1)))),
        ("expected val windows",   str(_count_windows(rc_val,   meta.get("seq_len", 0), meta.get("stride", 1)))),
        ("expected test windows",  str(_count_windows(rc_test,  meta.get("seq_len", 0), meta.get("stride", 1)))),
        ("overlap ratio",          f"{overlap:.3f}"),
    ], width=min(_term_width(), 84))
    print(_render_card("Windowing details", perf, c, style=style, align="right"))


# ========================== Dataset info (report card) ========================

def _print_dataset_info(loader: "LOBSTERData", c: _C, style: str, peek: int = 5) -> None:
    """Print detailed information about the dataset and feature set."""
    meta = loader.get_meta()
    feature_set = meta.get("feature_set")
    feats = meta.get("feature_names") or []

    # Fallback feature names if meta is empty
    if not feats:
        if feature_set == "core":
            feats = [
                "mid_price",
                "spread",
                "mid_log_return",
                "queue_imbalance_l1",
                "depth_imbalance_l10",
            ]
        elif feature_set == "raw10":
            feats = (
                [f"ask_price_{i}" for i in range(1, 11)] +
                [f"ask_size_{i}"  for i in range(1, 11)] +
                [f"bid_price_{i}" for i in range(1, 11)] +
                [f"bid_size_{i}"  for i in range(1, 11)]
            )

    lines: List[str] = [
        f"Feature set: {feature_set}",
        f"Total features: {len(feats)}",
        ""
    ]

    # aggregated statistics across splits (pretty tables)
    try:
        W_train, W_val, W_test = loader.load_arrays()
        if W_train.size + W_val.size + W_test.size == 0:
            raise ValueError("No windows produced; consider lowering seq_len or stride.")
        blocks = []
        for W in (W_train, W_val, W_test):
            if getattr(W, "size", 0):
                blocks.append(W.reshape(-1, W.shape[-1]))
        all_data = np.concatenate(blocks, axis=0)
        df = pd.DataFrame(all_data, columns=feats)

        # describe()
        lines.append("Statistical summary (aggregated across splits):")
        desc_df = df.describe().round(6)
        lines.extend(tabulate(desc_df, headers="keys", tablefmt="github").splitlines())
        lines.append("")

        # peaks: means and stds tables
        means = df.mean().sort_values(ascending=False).head(5)
        stds  = df.std().sort_values(ascending=False).head(5)

        lines.append("Highest-mean features:")
        lines.extend(tabulate(list(means.items()), headers=["feature", "mean"], tablefmt="github").splitlines())
        lines.append("")

        lines.append("Most-variable features (by std):")
        lines.extend(tabulate(list(stds.items()), headers=["feature", "std"], tablefmt="github").splitlines())
        lines.append("")

        # example rows
        lines.append("Example rows (first few timesteps):")
        ex_tbl = tabulate(df.head(peek).round(6), headers="keys", tablefmt="github", showindex=True)
        lines.extend(ex_tbl.splitlines())

    except Exception as e:
        lines.append(f"(Could not compute stats: {e})")

    print(_render_card("Dataset summary", lines, c, style=style, align="left"))


# ================================== CLI ======================================

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
    parser.add_argument("--style", choices=["chat", "box"], default="chat", help="Output style")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors in output")
    parser.add_argument("--verbose", action="store_true", help="Print extra diagnostics (memory, scaler, clip bounds)")
    parser.add_argument("--meta-json", type=str, default=None, help="Optional path to dump meta JSON")
    args = parser.parse_args()

    c = _C(_supports_color(args.no_color))
    _print_dir_listing(args.data_dir, c, style=args.style)

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
        _print_summary(lines, c, style=args.style)
        _print_dataset_info(loader, c, style=args.style, peek=args.peek)
        return

    W_train, W_val, W_test = loader.load_arrays()
    meta = loader.get_meta()

    # verbose context
    scaler_obj = loader.get_scaler()
    clip_bounds = None
    if meta.get("clip_bounds"):
        lo = np.array(meta["clip_bounds"]["lo"], dtype=float)
        hi = np.array(meta["clip_bounds"]["hi"], dtype=float)
        clip_bounds = (lo, hi)

    # best-effort message time coverage
    try:
        msg_df, _ = loader._load_csvs()
        tmin, tmax = _first_last_time(msg_df)
    except Exception:
        tmin = tmax = ""

    _print_report(
        W_train, W_val, W_test, meta, c, style=args.style,
        verbose=args.verbose, scaler_obj=scaler_obj,
        clip_bounds=clip_bounds, time_coverage=(tmin, tmax)
    )

    # optional meta dump
    if args.meta_json:
        import json
        with open(args.meta_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(_render_card("Saved", [f"meta: {args.meta_json}"], c, style=args.style, align="right"))

    # optional arrays NPZ
    if args.save_npz:
        np.savez_compressed(
            args.save_npz,
            train=W_train, val=W_val, test=W_test,
            feature_names=np.array(loader.get_feature_names(), dtype=object),
            meta=np.array([str(meta)], dtype=object),
        )
        print(_render_card("Saved", [f"windows: {args.save_npz}"], c, style=args.style, align="right"))


if __name__ == "__main__":
    _main_cli()
