"""
LOBSTER (Level-10) preprocessing for TimeGAN.

- Loads paired LOBSTER CSVs (message_10.csv, orderbook_10.csv), aligned by event index.
- Builds either a compact engineered 5-feature set ("core") or raw level-10 depth ("raw10").
- Chronological train/val/test split (prevents leakage), train-only scaling.
- Sliding-window sequences shaped (num_seq, seq_len, num_features).

Inputs (per trading session):
  message_10.csv, orderbook_10.csv
    - If headers are missing, pass --headerless-message / --headerless-orderbook (CLI),
      but auto-detection now assigns canonical headers when omitted.

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
- Scaling is fit on TRAIN only (Standard/MinMax/None). Advanced scalers: Robust, Quantile, Power.
- Optional whitening: PCA (variance threshold) or ZCA.
- Optional train-only sequence augmentations (jitter, scaling, time-warp) for GANs.
- Windows default to non-overlapping (stride=seq_len); set stride<seq_len for overlap.
- CamelCase headers (e.g., AskPrice1) auto-normalize.
- Headerless CSVs are auto-detected and canonical headers applied.

Created by: Radhesh Goel (Keys-I) | ID: s49088276
"""
from __future__ import annotations

import os
from typing import Tuple, List, Literal, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.decomposition import PCA
import json
try:
    import joblib  # optional persistence
except Exception:
    joblib = None


class LOBSTERData:
    """
    Loader → features → windows → splits for LOBSTER Level-10 data.

    Feature sets:
      - "core": engineered 5-feature set (+ optional extras)
      - "raw10": 40 raw columns (ask/bid price/size × levels 1..10) (+ optional extras)
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
        scaler: Literal["standard", "minmax", "robust", "quantile", "power", "none"] = "standard",
        feature_range: Tuple[float, float] = (0.0, 1.0),
        eps: float = 1e-8,
        headerless_message: bool = False,
        headerless_orderbook: bool = False,
        dropna: bool = True,
        output_dtype: Literal["float32", "float64"] = "float32",
        sort_by_time: bool = False,
        every: int = 1,
        clip_quantiles: Optional[Tuple[float, float]] = None,

        # --- extra feature engineering knobs ---
        add_rel_spread: bool = True,
        add_microprice: bool = True,
        add_imbalance_l5: bool = True,
        add_roll_stats: bool = True,
        roll_window: int = 64,
        add_diff1: bool = True,
        add_pct_change: bool = False,

        # --- whitening / dimensionality reduction ---
        whiten: Optional[Literal["pca", "zca"]] = None,
        pca_var: float = 0.99,

        # --- train-only augmentation for GANs ---
        aug_prob: float = 0.0,
        aug_jitter_std: float = 0.01,
        aug_scaling_std: float = 0.05,
        aug_timewarp_max: float = 0.1,

        # --- persistence ---
        save_dir: Optional[str] = None,
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

        # feature knobs
        self.add_rel_spread   = add_rel_spread
        self.add_microprice   = add_microprice
        self.add_imbalance_l5 = add_imbalance_l5
        self.add_roll_stats   = add_roll_stats
        self.roll_window      = int(roll_window)
        self.add_diff1        = add_diff1
        self.add_pct_change   = add_pct_change

        # whitening/DR
        self.whiten   = whiten
        self.pca_var  = float(pca_var)
        self._pca     = None  # set later
        self._zca_cov = None  # (mean, whitening_mat)

        # augmentation
        self.aug_prob         = float(aug_prob)
        self.aug_jitter_std   = float(aug_jitter_std)
        self.aug_scaling_std  = float(aug_scaling_std)
        self.aug_timewarp_max = float(aug_timewarp_max)

        # save
        self.save_dir = save_dir

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

        # enforce numeric types early (prevents string pollution)
        for col in ("time", "order_id", "size", "price"):
            if col in msg_df.columns:
                msg_df[col] = pd.to_numeric(msg_df[col], errors="coerce")
        ob_df[ob_df.columns] = ob_df[ob_df.columns].apply(pd.to_numeric, errors="coerce")

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

        # train-only augmentations for GANs
        W_train = self._augment_windows(W_train)

        W_train = W_train.astype(self.output_dtype, copy=False)
        W_val   = W_val.astype(self.output_dtype, copy=False)
        W_test  = W_test.astype(self.output_dtype, copy=False)

        # optional persistence
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(self.save_dir, "windows.npz"),
                train=W_train, val=W_val, test=W_test
            )
            meta = self.get_meta()
            meta["whiten"] = self.whiten
            meta["pca_var"] = self.pca_var
            meta["aug"] = {
                "prob": self.aug_prob, "jitter_std": self.aug_jitter_std,
                "scaling_std": self.aug_scaling_std, "timewarp_max": self.aug_timewarp_max
            }
            with open(os.path.join(self.save_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            if joblib is not None and self._scaler is not None:
                joblib.dump(self._scaler, os.path.join(self.save_dir, "scaler.pkl"))
            if joblib is not None and self._pca is not None:
                joblib.dump(self._pca, os.path.join(self.save_dir, "pca.pkl"))
            if joblib is not None and self._zca_cov is not None:
                joblib.dump(self._zca_cov, os.path.join(self.save_dir, "zca.pkl"))

        return W_train, W_val, W_test

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
            "scaler": (type(self._scaler).__name__ if self._scaler is not None else "None"),
            "row_counts": self._row_counts,
            "clip_bounds": None if self._clip_bounds is None else {
                "lo": self._clip_bounds[0].tolist(),
                "hi": self._clip_bounds[1].tolist(),
            },
            "every": self.every,
            "sorted_by_time": self.sort_by_time,
            "whiten": self.whiten,
            "pca_var": self.pca_var,
        }

    # ------------------- internals --------------------

    def _validate_splits(self) -> None:
        s = sum(self.splits)
        if not (abs(s - 1.0) < 1e-12):
            raise ValueError(f"splits must sum to 1.0, got {self.splits} (sum={s})")
        if any(x < 0 for x in self.splits):
            raise ValueError("splits cannot be negative")

    # ---- header detection helpers ----
    def _looks_headerless(self, path: str, expected_cols: int, min_numeric: int) -> bool:
        """
        Peek the first row with header=None. If the row is mostly numeric and the
        column count matches what we expect, assume there's NO header.
        """
        try:
            df0 = pd.read_csv(path, header=None, nrows=1)
        except Exception:
            return False
        if df0.shape[1] != expected_cols:
            return False
        num_ok = pd.to_numeric(df0.iloc[0], errors="coerce").notna().sum()
        return num_ok >= min_numeric

    def _read_with_possible_headerless(self, path: str, default_names: list[str],
                                       force_headerless: bool,
                                       normalize_fn=None) -> pd.DataFrame:
        """
        Read CSV, auto-detect headerlessness if not forced.
        - If forced: header=None, names=default_names
        - Else: if first row looks numeric & count matches, treat as headerless.
                otherwise try header=0 and optionally normalize columns.
        """
        expected_cols = len(default_names)
        if force_headerless:
            return pd.read_csv(path, header=None, names=default_names)

        # Auto-detect headerless
        if self._looks_headerless(path, expected_cols=expected_cols,
                                  min_numeric=max(4, int(0.6 * expected_cols))):  # threshold 60%
            return pd.read_csv(path, header=None, names=default_names)

        # Try with header row, then normalize if asked
        df = pd.read_csv(path)
        if normalize_fn is not None:
            df = normalize_fn(df, default_names)

        # If counts match but names/order differ, force canonical order & names
        if df.shape[1] == expected_cols and list(df.columns) != default_names:
            df = df.iloc[:, :expected_cols]  # ensure width
            df.columns = [str(c) for c in df.columns]
            # If normalize_fn was provided, it likely already tried to normalize.
            df.columns = default_names
        return df

    def _load_csvs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not os.path.isfile(self.orderbook_path):
            raise FileNotFoundError(f"Missing {self.orderbook_path}")
        if not os.path.isfile(self.message_path):
            raise FileNotFoundError(f"Missing {self.message_path}")

        # Message (6 columns)
        msg_cols = ["time", "type", "order_id", "size", "price", "direction"]
        msg_df = self._read_with_possible_headerless(
            self.message_path,
            default_names=msg_cols,
            force_headerless=self.headerless_message,
            normalize_fn=lambda df, _: (
                df.assign(**{}).rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
            )
        )
        # Enforce exact column order when shape matches but order differs
        if msg_df.shape[1] == 6 and list(msg_df.columns) != msg_cols:
            # Try reorder if all present; else force names in canonical order
            present = set(msg_df.columns)
            if set(msg_cols).issubset(present):
                msg_df = msg_df[msg_cols]
            msg_df.columns = msg_cols

        # Orderbook (40 columns)
        ob_cols = (
            [f"ask_price_{i}" for i in range(1, 11)] +
            [f"ask_size_{i}"  for i in range(1, 11)] +
            [f"bid_price_{i}" for i in range(1, 11)] +
            [f"bid_size_{i}"  for i in range(1, 11)]
        )
        ob_df = self._read_with_possible_headerless(
            self.orderbook_path,
            default_names=ob_cols,
            force_headerless=self.headerless_orderbook,
            normalize_fn=lambda df, target: self._normalize_orderbook_headers(df, target)
        )
        # Enforce exact column order when counts match but order differs
        if ob_df.shape[1] == len(ob_cols) and list(ob_df.columns) != ob_cols:
            if set(ob_cols).issubset(set(ob_df.columns)):
                ob_df = ob_df[ob_cols]
            ob_df.columns = ob_cols

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

    # ------ extra engineering helpers ------
    def _engineer_extra(self, ob_df: pd.DataFrame, base: np.ndarray) -> np.ndarray:
        """Append engineered features onto base matrix (N x d)."""
        feats = [base]

        ap1 = ob_df["ask_price_1"].to_numpy(np.float64)
        bp1 = ob_df["bid_price_1"].to_numpy(np.float64)
        as1 = ob_df["ask_size_1"].to_numpy(np.float64)
        bs1 = ob_df["bid_size_1"].to_numpy(np.float64)

        mid_price = 0.5 * (ap1 + bp1)
        spread = ap1 - bp1

        if self.add_rel_spread:
            rel_spread = spread / (mid_price + self.eps)
            feats.append(rel_spread[:, None])

        if self.add_microprice:
            # microprice using L1 sizes
            w_bid = bs1 / (bs1 + as1 + self.eps)
            w_ask = 1.0 - w_bid
            micro = w_ask * ap1 + w_bid * bp1
            feats.append(micro[:, None])

        if self.add_imbalance_l5:
            bid5 = np.sum([ob_df[f"bid_size_{i}"].to_numpy(np.float64) for i in range(1, 6)], axis=0)
            ask5 = np.sum([ob_df[f"ask_size_{i}"].to_numpy(np.float64) for i in range(1, 6)], axis=0)
            im5  = (bid5 - ask5) / (bid5 + ask5 + self.eps)
            feats.append(im5[:, None])

        if self.add_diff1:
            diff = np.vstack([np.zeros((1, base.shape[1])), np.diff(base, axis=0)])
            feats.append(diff)

        if self.add_pct_change:
            pct = np.zeros_like(base)
            pct[1:] = (base[1:] - base[:-1]) / (np.abs(base[:-1]) + self.eps)
            feats.append(pct)

        if self.add_roll_stats:
            W = max(2, int(self.roll_window))
            roll_mean = pd.Series(mid_price).rolling(W, min_periods=1).mean().to_numpy()
            roll_std  = pd.Series(mid_price).rolling(W, min_periods=1).std(ddof=0).fillna(0.0).to_numpy()
            vol = pd.Series(np.diff(np.log(np.clip(mid_price, 1e-12, None)), prepend=0.0) ** 2).rolling(W, min_periods=1).mean().to_numpy()
            feats += [roll_mean[:, None], roll_std[:, None], vol[:, None]]

        return np.concatenate(feats, axis=1)

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
            X = self._engineer_extra(ob_df, X)
            extras = []
            if self.add_rel_spread:   extras.append("rel_spread")
            if self.add_microprice:   extras.append("microprice")
            if self.add_imbalance_l5: extras.append("depth_imbalance_l5")
            if self.add_diff1:        extras += [f"diff1_{n}" for n in self._feature_names]
            if self.add_pct_change:   extras += [f"pct_{n}" for n in self._feature_names]
            if self.add_roll_stats:   extras += ["roll_mid_mean","roll_mid_std","roll_vol"]
            self._feature_names = self._feature_names + extras
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

            X_base = np.vstack([mid_price, spread, mid_log_return, qi_l1, di_l10]).T
            base_names = [
                "mid_price",
                "spread",
                "mid_log_return",
                "queue_imbalance_l1",
                "depth_imbalance_l10",
            ]
            X = self._engineer_extra(ob_df, X_base)

            extra_names = []
            if self.add_rel_spread:   extra_names.append("rel_spread")
            if self.add_microprice:   extra_names.append("microprice")
            if self.add_imbalance_l5: extra_names.append("depth_imbalance_l5")
            if self.add_diff1:        extra_names += [f"diff1_{n}" for n in base_names]
            if self.add_pct_change:   extra_names += [f"pct_{n}" for n in base_names]
            if self.add_roll_stats:   extra_names += ["roll_mid_mean","roll_mid_std","roll_vol"]

            self._feature_names = base_names + extra_names
            return X

        raise ValueError("feature_set must be 'core' or 'raw10'")

    def _split_chronologically(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        if n < self.seq_len:
            raise ValueError(
                f"Not enough rows ({n}) for seq_len={self.seq_len}. Reduce seq_len or use a longer session."
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
        kind = self.scaler_kind
        if kind == "none":
            scaler = None
            Xt, Xv, Xs = train, val, test
        else:
            if kind == "standard":
                scaler = StandardScaler()
            elif kind == "minmax":
                scaler = MinMaxScaler(feature_range=self.feature_range)
            elif kind == "robust":
                scaler = RobustScaler()
            elif kind == "quantile":
                scaler = QuantileTransformer(output_distribution="normal", subsample=100000, random_state=42)
            elif kind == "power":
                scaler = PowerTransformer(method="yeo-johnson", standardize=True)
            else:
                raise ValueError("scaler must be 'standard','minmax','robust','quantile','power', or 'none'")
            scaler.fit(train)
            Xt, Xv, Xs = scaler.transform(train), scaler.transform(val), scaler.transform(test)

        self._scaler = scaler

        # optional whitening
        if self.whiten is None:
            return Xt, Xv, Xs

        if self.whiten == "pca":
            p = PCA(n_components=self.pca_var, svd_solver="full", whiten=True, random_state=42)
            p.fit(Xt)
            self._pca = p
            return p.transform(Xt), p.transform(Xv), p.transform(Xs)

        if self.whiten == "zca":
            mu = Xt.mean(axis=0, keepdims=True)
            Xc = Xt - mu
            cov = (Xc.T @ Xc) / max(1, Xc.shape[0]-1)
            U, S, _ = np.linalg.svd(cov + 1e-6*np.eye(cov.shape[0]), full_matrices=False)
            S_inv_sqrt = np.diag(1.0 / np.sqrt(S + 1e-6))
            W = U @ S_inv_sqrt @ U.T
            self._zca_cov = (mu, W)

            def apply_zca(A: np.ndarray) -> np.ndarray:
                return (A - mu) @ W

            return apply_zca(Xt), apply_zca(Xv), apply_zca(Xs)

        raise ValueError("whiten must be None, 'pca', or 'zca'")

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

    # ------ augmentations (sequence-level, applied after windowing to TRAIN only) ------
    def _augment_windows(self, W: np.ndarray) -> np.ndarray:
        if self.aug_prob <= 0.0:
            return W
        out = W.copy()
        rng = np.random.default_rng(42)
        for i in range(out.shape[0]):
            if rng.random() < self.aug_prob:
                seq = out[i]
                # jitter (add Gaussian noise)
                seq = seq + rng.normal(0.0, self.aug_jitter_std, size=seq.shape)
                # scaling (per-feature)
                scale = rng.normal(1.0, self.aug_scaling_std, size=(1, seq.shape[-1]))
                seq = seq * scale
                # simple time warp (resample along time axis by a small factor)
                max_alpha = self.aug_timewarp_max
                alpha = float(np.clip(rng.normal(1.0, max_alpha/3), 1.0-max_alpha, 1.0+max_alpha))
                T, D = seq.shape
                new_idx = np.linspace(0, T-1, num=T) ** alpha
                new_idx = (new_idx / new_idx.max()) * (T-1)
                left = np.floor(new_idx).astype(int)
                right = np.clip(left+1, 0, T-1)
                w = (new_idx - left)[:, None]
                seq = (1-w) * seq[left, :] + w * seq[right, :]
                out[i] = seq
        return out


if __name__ == "__main__":
    # Demo / summary with styled box panels by default
    import argparse

    from helpers.textui import (
        C, supports_color, set_table_style,
        render_kv_panel, render_card, table, DEFAULT_STYLE
    )

    parser = argparse.ArgumentParser(description="Run dataset preprocessing demo or print a quick summary.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--message", default="message_10.csv")
    parser.add_argument("--orderbook", default="orderbook_10.csv")
    parser.add_argument("--feature-set", choices=["core", "raw10"], default="core")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--scaler", choices=["standard", "minmax", "robust", "quantile", "power", "none"], default="standard")
    parser.add_argument("--splits", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"), default=(0.7, 0.15, 0.15))
    parser.add_argument("--headerless-message", action="store_true")
    parser.add_argument("--headerless-orderbook", action="store_true")

    # style & summary controls
    parser.add_argument("--summary", action="store_true", help="Print a concise dataset summary (heads/dtypes/stats).")
    parser.add_argument("--peek", type=int, default=5, help="Rows to show for head/tail in --summary mode.")
    parser.add_argument("--style", choices=["box", "chat"], default=DEFAULT_STYLE, help="Output card style (default: box).")
    parser.add_argument("--table-style", choices=["github", "grid", "simple"], default="github", help="Tabulate table style.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")

    # extra feature engineering
    parser.add_argument("--no-rel-spread", dest="add_rel_spread", action="store_false")
    parser.add_argument("--no-microprice", dest="add_microprice", action="store_false")
    parser.add_argument("--no-imbalance-l5", dest="add_imbalance_l5", action="store_false")
    parser.add_argument("--no-roll-stats", dest="add_roll_stats", action="store_false")
    parser.add_argument("--roll-window", type=int, default=64)
    parser.add_argument("--no-diff1", dest="add_diff1", action="store_false")
    parser.add_argument("--pct-change", action="store_true")

    # whitening / DR
    parser.add_argument("--whiten", choices=["pca", "zca"], default=None)
    parser.add_argument("--pca-var", type=float, default=0.99)

    # augmentation
    parser.add_argument("--aug-prob", type=float, default=0.0)
    parser.add_argument("--aug-jitter-std", type=float, default=0.01)
    parser.add_argument("--aug-scaling-std", type=float, default=0.05)
    parser.add_argument("--aug-timewarp-max", type=float, default=0.1)

    # persistence
    parser.add_argument("--save-dir", type=str, default=None)

    args = parser.parse_args()

    set_table_style(args.table_style)
    c = C(enabled=supports_color(args.no_color))

    ds = LOBSTERData(
        data_dir=args.data_dir,
        message_file=args.message,
        orderbook_file=args.orderbook,
        feature_set=args.feature_set,
        seq_len=args.seq_len,
        stride=args.stride,
        splits=tuple(args.splits),
        scaler=args.scaler,
        headerless_message=args.headerless_message,
        headerless_orderbook=args.headerless_orderbook,

        add_rel_spread=getattr(args, "add_rel_spread", True),
        add_microprice=getattr(args, "add_microprice", True),
        add_imbalance_l5=getattr(args, "add_imbalance_l5", True),
        add_roll_stats=getattr(args, "add_roll_stats", True),
        roll_window=args.roll_window,
        add_diff1=getattr(args, "add_diff1", True),
        add_pct_change=args.pct_change,

        whiten=args.whiten,
        pca_var=args.pca_var,

        aug_prob=args.aug_prob,
        aug_jitter_std=args.aug_jitter_std,
        aug_scaling_std=args.aug_scaling_std,
        aug_timewarp_max=args.aug_timewarp_max,

        save_dir=args.save_dir,
    )

    # Always show a small preprocessing report card (even without --summary)
    base_rows = [
        ("data_dir", args.data_dir),
        ("message", args.message),
        ("orderbook", args.orderbook),
        ("feature_set", args.feature_set),
        ("seq_len", str(args.seq_len)),
        ("stride", str(args.stride)),
        ("scaler", args.scaler),
        ("whiten", str(args.whiten)),
        ("aug_prob", str(args.aug_prob)),
        ("save_dir", str(args.save_dir)),
    ]
    print(render_kv_panel("Preprocessing config", base_rows, c, style=args.style, align="right"))

    if args.summary:
        # ---------- helpers that render subpanels with textui and nest them ----------
        from helpers.textui import table as tx_table  # alias for clarity

        def _rows_from_df(df: pd.DataFrame, limit_rows: int, limit_cols: int) -> tuple[list[str], list[list[str]]]:
            cols_all = list(map(str, df.columns))
            cols = cols_all[:limit_cols]
            rows_df = df.iloc[:limit_rows, :limit_cols].astype(object).astype(str)
            headers = cols + (["…"] if len(cols_all) > limit_cols else [])
            rows = rows_df.values.tolist()
            if len(cols_all) > limit_cols:
                rows = [r + ["…"] for r in rows]
            return headers, rows

        def _subpanel_lines(title: str, body_lines: list[str]) -> list[str]:
            return render_card(title, body_lines, c, style=args.style, align="left").splitlines()

        def _panel_df(title: str, df: pd.DataFrame, peek: int) -> list[str]:
            headers, rows = _rows_from_df(df, limit_rows=peek, limit_cols=12)
            return _subpanel_lines(title, tx_table(rows, headers, c))

        def _panel_dtypes(df: pd.DataFrame) -> list[str]:
            headers = ["column", "dtype"]
            dtypes_rows = [[str(k), str(v)] for k, v in df.dtypes.items()]
            note = f"total: {len(df.columns)} columns" + (" (showing first 24)" if len(dtypes_rows) > 24 else "")
            dtypes_rows = dtypes_rows[:24]
            body = [note] + tx_table(dtypes_rows, headers, c)
            return _subpanel_lines("dtypes", body)

        def _panel_describe(df: pd.DataFrame) -> list[str]:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                return _subpanel_lines("describe (numeric subset)", ["no numeric columns"])
            sample = num_cols[: min(8, len(num_cols))]
            desc = df[sample].describe().round(6).reset_index(names="stat")
            headers = list(map(str, desc.columns))
            rows = desc.astype(object).astype(str).values.tolist()
            return _subpanel_lines("describe (numeric subset)", tx_table(rows, headers, c))

        def _big_panel(title: str, subpanels: list[list[str]]) -> str:
            body_lines: list[str] = []
            for i, block in enumerate(subpanels):
                if i > 0:
                    body_lines.append("")  # spacer line
                body_lines.extend(block)
            return render_card(title, body_lines, c, style=args.style, align="left")

        # ---------- load CSVs ----------
        msg_df, ob_df = ds._load_csvs()

        # high-level config card (already styled)
        print(render_kv_panel("CSV summary config", [
            ("message file", args.message),
            ("orderbook file", args.orderbook),
            ("rows (message, orderbook)", f"{len(msg_df)}, {len(ob_df)}"),
            ("columns (message, orderbook)", f"{msg_df.shape[1]}, {ob_df.shape[1]}"),
        ], c, style=args.style, align="right"))

        # ---------- message big panel ----------
        msg_subs = []
        msg_subs.append(_subpanel_lines("shape", [f"{msg_df.shape[0]} rows × {msg_df.shape[1]} cols"]))
        msg_subs.append(_panel_dtypes(msg_df))
        msg_subs.append(_panel_describe(msg_df))
        msg_subs.append(_panel_df("head", msg_df.head(args.peek), args.peek))
        msg_subs.append(_panel_df("tail", msg_df.tail(args.peek), args.peek))
        print(_big_panel("message_10.csv", msg_subs))

        # ---------- orderbook big panel ----------
        ob_subs = []
        ob_subs.append(_subpanel_lines("shape", [f"{ob_df.shape[0]} rows × {ob_df.shape[1]} cols"]))
        ob_subs.append(_panel_dtypes(ob_df))
        ob_subs.append(_panel_describe(ob_df))
        ob_subs.append(_panel_df("head", ob_df.head(args.peek), args.peek))
        ob_subs.append(_panel_df("tail", ob_df.tail(args.peek), args.peek))
        print(_big_panel("orderbook_10.csv", ob_subs))

        # ---------- windowed output card (after preprocessing) ----------
        W_train, W_val, W_test = ds.load_arrays()
        rows = [
            ("train windows", "×".join(map(str, W_train.shape))),
            ("val windows",   "×".join(map(str, W_val.shape))),
            ("test windows",  "×".join(map(str, W_test.shape))),
            ("#features",     str(len(ds.get_feature_names()))),
        ]
        print(render_kv_panel("Windows & features", rows, c, style=args.style, align="right"))
        print(render_card(
            "Feature names (first 12)",
            [", ".join(ds.get_feature_names()[:12]) + (" …" if len(ds.get_feature_names())>12 else "")],
            c, style=args.style, align="left"
        ))

    else:
        W_train, W_val, W_test = ds.load_arrays()
        rows = [
            ("train", "×".join(map(str, W_train.shape))),
            ("val",   "×".join(map(str, W_val.shape))),
            ("test",  "×".join(map(str, W_test.shape))),
            ("features", ", ".join(ds.get_feature_names()[:12]) + (" …" if len(ds.get_feature_names())>12 else "")),
        ]
        print(render_kv_panel("Output shapes", rows, c, style=args.style, align="right"))
