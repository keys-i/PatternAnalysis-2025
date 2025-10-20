#!/usr/bin/env python3
"""
Summarise a single LOBSTER order book file (orderbook_10.csv).

Outputs:
  - per_column_summary.csv             # min/max/mean/std/zero% for each of the 40 columns
  - depth_profile.png                  # average depth vs level (bid vs ask)
  - spread_hist.png                    # histogram of best-level spread (USD)
  - midprice_series.png                # mid-price over time (USD)
  - midlogret_hist.png                 # histogram of mid-price log returns
  - summary.md                         # concise human-readable summary

Assumptions:
  - LOBSTER order book file has 40 columns, no header:
      [ask_price_1, ask_size_1, ..., ask_price_10, ask_size_10,
       bid_price_1, bid_size_1, ..., bid_price_10, bid_size_10]
  - Prices are quoted as ticks = dollars * tick_scale (default 10_000); use --tick-scale to adjust.

Usage:
  python summarise_orderbook.py \
    --orderbook ./data/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv \
    --outdir   ./outs/summary_amzn_lvl10 \
    --tick-scale 10000 \
    --seq-len 128
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- Config / Types ---------------------------- #
@dataclass
class OBMeta:
    levels: int
    tick_scale: float
    seq_len: int | None


# --------------------------- Column Helpers ---------------------------- #
def make_orderbook_columns(levels: int = 10) -> List[str]:
    cols: List[str] = []
    for i in range(1, levels + 1):
        cols.append(f"ask_price_{i}")
        cols.append(f"ask_size_{i}")
    for i in range(1, levels + 1):
        cols.append(f"bid_price_{i}")
        cols.append(f"bid_size_{i}")
    return cols  # total 4*levels


# ------------------------------ I/O ----------------------------------- #
def load_orderbook(csv_path: str, levels: int) -> pd.DataFrame:
    cols = make_orderbook_columns(levels)
    try:
        ob = pd.read_csv(csv_path, header=None, names=cols)
    except Exception as e:
        raise RuntimeError(f"Failed to read orderbook CSV at {csv_path}: {e}")
    if ob.shape[1] != 4 * levels:
        raise ValueError(
            f"Expected {4*levels} columns for level={levels} (got {ob.shape[1]}). "
            "Check --levels or file format."
        )
    return ob


# ---------------------------- Computations ---------------------------- #
def compute_top_of_book(ob: pd.DataFrame, tick_scale: float) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    ask1 = ob["ask_price_1"] / tick_scale
    bid1 = ob["bid_price_1"] / tick_scale
    spread = ask1 - bid1
    mid_price = 0.5 * (ask1 + bid1)
    # guard tiny/zero
    mid_safe = mid_price.replace(0, np.nan).fillna(method="ffill").fillna(method="bfill")
    mid_logret = np.log(mid_safe + 1e-12).diff().fillna(0.0)
    return ask1, bid1, spread, mid_logret


def average_depth_profile(ob: pd.DataFrame, levels: int) -> tuple[np.ndarray, np.ndarray]:
    bid_cols = [f"bid_size_{i}" for i in range(1, levels + 1)]
    ask_cols = [f"ask_size_{i}" for i in range(1, levels + 1)]
    bid_depth = ob[bid_cols].astype(float).mean(axis=0).values  # shape [levels]
    ask_depth = ob[ask_cols].astype(float).mean(axis=0).values  # shape [levels]
    return bid_depth, ask_depth


def per_column_summary(ob: pd.DataFrame) -> pd.DataFrame:
    arr = ob.astype(float)
    zeros = (arr == 0).sum(axis=0)
    total = len(arr)
    desc = arr.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["zero_count"] = zeros
    desc["zero_percent"] = (zeros / total) * 100.0
    # reorder columns nicely
    keep = ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "zero_count", "zero_percent"]
    return desc[keep].rename_axis("column").reset_index()


def windows_possible(n_rows: int, seq_len: int | None) -> int | None:
    if seq_len is None:
        return None
    return max(0, n_rows - seq_len + 1)


# ------------------------------- Plots -------------------------------- #
def plot_depth_profile(outdir: str, bid_depth: np.ndarray, ask_depth: np.ndarray) -> str:
    levels = np.arange(1, len(bid_depth) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(levels, bid_depth, marker="o", label="Bid depth")
    plt.plot(levels, ask_depth, marker="o", label="Ask depth")
    plt.xlabel("Level")
    plt.ylabel("Average size")
    plt.title("Average depth profile (mean size per level)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "depth_profile.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def plot_spread_hist(outdir: str, spread: pd.Series) -> str:
    plt.figure(figsize=(7, 4))
    plt.hist(spread.values, bins=100)
    plt.xlabel("Spread (USD)")
    plt.ylabel("Count")
    plt.title("Histogram of best-level spread")
    plt.tight_layout()
    path = os.path.join(outdir, "spread_hist.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def plot_midprice_series(outdir: str, mid_price: pd.Series, max_points: int = 4000) -> str:
    # Downsample for visual clarity if huge
    if len(mid_price) > max_points:
        idx = np.linspace(0, len(mid_price) - 1, max_points).astype(int)
        mp = mid_price.iloc[idx]
        x = np.arange(len(mp))
    else:
        mp = mid_price
        x = np.arange(len(mid_price))
    plt.figure(figsize=(8, 4))
    plt.plot(x, mp.values, linewidth=1)
    plt.xlabel("Event index (downsampled)" if len(mid_price) > max_points else "Event index")
    plt.ylabel("Mid price (USD)")
    plt.title("Mid price over time")
    plt.tight_layout()
    path = os.path.join(outdir, "midprice_series.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def plot_midlogret_hist(outdir: str, mid_logret: pd.Series) -> str:
    plt.figure(figsize=(7, 4))
    # clip heavy tails for nicer viz
    vals = np.clip(mid_logret.values, np.percentile(mid_logret, 0.1), np.percentile(mid_logret, 99.9))
    plt.hist(vals, bins=100)
    plt.xlabel("log mid-price return")
    plt.ylabel("Count")
    plt.title("Histogram of log mid-price returns")
    plt.tight_layout()
    path = os.path.join(outdir, "midlogret_hist.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


# ------------------------------ Summary ------------------------------- #
def write_markdown_summary(
    outdir: str,
    ob_path: str,
    meta: OBMeta,
    n_rows: int,
    zeros_total: int,
    zeros_pct: float,
    spread_stats: dict,
    mid_ret_stats: dict,
    window_count: int | None,
    artifacts: dict[str, str],
) -> None:
    md = []
    md.append("# Order book summary\n")
    md.append(f"- **File**: `{ob_path}`")
    md.append(f"- **Rows**: {n_rows:,}")
    md.append(f"- **Levels**: {meta.levels}")
    md.append(f"- **Tick scale**: {meta.tick_scale:g} (price = ticks / tick_scale)")
    if meta.seq_len is not None:
        md.append(f"- **Seq len** (for windows estimate): {meta.seq_len}")
        md.append(f"- **Possible windows**: {window_count:,}")
    md.append("")
    md.append(f"- **Zeros**: {zeros_total:,} cells  ({zeros_pct:.2f}%)")
    md.append("")
    md.append("## Top-of-book (level 1)\n")
    md.append(f"- Spread (USD): mean={spread_stats['mean']:.6f}, std={spread_stats['std']:.6f}, "
              f"min={spread_stats['min']:.6f}, max={spread_stats['max']:.6f}")
    md.append(f"- |log mid-price return|: mean={mid_ret_stats['mean']:.6f}, std={mid_ret_stats['std']:.6f}, "
              f"p99={mid_ret_stats['p99']:.6f}")
    md.append("")
    md.append("## Artifacts\n")
    for name, path in artifacts.items():
        md.append(f"- {name}: `{path}`")
    md.append("")
    with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))


# ------------------------------ Runner -------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Standalone LOBSTER orderbook_10.csv summariser.")
    ap.add_argument("--orderbook", required=True, help="Path to orderbook_10.csv")
    ap.add_argument("--outdir", required=True, help="Output directory for plots and tables")
    ap.add_argument("--levels", type=int, default=10, help="Number of book levels (default 10)")
    ap.add_argument("--tick-scale", type=float, default=10_000.0, help="LOBSTER tick scale (price = ticks / scale)")
    ap.add_argument("--seq-len", type=int, default=None, help="Optional: sequence length to estimate windows")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    meta = OBMeta(levels=args.levels, tick_scale=float(args.tick_scale), seq_len=args.seq_len)

    # Load
    ob = load_orderbook(args.orderbook, meta.levels)

    # Column summary
    col_summary = per_column_summary(ob)
    col_summary_path = os.path.join(args.outdir, "per_column_summary.csv")
    col_summary.to_csv(col_summary_path, index=False)

    # Zeros overall
    zeros_total = (ob.values == 0).sum()
    zeros_pct = 100.0 * zeros_total / (ob.shape[0] * ob.shape[1])

    # Top-of-book derived series
    ask1, bid1, spread, mid_logret = compute_top_of_book(ob, meta.tick_scale)
    mid_price = 0.5 * (ask1 + bid1)

    # Depth profile
    bid_depth, ask_depth = average_depth_profile(ob, meta.levels)

    # Plots
    arts: dict[str, str] = {}
    arts["depth_profile"]   = plot_depth_profile(args.outdir, bid_depth, ask_depth)
    arts["spread_hist"]     = plot_spread_hist(args.outdir, spread)
    arts["midprice_series"] = plot_midprice_series(args.outdir, mid_price)
    arts["midlogret_hist"]  = plot_midlogret_hist(args.outdir, mid_logret)

    # Small stats for summary
    spread_stats = dict(mean=float(spread.mean()), std=float(spread.std()),
                        min=float(spread.min()), max=float(spread.max()))
    abs_ret = mid_logret.abs()
    mid_ret_stats = dict(mean=float(abs_ret.mean()), std=float(abs_ret.std()),
                         p99=float(abs_ret.quantile(0.99)))

    # Windows estimate
    wcount = windows_possible(len(ob), meta.seq_len)

    # Write markdown summary
    write_markdown_summary(
        outdir=args.outdir,
        ob_path=args.orderbook,
        meta=meta,
        n_rows=len(ob),
        zeros_total=int(zeros_total),
        zeros_pct=float(zeros_pct),
        spread_stats=spread_stats,
        mid_ret_stats=mid_ret_stats,
        window_count=wcount,
        artifacts=arts,
    )

    print(f"[done] Summary written to: {args.outdir}")

if __name__ == "__main__":
    main()
