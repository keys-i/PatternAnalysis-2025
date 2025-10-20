#!/usr/bin/env python3
# npy_to_csv.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.table import Table

console = Console()


def show_peek(df: pd.DataFrame, n: int) -> None:
    if n <= 0:
        return
    n = min(n, len(df))
    table = Table(title=f"Peek (first {n} rows)", show_lines=False)
    for c in df.columns:
        table.add_column(str(c))
    for _, row in df.head(n).iterrows():
        table.add_row(*[str(x) for x in row.to_list()])
    console.print(table)


def show_summary(df: pd.DataFrame, topk: int = 8) -> None:
    desc = df.describe().T  # count, mean, std, min, 25%, 50%, 75%, max
    # keep only first topk columns for display to keep it compact
    cols = ["count", "mean", "std", "min", "50%", "max"]
    table = Table(title="Summary stats (per column)", show_lines=False)
    for c in ["column"] + cols:
        table.add_column(c)
    for name, row in desc.head(topk).iterrows():
        table.add_row(
            str(name),
            *(f"{row[c]:.6g}" if pd.notnull(row[c]) else "nan" for c in cols),
        )
    console.print(table)
    if len(desc) > topk:
        console.print(f"[dim] {len(desc) - topk} more columns not shown[/dim]")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert a 2D NumPy .npy array to CSV with rich peek/summary."
    )
    ap.add_argument(
        "--in", dest="inp", default="./outs/gen_data.npy", help="Input .npy file"
    )
    ap.add_argument(
        "--out", dest="outp", default="./outs/gen_data.csv", help="Output .csv file"
    )
    ap.add_argument("--prefix", default="f", help="Column name prefix (default: f)")
    ap.add_argument(
        "--peek",
        type=int,
        default=5,
        help="Show first N rows in the console (0 = disable)",
    )
    ap.add_argument(
        "--summary", action="store_true", help="Print per-column summary statistics"
    )
    ap.add_argument(
        "--no-save", action="store_true", help="Do not write CSV (preview only)"
    )
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.outp)
    outp.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        console.print(f"[red]Input not found:[/red] {inp}")
        raise SystemExit(1)

    with Status(f"[cyan]Loading[/cyan] {inp}", console=console):
        arr = np.load(inp)

    if arr.ndim != 2:
        console.print(f"[red]Expected a 2D array, got shape {arr.shape}[/red]")
        raise SystemExit(2)

    n_rows, n_cols = arr.shape
    cols = [f"{args.prefix}{i}" for i in range(n_cols)]

    console.print(
        Panel.fit(f"[bold]Array shape[/bold]: {n_rows} × {n_cols}", border_style="cyan")
    )

    df = pd.DataFrame(arr, columns=cols)

    # Peek and summary
    show_peek(df, args.peek)
    if args.summary:
        show_summary(df)

    # Save CSV unless suppressed
    if not args.no - save:
        with Status(f"[cyan]Writing CSV[/cyan] → {outp}", console=console):
            df.to_csv(outp, index=False)
        console.print(f"[green]Done:[/green] wrote [bold]{outp}[/bold]")


if __name__ == "__main__":
    main()
