#!/usr/bin/env python3
"""
Rich-powered logging helpers with animated status for CLI output.

This module provides a small wrapper around `rich` to standardize console UX:

- `log(msg)`: structured logging with Rich, falling back to `print`.
- `status(msg)`: re-entrant-safe status context manager with a moving ellipsis.
- `rule(text)`: horizontal rule separator.
- `dataset_summary(...)`: formatted header and splits table for dataset stats.

If `rich` is not available, all helpers degrade gracefully to plain text.
"""

from __future__ import annotations

import contextvars
import itertools
import threading
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _CONSOLE: Optional[Console] = Console()
except Exception:  # fallback if rich isn’t installed
    _CONSOLE = None

# Track nesting depth per context/thread for status() re-entrancy.
_live_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_live_depth", default=0
)


def log(msg: str) -> None:
    """Write a single log line to the console.

    Uses Rich's Console.log when available, otherwise falls back to print().

    Args:
        msg: Text to log.
    """
    if _CONSOLE:
        _CONSOLE.log(msg)
    else:
        print(msg)


def status(msg: str):
    """Return a re-entrant-safe status context manager with animated ellipsis.

    The outermost `with status("..."):` call starts a Rich status spinner and a
    tiny background thread that appends a moving ellipsis to the message at a
    fixed cadence. Nested calls are no-ops to avoid stacking multiple spinners.

    Example:
        with status("Loading data"):
            ...  # long-running work

    Args:
        msg: Base message to display next to the spinner.

    Returns:
        A context manager suitable for `with` statements.
    """
    depth = _live_depth.get()
    if _CONSOLE and depth == 0:
        rich_status = _CONSOLE.status(msg)
        stop_flag = threading.Event()
        dots = itertools.cycle(
            ["", ".", "..", "...", "....", ".....", "....", "...", "..", "."]
        )

        class _Wrapper:
            """Context manager that manages spinner lifecycle and animation."""

            def __enter__(self):
                self._token = _live_depth.set(depth + 1)
                self._ctx = rich_status.__enter__()

                def _tick():
                    """Animate ellipsis by updating the status message periodically."""
                    next_tick = time.time() + 0.35
                    while not stop_flag.wait(timeout=max(0.0, next_tick - time.time())):
                        try:
                            rich_status.update(f"{msg}{next(dots)}")
                        except Exception:
                            # Be resilient to any console teardown
                            pass
                        next_tick = time.time() + 0.35

                self._thr = threading.Thread(
                    target=_tick, name="rich-status-ellipsis", daemon=True
                )
                self._thr.start()
                return self._ctx

            def __exit__(self, exc_type, exc, tb):
                """Stop animation, restore base message, and exit the Rich status."""
                try:
                    stop_flag.set()
                    if hasattr(self, "_thr"):
                        self._thr.join(timeout=0.3)
                    try:
                        rich_status.update(msg)
                    except Exception:
                        pass
                    return rich_status.__exit__(exc_type, exc, tb)
                finally:
                    _live_depth.reset(self._token)

        return _Wrapper()

    # Nested calls: return a no-op context manager to keep output clean.
    class _Noop:
        """No-op context manager used for nested `status` calls."""

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    return _Noop()


def rule(text: str = "") -> None:
    """Draw a horizontal rule in the console.

    Args:
        text: Optional caption rendered in the rule.
    """
    if _CONSOLE:
        _CONSOLE.rule(text)


def dataset_summary(
    *,
    file_path: Path,
    seq_len: int,
    dtype_name: str,
    filter_zero_rows: bool,
    splits: Iterable[Tuple[str, Tuple[int, int]]],
) -> None:
    """Render a dataset header panel and a splits table.

    On Rich consoles, shows a styled panel with file path, sequence length,
    dtype, and zero-row filtering, followed by a table of split sizes and
    window counts. Falls back to plain text when Rich is unavailable.

    Args:
        file_path: Path to the loaded orderbook file.
        seq_len: Sequence length used for windowing.
        dtype_name: Name of the dtype used for arrays (e.g., 'float32').
        filter_zero_rows: Whether zero-containing rows were filtered out.
        splits: Iterable of (split_name, (rows, windows)) entries.
    """
    if _CONSOLE is None:
        print(
            f"Dataset: {file_path} | seq_len={seq_len} | dtype={dtype_name} | filter_zero_rows={filter_zero_rows}"
        )
        for name, (rows, wins) in splits:
            print(f"{name:>6}: rows={rows:,} windows={wins:,}")
        return

    header = Panel.fit(
        f"[bold cyan]LOBSTER dataset summary[/bold cyan]\n"
        f"[dim]file:[/dim] {file_path}\n"
        f"[dim]seq_len:[/dim] {seq_len}   "
        f"[dim]dtype:[/dim] {dtype_name}   "
        f"[dim]filter_zero_rows:[/dim] {filter_zero_rows}",
        border_style="cyan",
    )

    table = Table(
        title="Splits",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        header_style="bold",
        expand=False,
    )
    table.add_column("Split")
    table.add_column("Rows", justify="right")
    table.add_column("Windows", justify="right")

    for name, (rows, wins) in splits:
        table.add_row(name, f"{rows:,}", f"{wins:,}")

    _CONSOLE.rule()
    _CONSOLE.print(header)
    _CONSOLE.print(table)
    _CONSOLE.rule()
