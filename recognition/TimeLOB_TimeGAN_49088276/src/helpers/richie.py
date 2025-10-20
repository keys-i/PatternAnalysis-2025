# src/helpers/richie.py
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

# track nesting depth per context/thread
_live_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_live_depth", default=0
)


def log(msg: str) -> None:
    if _CONSOLE:
        _CONSOLE.log(msg)
    else:
        print(msg)


def status(msg: str):
    """Re-entrant-safe status spinner with animated ellipsis.
    - Outermost call starts a Rich status + a background thread that updates the text with a moving ellipsis.
    - Nested calls become no-ops to avoid stacking spinners.
    """
    depth = _live_depth.get()
    if _CONSOLE and depth == 0:
        rich_status = _CONSOLE.status(msg)
        stop_flag = threading.Event()
        dots = itertools.cycle(
            ["", ".", "..", "...", "....", ".....", "....", "...", "..", "."]
        )

        class _Wrapper:
            def __enter__(self):
                self._token = _live_depth.set(depth + 1)
                self._ctx = rich_status.__enter__()

                # start a tiny background updater that animates the message
                def _tick():
                    # small initial delay so first frame shows base msg
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
                try:
                    # stop animation and restore base message for a clean exit frame
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

    # nested: no-op to keep output clean
    class _Noop:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    return _Noop()


def rule(text: str = "") -> None:
    if _CONSOLE:
        _CONSOLE.rule(text)


def dataset_summary(
    *,
    file_path: Path,
    seq_len: int,
    dtype_name: str,
    filter_zero_rows: bool,
    splits: Iterable[Tuple[str, Tuple[int, int]]],  # (name, (rows, windows))
) -> None:
    """Render a header + splits table."""
    if _CONSOLE is None:
        # Plain fallback
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
