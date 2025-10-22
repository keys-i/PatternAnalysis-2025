#!/usr/bin/env python3
"""
Unified CLI options for TimeGAN–LOBSTER.

This module defines three option groups:
- DataOptions: dataset loading and preprocessing flags.
- ModulesOptions: model and training hyperparameters.
- VisualiseOptions: visualisation and metric reporting flags.

The top-level Options router splits argv into subsections using
`--dataset …`, `--modules …`, and `--viz …`, then parses each with the
corresponding subparser. All groups degrade safely if a section is omitted.
"""

from __future__ import annotations

import sys
from argparse import REMAINDER, ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.helpers.constants import (
    DATA_DIR,
    NUM_TRAINING_ITERATIONS,
    ORDERBOOK_FILENAME,
    OUTPUT_DIR,
    TRAIN_TEST_SPLIT,
)

# Optional override for orderbook filename from constants
try:
    from src.helpers.constants import ORDERBOOK_FILENAME as _OB_ALT

    ORDERBOOK_DEFAULT = _OB_ALT
except Exception:
    ORDERBOOK_DEFAULT = ORDERBOOK_FILENAME

# Default LOB levels (fallback to 10 if not exported)
try:
    from src.helpers.constants import NUM_LEVELS as _DEFAULT_LEVELS
except Exception:
    _DEFAULT_LEVELS = 10


class DataOptions:
    """Dataset options parsed after `--dataset`."""

    def __init__(self) -> None:
        parser = ArgumentParser(
            prog="timeganlob_dataset",
            description="Lightweight LOBSTER preprocessing + MinMax scaling",
        )
        parser.add_argument("--seq-len", type=int, default=128)
        parser.add_argument(
            "--data-dir", dest="data_dir", type=str, default=str(DATA_DIR)
        )
        parser.add_argument(
            "--orderbook-filename",
            dest="orderbook_filename",
            type=str,
            default=ORDERBOOK_DEFAULT,
        )
        parser.add_argument(
            "--no-shuffle",
            action="store_true",
            help="Disable shuffling of windowed sequences",
        )
        parser.add_argument(
            "--keep-zero-rows",
            dest="keep_zero_rows",
            action="store_true",
            help="Do NOT filter rows containing zeros.",
        )
        parser.add_argument(
            "--splits",
            type=float,
            nargs=3,
            metavar=("TRAIN", "VAL", "TEST"),
            help="Either proportions that sum to ~1.0 or cumulative cutoffs (e.g., 0.6 0.8 1.0).",
            default=None,
        )
        self._parser = parser

    def parse(self, argv: Optional[List[str]]) -> Namespace:
        """Parse dataset arguments."""
        if argv is None:
            argv = []
        ds = self._parser.parse_args(argv)
        return Namespace(
            seq_len=ds.seq_len,
            data_dir=ds.data_dir,
            orderbook_filename=ds.orderbook_filename,
            splits=tuple(ds.splits) if ds.splits is not None else TRAIN_TEST_SPLIT,
            shuffle_windows=not ds.no_shuffle,
            dtype=np.float32,
            filter_zero_rows=not ds.keep_zero_rows,
        )


class ModulesOptions:
    """Model and training hyperparameters parsed after `--modules`."""

    def __init__(self) -> None:
        parser = ArgumentParser(
            prog="timeganlob_modules",
            description="Module/model hyperparameters and training weights.",
        )
        # core shapes
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument(
            "--seq-len",
            type=int,
            default=128,
            help="Sequence length (convenience mirror).",
        )
        parser.add_argument(
            "--z-dim", type=int, default=40, help="Latent/input feature dimension."
        )
        parser.add_argument(
            "--hidden-dim", type=int, default=64, help="GRU hidden size."
        )
        parser.add_argument(
            "--num-layer", type=int, default=3, help="Number of stacked GRU layers."
        )
        # optimizer
        parser.add_argument(
            "--lr", type=float, default=1e-4, help="Adam learning rate."
        )
        parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1.")
        # loss weights
        parser.add_argument(
            "--w-gamma",
            type=float,
            default=1.0,
            help="Supervisor-related adversarial weight.",
        )
        parser.add_argument(
            "--w-g", type=float, default=1.0, help="Generator loss and moments weight."
        )
        # schedule
        parser.add_argument(
            "--num-iters",
            type=int,
            default=NUM_TRAINING_ITERATIONS,
            help="Iterations per phase (ER, Supervisor, Joint).",
        )
        self._parser = parser

    def parse(self, argv: Optional[List[str]]) -> Namespace:
        """Parse modules arguments."""
        if argv is None:
            argv = []
        m = self._parser.parse_args(argv)
        return Namespace(
            batch_size=m.batch_size,
            seq_len=m.seq_len,
            z_dim=m.z_dim,
            hidden_dim=m.hidden_dim,
            num_layer=m.num_layer,
            lr=m.lr,
            beta1=m.beta1,
            w_gamma=m.w_gamma,
            w_g=m.w_g,
            num_iters=m.num_iters,
        )


class VisualiseOptions:
    """Visualisation and metric reporting flags parsed after `--viz`."""

    def __init__(self) -> None:
        parser = ArgumentParser(
            prog="timeganlob_viz",
            description="Visualisation and metric reporting options for generated LOB sequences.",
        )
        parser.add_argument(
            "--samples",
            type=int,
            default=3,
            help="Number of synthetic samples to generate",
        )
        parser.add_argument(
            "--out-dir",
            type=Path,
            default=Path(OUTPUT_DIR) / "viz",
            help="Output directory",
        )
        parser.add_argument(
            "--bins", type=int, default=100, help="Histogram bins for KL computation"
        )
        parser.add_argument(
            "--cmap", type=str, default="coolwarm", help="Colormap for heatmaps"
        )
        parser.add_argument(
            "--no-log1p",
            action="store_true",
            help="Disable log1p transform in heatmaps",
        )
        parser.add_argument(
            "--dpi", type=int, default=220, help="DPI for saved figures"
        )
        parser.add_argument(
            "--levels",
            type=int,
            default=None,
            help=f"Override LOB levels (default: {_DEFAULT_LEVELS})",
        )
        parser.add_argument(
            "--no-ssim", action="store_true", help="Skip SSIM computation"
        )
        parser.add_argument(
            "--no-kl", action="store_true", help="Skip KL(spread/mpr) computation"
        )
        parser.add_argument(
            "--no-temp",
            action="store_true",
            help="Skip temporal correlation computation",
        )
        parser.add_argument(
            "--no-lat", action="store_true", help="Skip latent distance computation"
        )
        parser.add_argument(
            "--metrics-csv",
            type=Path,
            default=None,
            help="Optional path to save metrics CSV",
        )
        # latent walk controls
        parser.add_argument(
            "--walk", action="store_true", help="Enable latent-space walks"
        )
        parser.add_argument(
            "--walk-steps",
            type=int,
            default=8,
            help="Number of interpolation steps for walks",
        )
        parser.add_argument(
            "--walk-mode",
            type=str,
            default="both",
            choices=["within", "cross", "both"],
            help="Which walk(s) to generate",
        )
        parser.add_argument(
            "--walk-prefix",
            type=str,
            default="latent_walk",
            help="Prefix for walk panel filenames",
        )
        self._parser = parser

    def parse(self, argv: Optional[List[str]]) -> Namespace:
        """Parse visualisation arguments."""
        if argv is None:
            argv = []
        v = self._parser.parse_args(argv)
        return Namespace(
            samples=int(v.samples),
            out_dir=Path(v.out_dir),
            bins=int(v.bins),
            cmap=str(v.cmap),
            no_log1p=bool(v.no_log1p),
            dpi=int(v.dpi),
            levels=(int(v.levels) if v.levels is not None else None),
            no_ssim=bool(v.no_ssim),
            no_kl=bool(v.no_kl),
            no_temp=bool(v.no_temp),
            no_lat=bool(v.no_lat),
            metrics_csv=(Path(v.metrics_csv) if v.metrics_csv is not None else None),
            walk=bool(v.walk),
            walk_steps=int(v.walk_steps),
            walk_mode=str(v.walk_mode).lower(),
            walk_prefix=str(v.walk_prefix),
        )


class Options:
    """Top-level router that splits argv into dataset, modules, and viz sections."""

    def __init__(self) -> None:
        parser = ArgumentParser(
            prog="timeganlob",
            description="TimeGAN-LOB entrypoint with nested dataset/module/viz options.",
        )
        parser.add_argument("--seed", type=int, default=42, help="Global random seed")
        parser.add_argument("--run-name", type=str, default="exp1", help="Run name")
        parser.add_argument(
            "--dataset",
            nargs=REMAINDER,
            help="All arguments after this flag go to DataOptions",
        )
        parser.add_argument(
            "--modules",
            nargs=REMAINDER,
            help="All arguments after this flag go to ModulesOptions",
        )
        parser.add_argument(
            "--viz",
            nargs=REMAINDER,
            help="All arguments after this flag go to VisualiseOptions",
        )
        self._parser = parser

    def _extract(
        self, flag: str, toks: List[str], stops: tuple[str, ...]
    ) -> tuple[List[str], List[str]]:
        """Extract the sub-sequence after `flag` until the next flag in `stops` or end."""
        if flag not in toks:
            return [], toks
        i = toks.index(flag)
        rest = toks[i + 1 :]
        next_idx = [j for j, t in enumerate(rest) if t in stops]
        end = next_idx[0] if next_idx else len(rest)
        section = rest[:end]
        remaining = toks[:i] + rest[end:]
        return section, remaining

    def parse(self, argv: Optional[List[str]] = None) -> Namespace:
        """Parse top-level and nested option groups."""
        tokens: List[str] = list(sys.argv[1:] if argv is None else argv)
        stop_flags = ("--dataset", "--modules", "--viz")

        ds_args, rem = self._extract("--dataset", tokens, stop_flags)
        mod_args, rem = self._extract("--modules", rem, stop_flags)
        viz_args, rem = self._extract("--viz", rem, stop_flags)

        top = self._parser.parse_args(rem)

        dataset_ns = DataOptions().parse(ds_args or [])
        modules_ns = ModulesOptions().parse(mod_args or [])
        visual_ns = VisualiseOptions().parse(viz_args or [])

        return Namespace(
            seed=top.seed,
            run_name=top.run_name,
            dataset=dataset_ns,
            modules=modules_ns,
            viz=visual_ns,
        )


if __name__ == "__main__":
    print(Options().parse())
