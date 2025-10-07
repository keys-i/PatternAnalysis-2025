"""
Options for the entire model
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace, REMAINDER
from typing import Optional

import numpy as np

from src.helpers.constants import DATA_DIR, TRAIN_TEST_SPLIT, ORDERBOOK_FILENAME

try:
    # tolerate alternates if present in your helpers
    from src.helpers.constants import ORDERBOOK_FILENAME as _OB_ALT
    ORDERBOOK_DEFAULT = _OB_ALT
except Exception:
    ORDERBOOK_DEFAULT = ORDERBOOK_FILENAME

class DataOptions:
    """
    Thin wrapper around argparse that produces a Namespace suitable for DatasetConfig.
    Usage:
        opts = DataOptions().parse()
        train_w, val_w, test_w = load_data(opts)
    """

    def __init__(self) -> None:
        parser = ArgumentParser(
            prog="timeganlob_dataset",
            description="Lightweight LOBSTER preprocessing + MinMax scaling",
        )
        parser.add_argument("--seq-len", type=int, default=128)
        parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
        parser.add_argument("--orderbook_filename", type=str, default=ORDERBOOK_FILENAME)
        parser.add_argument(
            "--no-shuffle",
            action="store_true",
            help="Disable shuffling of windowed sequences"
        )
        parser.add_argument(
            "--keep_zero_rows",
            action="store_true",
            help="Do NOT filter rows containing zeros."
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

    def parse(self, argv: Optional[list | str]) -> Namespace:
        args = self._parser.parse_args(argv)

        ns = Namespace(
            seq_len=args.seq_len,
            data_dir=args.data_dir,
            orderbook_filename=args.orderbook_filename,
            splits=tuple(args.splits) if args.splits is not None else TRAIN_TEST_SPLIT,
            shuffle_windows=not args.no_shuffle,
            dtype=np.float32,
            keep_zero_rows=not args.keep_zero_rows,
        )

        return ns

class ModulesOptions:
    """
    Hyperparameters for modules & training. Designed to feel like an `opt` object.

    Usage:
        mods = ModulesOptions().parse(argv_after_flag)
        # Access:
        mods.batch_size, mods.seq_len, mods.z_dim, mods.hidden_dim, mods.num_layer,
        mods.lr, mods.beta1, mods.w_gamma, mods.w_g
    """
    def __init__(self) -> None:
        parser = ArgumentParser(
            prog="timeganlob_modules",
            description="Module/model hyperparameters and training weights.",
        )
        # Core shapes
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--seq-len", type=int, default=128,
                            help="Sequence length (kept here for convenience to sync with data).")
        parser.add_argument("--z-dim", type=int, default=40,
                            help="Latent/input feature dim (e.g., LOB feature count).")
        parser.add_argument("--hidden-dim", type=int, default=64,
                            help="Module hidden size.")
        parser.add_argument("--num-layer", type=int, default=3,
                            help="Number of stacked layers per RNN/TCN block.")

        # Optimizer
        parser.add_argument("--lr", type=float, default=1e-4,
                            help="Learning rate (generator/supervisor/discriminator if shared).")
        parser.add_argument("--beta1", type=float, default=0.5,
                            help="Adam beta1.")

        # Loss weights
        parser.add_argument("--w-gamma", type=float, default=1.0,
                            help="Supervisor loss weight (γ).")
        parser.add_argument("--w-g", type=float, default=1.0,
                            help="Generator adversarial loss weight (g).")

        self._parser = parser

    def parse(self, argv: Optional[list | str]) -> Namespace:
        m = self._parser.parse_args(argv)

        # Provide both snake_case and "opt-like" names already as attributes
        # (so downstream code can do opt.lr, opt.beta1, opt.w_gamma, opt.w_g).
        ns = Namespace(
            batch_size=m.batch_size,
            seq_len=m.seq_len,
            z_dim=m.z_dim,
            hidden_dim=m.hidden_dim,
            num_layer=m.num_layer,
            lr=m.lr,
            beta1=m.beta1,
            w_gamma=m.w_gamma,
            w_g=m.w_g,
        )
        return ns

class Options:
    """
    Top-level options that *route* anything after `--dataset` to DatasetOptions.

    Example:
        opts = Options().parse()
        ds = opts.dataset  # Namespace from DatasetOptions
    """
    def __init__(self) -> None:
        parser = ArgumentParser(
            prog="timeganlob",
            description="TimeGAN-LOB entrypoint with nested dataset options."
        )
        parser.add_argument("--seed", type=int, default=42, help="Global random seed")
        parser.add_argument("--run-name", type=str, default="exp1", help="Run name")

        parser.add_argument(
            "--dataset",
            nargs=REMAINDER,
            help=(
                "All arguments following this flag are parsed by DatasetOptions. "
                "Example: --dataset --seq-len 256 --no-shuffle"
            ),
        )
        parser.add_argument(
            "--modules",
            nargs=REMAINDER,
            help=(
                "All arguments following this flag are parsed by ModulesOptions. "
                "Example: --modules --batch-size 256 --hidden-dim 128 --lr 3e-4"
            ),
        )
        self._parser = parser

    def parse(self, argv: Optional[list | str] = None) -> Namespace:
        top = self._parser.parse_args(argv)

        ds_argv = top.dataset if top.dataset is not None else []
        dataset_ns = DataOptions().parse(ds_argv)

        # attach nested namespace to the top-level namespace
        out = Namespace(
            seed=top.seed,
            run_name=top.run_name,
            dataset=dataset_ns,
        )

        return out

if __name__ == "__main__":
    opts = Options().parse()

    print(opts)