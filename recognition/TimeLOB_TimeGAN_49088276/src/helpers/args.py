"""
Options for the entire model
"""
from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace, REMAINDER
from typing import Optional, List

import numpy as np

from src.helpers.constants import DATA_DIR, TRAIN_TEST_SPLIT, ORDERBOOK_FILENAME, NUM_TRAINING_ITERATIONS

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
        parser.add_argument("--data-dir", dest="data_dir", type=str, default=str(DATA_DIR))
        parser.add_argument("--orderbook-filename", dest="orderbook_filename", type=str, default=ORDERBOOK_FILENAME)
        parser.add_argument(
            "--no-shuffle",
            action="store_true",
            help="Disable shuffling of windowed sequences"
        )
        parser.add_argument(
            "--keep-zero-rows", dest="keep_zero_rows",
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

    def parse(self, argv: Optional[List[str]]) -> Namespace:
        if argv is None:
            argv = []
        ds = self._parser.parse_args(argv)

        ns = Namespace(
            seq_len=ds.seq_len,
            data_dir=ds.data_dir,
            orderbook_filename=ds.orderbook_filename,
            splits=tuple(ds.splits) if ds.splits is not None else TRAIN_TEST_SPLIT,
            shuffle_windows=not ds.no_shuffle,
            dtype=np.float32,
            filter_zero_rows=not ds.keep_zero_rows,
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
        # core shapes
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--seq-len", type=int, default=128,
                            help="Sequence length (kept here for convenience to sync with data).")
        parser.add_argument("--z-dim", type=int, default=40,
                            help="Latent/input feature dim (e.g., LOB feature count).")
        parser.add_argument("--hidden-dim", type=int, default=64,
                            help="Module hidden size.")
        parser.add_argument("--num-layer", type=int, default=3,
                            help="Number of stacked layers per RNN/TCN block.")

        # optimizer
        parser.add_argument("--lr", type=float, default=1e-4,
                            help="Learning rate (generator/supervisor/discriminator if shared).")
        parser.add_argument("--beta1", type=float, default=0.5,
                            help="Adam beta1.")

        # Loss weights
        parser.add_argument("--w-gamma", type=float, default=1.0,
                            help="Supervisor loss weight (γ).")
        parser.add_argument("--w-g", type=float, default=1.0,
                            help="Generator adversarial loss weight (g).")

        parser.add_argument("--num-iters", type=int, default=NUM_TRAINING_ITERATIONS,
                            help="Number of training iterations per phase (ER, S, Joint).")

        self._parser = parser

    def parse(self, argv: Optional[List[str]]) -> Namespace:
        if argv is None:
            argv = []
        m = self._parser.parse_args(argv)

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
            num_iters=m.num_iters,
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

    def parse(self, argv: Optional[List[str]] = None) -> Namespace:

        # raw tokens (exclude program name)
        tokens: List[str] = list(sys.argv[1:] if argv is None else argv)

        # extract sections: --dataset ..., --modules ...
        def extract(flag: str, toks: List[str]) -> tuple[List[str], List[str]]:
            if flag not in toks:
                return [], toks
            i = toks.index(flag)
            rest = toks[i + 1:]
            # stop at the next section flag (or end)
            next_indices = [j for j, t in enumerate(rest) if t in ("--dataset", "--modules")]
            end = next_indices[0] if next_indices else len(rest)
            section = rest[:end]
            remaining = toks[:i] + rest[end:]
            return section, remaining

        ds_args, remaining = extract("--dataset", tokens)
        mod_args, remaining = extract("--modules", remaining)

        # parse top-level only from what's left (seed/run-name)
        top = self._parser.parse_args(remaining)

        # parse subsections (never read global argv inside these)
        dataset_ns = DataOptions().parse(ds_args or [])
        modules_ns = ModulesOptions().parse(mod_args or [])

        # assemble composite namespace
        return Namespace(
            seed=top.seed,
            run_name=top.run_name,
            dataset=dataset_ns,
            modules=modules_ns,
        )


if __name__ == "__main__":
    opts = Options().parse()
    print(opts)

