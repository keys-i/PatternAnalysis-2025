"""
Generate LOB depth heatmaps and compute SSIM between real vs synthetic images.
Refactored to be faster, cleaner, and compatible with the new modules/utils.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity as ssim
from skimage.util import img_as_float

from src.dataset import load_data

# use nested CLI options + constants from src.helpers
from src.helpers.args import Options
from src.helpers.constants import NUM_LEVELS, OUTPUT_DIR
from src.helpers.richie import log as rlog
from src.helpers.richie import rule as rrule
from src.helpers.richie import status as rstatus
from src.modules import TimeGAN

# optional pretty table for SSIM results (graceful fallback if rich unavailable)
try:
    from rich import box
    from rich.table import Table

    _HAS_RICH_TABLE = True
except Exception:
    _HAS_RICH_TABLE = False


def get_ssim(img1_path: Path | str, img2_path: Path | str) -> float:
    """
    Compute SSIM between two image files.

    Uses `channel_axis=2` (new skimage API). Images are read via matplotlib.
    """
    img1 = img_as_float(plt.imread(str(img1_path)))
    img2 = img_as_float(plt.imread(str(img2_path)))

    # if grayscale, add channel axis
    if img1.ndim == 2:
        img1 = img1[..., None]
    if img2.ndim == 2:
        img2 = img2[..., None]
    return float(ssim(img1, img2, channel_axis=2, data_range=1.0))


def plot_heatmap(
    data_2d: NDArray,  # shape [T, F]
    *,
    title: str | None = None,
    save_path: Path | str | None = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """
    Scatter-based depth heatmap.

    Assumes features are interleaved per level: [ask_price, ask_vol, bid_price, bid_vol] x NUM_LEVELS.
    Colors: red=ask, blue=bid, alpha encodes relative volume in [0,1].
    """
    T, F = data_2d.shape
    assert F >= 4 * NUM_LEVELS, "Expected at least 4 features per level"

    # slice views
    # for each level L: price indices = 4*L + (0 for ask, 2 for bid)
    # vol indices = price_idx + 1
    prices_ask = np.stack([data_2d[:, 4 * L + 0] for L in range(NUM_LEVELS)], axis=1)  # [T, L]
    vols_ask = np.stack([data_2d[:, 4 * L + 1] for L in range(NUM_LEVELS)], axis=1)  # [T, L]
    prices_bid = np.stack([data_2d[:, 4 * L + 2] for L in range(NUM_LEVELS)], axis=1)  # [T, L]
    vols_bid = np.stack([data_2d[:, 4 * L + 3] for L in range(NUM_LEVELS)], axis=1)  # [T, L]

    # Normalise volumes for alpha
    max_vol = float(max(prices_ask.size and vols_ask.max(), prices_bid.size and vols_bid.max()))
    if not np.isfinite(max_vol) or max_vol <= 0:
        max_vol = 1.0
    a_ask = (vols_ask / max_vol).astype(np.float32)
    a_bid = (vols_bid / max_vol).astype(np.float32)

    # build scatter arrays
    # x: time indices repeated for each level
    t_idx = np.arange(T, dtype=np.float32)[:, None]
    x_ask = np.repeat(t_idx, NUM_LEVELS, axis=1).ravel()
    x_bid = x_ask.copy()
    y_ask = prices_ask.astype(np.float32).ravel()
    y_bid = prices_bid.astype(np.float32).ravel()

    # colors rgba
    c_ask = np.stack(
        [
            np.full_like(y_ask, 0.99),  # r
            np.full_like(y_ask, 0.05),  # g
            np.full_like(y_ask, 0.05),  # b
            a_ask.astype(np.float32).ravel(),  # A
        ],
        axis=1,
    )
    c_bid = np.stack(
        [
            np.full_like(y_ask, 0.05),  # r
            np.full_like(y_ask, 0.05),  # g
            np.full_like(y_ask, 0.99),  # b
            a_bid.astype(np.float32).ravel(),  # A
        ],
        axis=1,
    )

    # limits
    pmin = float(min(prices_ask.min(), prices_bid.min()))
    pmax = float(max(prices_ask.max(), prices_bid.max()))

    # plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax.set_ylim(pmin, pmax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    if title:
        ax.set_title(title)

    ax.scatter(x_ask, y_ask, c=c_ask, s=1)
    ax.scatter(x_bid, y_bid, c=c_bid, s=1)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _print_ssim_table(rows: List[Tuple[str, float]]) -> None:
    """Pretty-print SSIM results if rich is available; fall back to logs."""
    if _HAS_RICH_TABLE:
        table = Table(title="SSIM: Real vs Synthetic", header_style="bold", box=box.SIMPLE_HEAVY)
        table.add_column("Sample")
        table.add_column("SSIM", justify="right")
        for k, v in rows:
            table.add_row(k, f"{v:.4f}")
        # use richie's rule/log if available
        rrule()
        # `rlog` prints line-wise; here we directly print the table via rich's console if available
        try:
            from rich.console import Console

            Console().print(table)
        except Exception:
            # fallback to logging lines
            for k, v in rows:
                rlog(f"SSIM({k}) = {v:.4f}")
        rrule()
    else:
        rlog("SSIM: Real vs Synthetic")
        for k, v in rows:
            rlog(f"  {k:<16} {v:.4f}")


if __name__ == "__main__":
    rrule("[bold cyan]Heatmaps & SSIM[/bold cyan]")

    # cli
    top = Options().parse()

    # data
    with rstatus("[cyan]Loading data…"):
        train, val, test = load_data(top.dataset)
        # flatten windowed val/test ([N,T,F] -> [T',F]) for viz/metrics
        if getattr(val, "ndim", None) == 3:
            val = val.reshape(-1, val.shape[-1])
        if getattr(test, "ndim", None) == 3:
            test = test.reshape(-1, test.shape[-1])

    rlog(
        f"Splits: train_w={train.shape}  val={getattr(val, 'shape', None)}  test={getattr(test, 'shape', None)}"
    )

    # model (load weights)
    with rstatus("[cyan]Restoring TimeGAN checkpoint…"):
        model = TimeGAN(top.modules, train, val, test, load_weights=True)

    # real heatmap from test data
    real_path = OUTPUT_DIR / "real.png"
    with rstatus("[cyan]Rendering real heatmap…"):
        plot_heatmap(test, title="Real LOB Depth", save_path=real_path, show=False)
    rlog(f"Saved: {real_path}")

    # generate and compare a few samples
    scores: List[Tuple[str, float]] = []
    for i in range(3):
        with rstatus(f"[cyan]Sampling synthetic #{i}…"):
            synth = model.generate(num_rows=int(test.shape[0]))
        synth_path = OUTPUT_DIR / f"synthetic_heatmap_{i}.png"
        with rstatus(f"[cyan]Rendering synthetic heatmap #{i}…"):
            plot_heatmap(synth, title=f"Synthetic LOB Depth #{i}", save_path=synth_path, show=False)
        score = get_ssim(real_path, synth_path)
        scores.append((f"synthetic_{i}", score))
        rlog(f"SSIM(real, synthetic_{i}) = {score:.4f}  [{synth_path.name}]")

    _print_ssim_table(scores)
    rrule("[bold green]Done[/bold green]")
