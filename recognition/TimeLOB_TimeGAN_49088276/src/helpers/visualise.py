"""
Generate LOB depth heatmaps and compute SSIM between real vs synthetic images.
Refactored to be faster, cleaner, and compatible with the new modules/utils.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim

from args import Options
from constants import NUM_LEVELS
from src.dataset import load_data
from src.helpers.constants import OUTPUT_DIR
from src.modules import TimeGAN


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
    max_vol = float(np.max([vols_ask.max(initial=0), vols_bid.max(initial=0)])) or 1.0
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
    c_ask = np.stack([
        np.full_like(y_ask, 0.99),  # r
        np.full_like(y_ask, 0.05),  # g
        np.full_like(y_ask, 0.05),  # b
        a_ask.astype(np.float32).ravel(),  # A
    ], axis=1)
    c_bid = np.stack([
        np.full_like(y_ask, 0.05),  # r
        np.full_like(y_ask, 0.05),  # g
        np.full_like(y_ask, 0.99),  # b
        a_bid.astype(np.float32).ravel(),  # A
    ], axis=1)

    # limits
    pmin = float(np.minimum(prices_ask.min(initial=0), prices_bid.min(initial=0)))
    pmax = float(np.maximum(prices_ask.max(initial=0), prices_bid.max(initial=0)))

    # plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax.set_ylim(pmin, pmax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    if title:
        ax.set_title(title)

    ax.scatter(x_ask, y_ask, c=c_ask)
    ax.scatter(x_bid, y_bid, c=c_bid)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

if "__main__" == __name__:
    # cli
    opt = Options().parse()

    # data
    train, val, test = load_data(opt)

    # model (load weights)
    model = TimeGAN(opt, train, val, test, load_weights=True)

    # real heatmap from test data
    real_path = Path(OUTPUT_DIR) / "real.png"
    plot_heatmap(test, title="Real LOB Depth", save_path=real_path, show=False)

    for i in range(3):
        synth = model.generate(num_rows=len(test))
        synth_path = Path(OUTPUT_DIR) / f"synthetic_heatmap_{i}.png"
        plot_heatmap(synth, title=f"Synthetic LOB Depth #{i}", save_path=synth_path, show=False)
        score = get_ssim(real_path, synth_path)
        print(f"SSIM(real, synthetic_{i}) = {score:.4f}")
