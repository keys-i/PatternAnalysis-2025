from __future__ import annotations
from typing import Iterable, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

Metric = Literal["spread", "mpr"]

def extract_seq_lengths(
    sequences: Iterable[NDArray[np.floating]]
) -> Tuple[NDArray[np.int32], int]:
    lengths = np.asarray([int(s.shape[0]) for s in sequences], dtype=np.int32)
    return lengths, int(lengths.max(initial=0))

def sample_noise(
        batch_size: int,
        z_dim: int,
        seq_len: int,
        *,
        mean: float | None = None,
        std: float | None = None,
        rng: np.random.Generator | None = None,
) -> NDArray[np.float32]:
    if rng is None:
        rng = np.random.default_rng()

    if (mean is None) ^ (std is None):
        raise ValueError("Provide both mean and std, or neither")

    if mean is None and std is None:
        out = rng.random((batch_size, seq_len, z_dim), dtype=np.float32)
    else:
        interval = float(std) * np.sqrt(12.0)
        lo = float(mean) - interval / 2.0
        hi = float(mean) + interval / 2.0
        out = rng.uniform(lo, hi, size=(batch_size, seq_len, z_dim)).astype(np.float32)

    return out

def minmax_scale(
    data: NDArray[np.floating],
    epsilon: float = 1e-7
)-> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    if data.ndim != 3:
        raise ValueError(f"Expected data with 3 dimensions [N, T, F], got shape {data.shape}")

    fmin = np.min(data, axis=(0, 1)).astype(np.float32)
    fmax = np.max(data, axis=(0, 1)).astype(np.float32)
    denom = (fmax - fmin).astype(np.float32)

    norm = (data.astype(np.float32) - fmin) / (denom + epsilon)
    return norm, fmin, fmax

def minmax_inverse(
    norm: NDArray[np.floating],
    fmin: NDArray[np.floating],
    fmax: NDArray[np.floating],
) -> NDArray[np.float32]:
    """
    Inverse of `minmax_scale`.

    Args:
        norm: scaled data [N,T,F] or [...,F]
        fmin: per-feature minima [F]
        fmax: per-feature maxima [F]

    Returns:
        original-scale data, float32
    """
    fmin = np.asarray(fmin, dtype=np.float32)
    fmax = np.asarray(fmax, dtype=np.float32)
    return norm.astype(np.float32) * (fmax - fmin) + fmin

def _spread(series: NDArray[np.floating]) -> NDArray[np.float64]:
    """
    Compute spread = best_ask - best_bid from a 2D array [T, F] with
    columns: best ask at index 0 and best bid at index 2.
    """
    if series.ndim != 2 or series.shape[1] < 3:
        raise ValueError("Expected shape [T, >=3]; columns 0 (ask) and 2 (bid) required.")
    return (series[:, 0] - series[:, 2]).astype(np.float64)


def _midprice_returns(series: NDArray[np.floating]) -> NDArray[np.float64]:
    """
    Compute log midprice returns from a 2D array [T, F] with ask at 0 and bid at 2.
    """
    if series.ndim != 2 or series.shape[1] < 3:
        raise ValueError("Expected shape [T, >=3]; columns 0 (ask) and 2 (bid) required.")
    mid = 0.5 * (series[:, 0] + series[:, 2])
    # avoid log(0)
    mid = np.clip(mid, a_min=np.finfo(np.float64).tiny, a_max=None)
    r = np.log(mid[1:]) - np.log(mid[:-1])
    return r.astype(np.float64)

def kl_divergence_hist(
        real: NDArray[np.floating],
        fake: NDArray[np.floating],
        metric: Literal["spread", "mpr"] = "spread",
        *,
        bins: int = 100,
        show_plot: bool = False,
        epsilon: float = 1e-12
) -> float:
    if real.ndim != 2 or fake.ndim != 2:
        raise ValueError("Inputs must be 2D arrays [T, F].")

    if metric == "spread":
        r_series = _spread(real)
        f_series = _spread(fake)
    elif metric == "mpr":
        r_series = _midprice_returns(real)
        f_series = _midprice_returns(fake)
    else:
        raise ValueError("metric must be 'spread' or 'mpr'.")

    lo = float(min(r_series.min(initial=0.0), f_series.min(initial=0.0)))
    hi = float(max(r_series.max(initial=0.0), f_series.max(initial=0.0)))

    # if degenerate, expand a hair to avoid zero-width bins
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6

    r_hist, edges = np.histogram(r_series, bins=bins, range=(lo, hi), density=False)
    f_hist, _ = np.histogram(f_series, bins=edges, density=False)

    # convert to probability masses with smoothing
    r_p = (r_hist.astype(np.float64) + epsilon)
    f_p = (f_hist.astype(np.float64) + epsilon)
    r_p /= r_p.sum()
    f_p /= f_p.sum()

    # KL(real || fake) = sum p * log(p/q)
    mask = r_p > 0  # should be true after smoothing, but keep for safety
    kl = np.sum(r_p[mask] * (np.log(r_p[mask]) - np.log(f_p[mask])))

    if show_plot:
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(centers, r_p, label="real")
        plt.plot(centers, f_p, label="fake")
        plt.title(f"Histogram ({metric}); KL={kl:.4g}")
        plt.legend()
        plt.show()

    # numerical guard: KL should be >= 0
    return float(max(kl, 0.0))