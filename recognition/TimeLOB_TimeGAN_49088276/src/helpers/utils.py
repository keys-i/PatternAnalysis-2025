#!/usr/bin/env python3
"""
Utility metrics and helpers for TimeGAN on LOBSTER data.

This module provides:
- Sequence utilities: length extraction, noise sampling.
- Feature scaling: min–max forward and inverse transforms.
- Market metrics: spread, mid-price returns, KL divergence on histograms.
- Visual metrics: SSIM between saved heatmap images.
- Consistency metrics: temporal correlation of deltas, latent divergence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity as ssim
from skimage.util import img_as_float

Metric = Literal["spread", "mpr"]


def extract_seq_lengths(
    sequences: Iterable[NDArray[np.floating]],
) -> Tuple[NDArray[np.int32], int]:
    """Return per-sequence lengths and the maximum length.

    Args:
        sequences: Iterable of arrays where the first dimension is time.

    Returns:
        lengths: Vector of sequence lengths (int32).
        max_len: Maximum length across sequences.
    """
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
    """Sample noise windows for the generator.

    If mean and std are provided, draws from a uniform distribution whose
    standard deviation matches ``std`` (variance of uniform a...b is (b-a)^2/12).
    Otherwise, draws from U[0,1].

    Args:
        batch_size: Number of windows to sample.
        z_dim: Latent dimensionality per time step.
        seq_len: Time length per window.
        mean: Optional target mean for uniform sampling.
        std: Optional target standard deviation for uniform sampling.
        rng: Optional NumPy Generator for reproducibility.

    Returns:
        Array of shape [batch_size, seq_len, z_dim] (float32).

    Raises:
        ValueError: If only one of mean or std is provided.
    """
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
    epsilon: float = 1e-7,
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Apply feature-wise min–max scaling across all time and windows.

    Args:
        data: Array of shape [N, T, F] to be scaled.
        epsilon: Small constant to avoid division by zero.

    Returns:
        norm: Scaled data in [0,1], shape [N, T, F].
        fmin: Per-feature minima, shape [F].
        fmax: Per-feature maxima, shape [F].

    Raises:
        ValueError: If input is not 3D.
    """
    if data.ndim != 3:
        raise ValueError(
            f"Expected data with 3 dimensions [N, T, F], got shape {data.shape}"
        )

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
    """Invert a min–max scaling transform.

    Args:
        norm: Scaled data of shape [N, T, F] or [..., F].
        fmin: Per-feature minima [F].
        fmax: Per-feature maxima [F].

    Returns:
        Data restored to original scale (float32).
    """
    fmin = np.asarray(fmin, dtype=np.float32)
    fmax = np.asarray(fmax, dtype=np.float32)
    return norm.astype(np.float32) * (fmax - fmin) + fmin


def _spread(series: NDArray[np.floating]) -> NDArray[np.float64]:
    """Compute best-level spread from a [T, F] series.

    Assumes column 0 is best ask price and column 2 is best bid price.

    Args:
        series: Array of shape [T, F].

    Returns:
        Spread time series [T] (float64).

    Raises:
        ValueError: If shape is not [T, >=3].
    """
    if series.ndim != 2 or series.shape[1] < 3:
        raise ValueError(
            "Expected shape [T, >=3]; columns 0 (ask) and 2 (bid) required."
        )
    return (series[:, 0] - series[:, 2]).astype(np.float64)


def _midprice_returns(series: NDArray[np.floating]) -> NDArray[np.float64]:
    """Compute log mid-price returns from a [T, F] series.

    Uses columns 0 (ask) and 2 (bid). Mid is clipped away from zero for numerical stability.

    Args:
        series: Array of shape [T, F].

    Returns:
        Log returns of mid-price, shape [T-1] (float64).

    Raises:
        ValueError: If shape is not [T, >=3].
    """
    if series.ndim != 2 or series.shape[1] < 3:
        raise ValueError(
            "Expected shape [T, >=3]; columns 0 (ask) and 2 (bid) required."
        )
    mid = 0.5 * (series[:, 0] + series[:, 2])
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
    epsilon: float = 1e-12,
) -> float:
    """Estimate KL divergence between real and fake distributions via histograms.

    Args:
        real: Real series [T, F].
        fake: Synthetic series [T, F].
        metric: Which 1D series to compare: 'spread' or 'mpr' (mid-price returns).
        bins: Number of histogram bins.
        show_plot: If True, display the normalized histograms.
        epsilon: Smoothing mass added to each bin to avoid zeros.

    Returns:
        Non-negative KL(real || fake) as a float.

    Raises:
        ValueError: If inputs are not [T, F] or if metric is invalid.
    """
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
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6  # avoid zero-width range

    r_hist, edges = np.histogram(r_series, bins=bins, range=(lo, hi), density=False)
    f_hist, _ = np.histogram(f_series, bins=edges, density=False)

    r_p = r_hist.astype(np.float64) + epsilon
    f_p = f_hist.astype(np.float64) + epsilon
    r_p /= r_p.sum()
    f_p /= f_p.sum()

    kl = float(np.sum(r_p * (np.log(r_p) - np.log(f_p))))
    if show_plot:
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(centers, r_p, label="real")
        plt.plot(centers, f_p, label="fake")
        plt.title(f"Histogram ({metric}); KL={kl:.4g}")
        plt.legend()
        plt.show()
    return max(kl, 0.0)


def get_ssim(img1_path: Path | str, img2_path: Path | str) -> float:
    """Compute SSIM between two image files read via matplotlib."""
    img1 = img_as_float(plt.imread(str(img1_path)))
    img2 = img_as_float(plt.imread(str(img2_path)))
    if img1.ndim == 2:
        img1 = img1[..., None]
    if img2.ndim == 2:
        img2 = img2[..., None]
    return float(ssim(img1, img2, channel_axis=2, data_range=1.0))


def get_kl_metrics(
    real_2d: NDArray, fake_2d: NDArray, bins: int = 100
) -> Dict[str, float]:
    """Compute KL divergences for spread and mid-price returns.

    Args:
        real_2d: Real series [T, F].
        fake_2d: Synthetic series [T, F].
        bins: Number of histogram bins.

    Returns:
        Dict with keys 'spread' and 'midprice_returns'.
    """
    t = min(len(real_2d), len(fake_2d))
    real, fake = real_2d[:t], fake_2d[:t]
    kl_spread = kl_divergence_hist(real, fake, metric="spread", bins=bins)
    kl_mpr = kl_divergence_hist(real, fake, metric="mpr", bins=bins)
    return {"spread": float(kl_spread), "midprice_returns": float(kl_mpr)}


def temporal_consistency(real: NDArray, fake: NDArray) -> float:
    """Measure correlation between successive deltas in real and synthetic series.

    Computes first differences versus time for each feature, then averages the
    Pearson correlation across overlapping features.

    Args:
        real: Real matrix [T, F].
        fake: Synthetic matrix [T, F].

    Returns:
        Mean correlation of deltas over features (float).
    """

    def deltas(x: NDArray) -> NDArray:
        return np.diff(x, axis=0)

    real_d, fake_d = deltas(real), deltas(fake)
    f = min(real_d.shape[1], fake_d.shape[1])
    if f == 0:
        return 0.0
    corrs = []
    for i in range(f):
        c = np.corrcoef(real_d[:, i], fake_d[:, i])[0, 1]
        corrs.append(c)
    return float(np.nan_to_num(np.mean(corrs)))


def latent_divergence(real: NDArray, fake: NDArray) -> float:
    """Compute a simple Frobenius distance between aligned real and fake matrices.

    Args:
        real: Real matrix [T, F] or [T, d].
        fake: Synthetic matrix [T, F] or [T, d].

    Returns:
        Normalized Frobenius norm divided by the number of aligned rows.
    """
    t = min(len(real), len(fake))
    if t == 0:
        return 0.0
    diff = real[:t] - fake[:t]
    return float(np.linalg.norm(diff, ord="fro") / t)
