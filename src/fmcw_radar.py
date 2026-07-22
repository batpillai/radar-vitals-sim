"""Simplified FMCW radar model (range-oscillation).

A full FMCW radar transmits a frequency-swept chirp; the beat frequency between the
transmitted and received signals is proportional to target range. For breathing detection
the informative quantity is how the chest's **range** oscillates over time, so this
simplified model represents range directly as ``R(t) = R0 + d(t)`` instead of synthesizing
and mixing chirps. It captures the same breathing-detection principle at far lower
implementation complexity, and (see the tests) recovers the same rate as the CW path.
"""

from __future__ import annotations

import numpy as np


def range_series(d_mm, fs: float = 1000.0, r0_m: float = 1.0) -> np.ndarray:
    """Chest range over time (metres): baseline ``r0_m`` plus displacement ``d_mm``."""
    return r0_m + np.asarray(d_mm, dtype=np.float64) / 1000.0


def range_bin(range_m, bin_size_m: float = 0.05) -> np.ndarray:
    """Quantise a range series into discrete range-bin indices."""
    return np.round(np.asarray(range_m, dtype=np.float64) / bin_size_m).astype(int)


def range_time_map(d_mm, r0_m: float = 1.0, n_bins: int = 64,
                   bin_size_m: float = 0.002, spread: float = 1.0) -> np.ndarray:
    """A range-time intensity map (n_bins x n_samples) for visualisation.

    Each column is a Gaussian bump centred on the chest's current range bin, so the map
    shows the range bin gently oscillating with breathing -- the FMCW analogue of the
    CW phase signal. Purely for plotting; not used by the rate estimator.
    """
    rng = range_series(d_mm, r0_m=r0_m)
    centre = (rng - (r0_m - n_bins / 2 * bin_size_m)) / bin_size_m
    bins = np.arange(n_bins)[:, None]
    return np.exp(-0.5 * ((bins - centre[None, :]) / spread) ** 2)
