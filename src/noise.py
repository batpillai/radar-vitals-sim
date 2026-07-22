"""Additive thermal noise and phase jitter for radar I/Q signals.

Real radar returns are not clean. Two effects are modelled here so validation is not run
on a noiseless toy case:

* **Thermal noise** -- additive white Gaussian noise on both I and Q, scaled to a target
  signal-to-noise ratio (SNR) in dB. SNR is defined on the total I/Q power
  ``mean(I^2 + Q^2)`` versus the total added-noise power.
* **Phase jitter** -- a slow random walk added to the instantaneous phase (oscillator /
  clock instability), leaving the envelope amplitude unchanged.
"""

from __future__ import annotations

import numpy as np


def add_thermal_noise(i, q, snr_db: float, seed: int | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Add white Gaussian noise to I/Q to reach ``snr_db`` (total-power definition)."""
    rng = np.random.default_rng(seed)
    i = np.asarray(i, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    sig_power = np.mean(i ** 2 + q ** 2)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_power / 2.0)  # split across the two components
    return (i + rng.normal(0.0, sigma, i.shape),
            q + rng.normal(0.0, sigma, q.shape))


def add_phase_jitter(i, q, sigma_rad: float, seed: int | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Add a random-walk phase perturbation, preserving the envelope amplitude."""
    rng = np.random.default_rng(seed)
    i = np.asarray(i, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    phase = np.arctan2(q, i)
    amp = np.hypot(i, q)
    jitter = np.cumsum(rng.normal(0.0, sigma_rad, phase.shape))
    return amp * np.cos(phase + jitter), amp * np.sin(phase + jitter)


def measured_snr_db(clean_i, clean_q, noisy_i, noisy_q) -> float:
    """Measured SNR (dB) between a clean I/Q pair and its noisy version."""
    clean_i, clean_q = np.asarray(clean_i), np.asarray(clean_q)
    sig = np.mean(clean_i ** 2 + clean_q ** 2)
    noise = np.mean((np.asarray(noisy_i) - clean_i) ** 2
                    + (np.asarray(noisy_q) - clean_q) ** 2)
    return float(10.0 * np.log10(sig / noise))
