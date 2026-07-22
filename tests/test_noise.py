"""Tests for the thermal-noise and phase-jitter models (``src.noise``)."""

import numpy as np

from src.breathing_models import chest_displacement
from src.cw_radar import cw_iq
from src.demodulation import recover_displacement
from src.noise import add_phase_jitter, add_thermal_noise, measured_snr_db
from src.rate_extraction import estimate_bpm

FS = 200.0


def test_thermal_noise_hits_target_snr():
    t = np.arange(0, 20, 1 / 1000.0)
    d = 6.0 * np.sin(2 * np.pi * 0.25 * t)
    i, q = cw_iq(d)

    ni, nq = add_thermal_noise(i, q, snr_db=10.0, seed=0)

    assert abs(measured_snr_db(i, q, ni, nq) - 10.0) < 1.0


def test_rate_survives_high_snr_and_quality_drops_at_low_snr():
    _, d = chest_displacement("normal", duration=120, fs=FS, rate_bpm=15)
    i, q = cw_iq(d)

    hi_i, hi_q = add_thermal_noise(i, q, snr_db=20.0, seed=1)
    bpm_hi, snr_hi = estimate_bpm(recover_displacement(hi_i, hi_q), FS)
    assert abs(bpm_hi - 15) < 1.0

    lo_i, lo_q = add_thermal_noise(i, q, snr_db=-5.0, seed=1)
    _, snr_lo = estimate_bpm(recover_displacement(lo_i, lo_q), FS)
    assert snr_lo < snr_hi


def test_phase_jitter_preserves_amplitude_but_adds_phase_noise():
    t = np.arange(0, 10, 1 / 1000.0)
    d = 6.0 * np.sin(2 * np.pi * 0.25 * t)
    i, q = cw_iq(d, amplitude=1.0)

    ji, jq = add_phase_jitter(i, q, sigma_rad=0.01, seed=2)

    # envelope amplitude is unchanged by phase jitter
    assert np.allclose(np.hypot(ji, jq), np.hypot(i, q), atol=1e-9)
    # but the signal itself is perturbed
    assert not np.allclose(ji, i, atol=1e-3)
