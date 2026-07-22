"""Unit tests for the chest-displacement breathing models (``src.breathing_models``)."""

import numpy as np

from src.breathing_models import PATTERNS, chest_displacement


def _dominant_bpm(d, fs):
    x = d - d.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    band = (freqs >= 0.05) & (freqs <= 1.0)
    return freqs[band][np.argmax(spec[band])] * 60.0


def _window_rms(d, fs, win_s=1.0):
    win = int(fs * win_s)
    return np.array([np.sqrt(np.mean(d[i:i + win] ** 2))
                     for i in range(0, len(d) - win, win)])


def test_normal_rate_and_amplitude():
    t, d = chest_displacement("normal", duration=60, fs=200, rate_bpm=15, amplitude_mm=6)
    assert len(t) == len(d)
    assert abs(_dominant_bpm(d, 200) - 15) < 0.5
    assert 4.0 <= np.max(np.abs(d)) <= 12.0


def test_tachypnea_is_fast():
    _, d = chest_displacement("tachypnea", duration=60, fs=200)
    assert _dominant_bpm(d, 200) > 20


def test_bradypnea_is_slow():
    _, d = chest_displacement("bradypnea", duration=90, fs=200)
    assert _dominant_bpm(d, 200) < 12


def test_cheyne_stokes_has_apnea_gap_and_amplitude_modulation():
    _, d = chest_displacement("cheyne_stokes", duration=120, fs=200, cycle_s=60)
    rms = _window_rms(d, 200)
    # crescendo-decrescendo => strong amplitude modulation, and an apneic near-silent window
    assert rms.min() < 0.15 * rms.max()
    assert rms.std() / rms.mean() > 0.4


def test_obstructive_apnea_has_flat_segments():
    _, d = chest_displacement("obstructive_apnea", duration=90, fs=200,
                              breath_s=20, apnea_s=12)
    rms = _window_rms(d, 200)
    # at least one near-silent (apnea) window, but breathing is present elsewhere
    assert rms.min() < 0.1 * rms.max()
    assert rms.max() > 1.0


def test_all_patterns_dispatch_and_shape():
    for pattern in PATTERNS:
        t, d = chest_displacement(pattern, duration=30, fs=200)
        assert t.shape == d.shape
        assert abs(len(d) - 30 * 200) <= 1
        assert np.all(np.isfinite(d))
