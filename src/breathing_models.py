"""Synthetic chest-wall displacement models for five breathing patterns.

Each generator returns ``(t, d)`` where ``t`` is a time vector (seconds) and ``d`` is the
antero-posterior chest-wall displacement in **millimetres**. Parameter values follow the
clinical ranges in the project brief (see ``docs/physics_notes.md`` for sources):

* **normal**     12-20 breaths/min, roughly sinusoidal, ~4-12 mm amplitude
* **tachypnea**  > 20 breaths/min, shallower
* **bradypnea**  < 12 breaths/min
* **cheyne_stokes**  crescendo-decrescendo amplitude with apneic pauses, ~45-90 s cycle
* **obstructive_apnea**  normal breathing interrupted by near-absent chest motion

A small second harmonic is added so the waveform is not a pure sinusoid (real breathing
has a faster inspiration than expiration); the fundamental still dominates the spectrum.
"""

from __future__ import annotations

import numpy as np

PATTERNS = ("normal", "tachypnea", "bradypnea", "cheyne_stokes", "obstructive_apnea")


def _time(duration: float, fs: float) -> np.ndarray:
    return np.arange(0, duration, 1.0 / fs)


def _sinusoid(t: np.ndarray, rate_bpm: float, amplitude_mm: float,
              harmonic: float = 0.12) -> np.ndarray:
    f = rate_bpm / 60.0
    wave = np.sin(2 * np.pi * f * t) + harmonic * np.sin(2 * np.pi * 2 * f * t + 0.5)
    wave /= 1.0 + harmonic
    return amplitude_mm * wave


def normal(duration: float = 60.0, fs: float = 1000.0,
           rate_bpm: float = 15.0, amplitude_mm: float = 6.0, **_):
    t = _time(duration, fs)
    return t, _sinusoid(t, rate_bpm, amplitude_mm)


def tachypnea(duration: float = 60.0, fs: float = 1000.0,
              rate_bpm: float = 30.0, amplitude_mm: float = 3.0, **_):
    t = _time(duration, fs)
    return t, _sinusoid(t, rate_bpm, amplitude_mm)


def bradypnea(duration: float = 60.0, fs: float = 1000.0,
              rate_bpm: float = 8.0, amplitude_mm: float = 8.0, **_):
    t = _time(duration, fs)
    return t, _sinusoid(t, rate_bpm, amplitude_mm)


def cheyne_stokes(duration: float = 120.0, fs: float = 1000.0,
                  carrier_bpm: float = 15.0, amplitude_mm: float = 8.0,
                  cycle_s: float = 60.0, apnea_frac: float = 0.3, **_):
    """Crescendo-decrescendo breathing with a periodic apneic pause."""
    t = _time(duration, fs)
    phase = np.mod(t, cycle_s) / cycle_s          # 0..1 within each cycle
    active = 1.0 - apnea_frac
    # Raised-sine hump during the active portion, flat zero during apnea.
    envelope = np.where(phase < active,
                        np.sin(np.pi * np.clip(phase / active, 0, 1)) ** 2,
                        0.0)
    carrier = _sinusoid(t, carrier_bpm, 1.0)
    return t, amplitude_mm * envelope * carrier


def obstructive_apnea(duration: float = 90.0, fs: float = 1000.0,
                      rate_bpm: float = 15.0, amplitude_mm: float = 6.0,
                      breath_s: float = 20.0, apnea_s: float = 12.0,
                      apnea_residual: float = 0.03, **_):
    """Normal breathing interrupted by periods of near-absent chest motion."""
    t = _time(duration, fs)
    period = breath_s + apnea_s
    in_cycle = np.mod(t, period)
    mask = np.where(in_cycle < breath_s, 1.0, apnea_residual)
    return t, mask * _sinusoid(t, rate_bpm, amplitude_mm)


_DISPATCH = {
    "normal": normal,
    "tachypnea": tachypnea,
    "bradypnea": bradypnea,
    "cheyne_stokes": cheyne_stokes,
    "obstructive_apnea": obstructive_apnea,
}


def chest_displacement(pattern: str, duration: float = 60.0, fs: float = 1000.0,
                       **params):
    """Return ``(t, d_mm)`` for ``pattern`` (one of :data:`PATTERNS`)."""
    if pattern not in _DISPATCH:
        raise ValueError(f"unknown pattern {pattern!r}; choose from {PATTERNS}")
    return _DISPATCH[pattern](duration=duration, fs=fs, **params)
