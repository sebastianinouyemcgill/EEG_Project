"""
Spectral feature extraction for EEG epochs.

Computes per-epoch band power and PSD features that can be:
  - Used directly with sklearn baselines for quick iteration
  - Logged as additional context during model evaluation
  - Visualised in exploration.ipynb to build intuition

Standard EEG frequency bands:
  Delta  0.5 – 4  Hz  (dominant in deep sleep N3)
  Theta  4   – 8  Hz  (N1, drowsiness)
  Alpha  8   – 13 Hz  (relaxed wakefulness)
  Sigma  12  – 15 Hz  (sleep spindles, N2)
  Beta   15  – 30 Hz  (active wakefulness)
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

SFREQ = 100   # Hz — must match preprocess.py

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 15.0),
    "beta":  (15.0, 30.0),
}


class SpectralFeatures(NamedTuple):
    band_power:    NDArray  # (N, n_bands)  absolute power per band
    rel_band_power: NDArray  # (N, n_bands)  power / total power
    psd:           NDArray  # (N, n_freqs)  full PSD
    freqs:         NDArray  # (n_freqs,)    frequency axis


def compute_psd(
    epochs: NDArray,   # (N, T)  float32
    sfreq: float = SFREQ,
) -> tuple[NDArray, NDArray]:
    """
    Welch PSD for a batch of epochs.
    Returns (psd, freqs): shapes (N, n_freqs) and (n_freqs,).
    """
    from scipy.signal import welch

    n_epochs = epochs.shape[0]
    freqs, _ = welch(epochs[0], fs=sfreq, nperseg=256)
    psd = np.zeros((n_epochs, len(freqs)), dtype=np.float32)

    for i, epoch in enumerate(epochs):
        _, psd[i] = welch(epoch, fs=sfreq, nperseg=256)

    return psd, freqs


def band_power_from_psd(
    psd: NDArray,    # (N, n_freqs)
    freqs: NDArray,  # (n_freqs,)
) -> tuple[NDArray, NDArray]:
    """
    Integrate PSD within each frequency band using the trapezoidal rule.
    Returns (abs_power, rel_power): both (N, n_bands).
    """
    n_epochs = psd.shape[0]
    n_bands  = len(BANDS)
    abs_power = np.zeros((n_epochs, n_bands), dtype=np.float32)

    for i, (lo, hi) in enumerate(BANDS.values()):
        mask = (freqs >= lo) & (freqs <= hi)
        abs_power[:, i] = np.trapezoid(psd[:, mask], freqs[mask], axis=1)

    total = abs_power.sum(axis=1, keepdims=True) + 1e-10
    rel_power = abs_power / total
    return abs_power, rel_power


def extract_features(
    epochs: NDArray,   # (N, T)  raw or normalised EEG
    sfreq: float = SFREQ,
) -> SpectralFeatures:
    """Full pipeline: epochs → SpectralFeatures."""
    psd, freqs         = compute_psd(epochs, sfreq)
    abs_power, rel_power = band_power_from_psd(psd, freqs)
    return SpectralFeatures(abs_power, rel_power, psd, freqs)


def feature_names() -> list[str]:
    """Column names for the flattened feature vector [abs_power | rel_power]."""
    return (
        [f"abs_{b}" for b in BANDS]
        + [f"rel_{b}" for b in BANDS]
    )
