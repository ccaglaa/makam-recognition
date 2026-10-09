"""Turning a pitch track (Hz) into features that describe a makam

Key idea: we don't care about absolute frequency (a singer can perform the
same makam higher or lower). We care about each note's distance FROM THE
TONIC, measured in cents

cents = 1200 * log2(f / tonic)

  * 0 cents = the tonic itself
  * 1200 cents = one octave above (frequency doubled)
  * 100 cents = one piano semitone (Western 12-tone equal temperament)
  * Turkish theory splits the whole tone (~204 cents) into 9 "commas"
    (ar. 22.6 cents each), so many makam notes fall BETWEEN piano keys
"""

from __future__ import annotations

import numpy as np

OCTAVE_CENTS = 1200.0


def hz_to_cents(pitch_hz: np.ndarray, tonic_hz: float) -> np.ndarray:
    """Convert voiced frequencies to cents relative to the tonic

    Unvoiced samples (0 Hz) are DROPPED, because log2(0) is -infinity
    """
    if tonic_hz <= 0:
        raise ValueError(f"tonic must be positive, got {tonic_hz}")
    voiced = pitch_hz[pitch_hz > 0]
    return OCTAVE_CENTS * np.log2(voiced / tonic_hz)


def fold_to_octave(cents: np.ndarray) -> np.ndarray:
    """Map every note into [0, 1200): an A in any octave becomes 'the same' A

    This is the 'pitch-class' view, np.mod handles negatives correctly:
    -100 cents (just below the tonic) -> 1100 cents
    """
    return np.mod(cents, OCTAVE_CENTS)


def pitch_class_histogram(
    cents: np.ndarray,
    bin_width: float = 7.5,
    smoothing_cents: float | None = 7.5,
) -> np.ndarray:
    """Octave-folded histogram of time spent on each pitch, normalized to sum to 1

    bin_width: resolution in cents, 7.5 cents = 160 bins per octave, fine
        enough to separate notes ar. 1 comma (ar. 22.6 cents) apart
    smoothing_cents: std of a Gaussian blur applied around the circle
        (the octave wraps: bin 159 is a neighbour of bin 0) smoothing makes
        histograms of different performances easier to compare, None = off
    """
    n_bins = int(round(OCTAVE_CENTS / bin_width))
    folded = fold_to_octave(cents)
    hist, _ = np.histogram(folded, bins=n_bins, range=(0.0, OCTAVE_CENTS))
    hist = hist.astype(np.float64)

    if smoothing_cents:
        hist = _circular_gaussian_smooth(hist, sigma_bins=smoothing_cents / bin_width)

    total = hist.sum()
    return hist / total if total > 0 else hist


def bin_centers(bin_width: float = 7.5) -> np.ndarray:
    """Cents value at the center of each histogram bin (useful for plotting)"""
    n_bins = int(round(OCTAVE_CENTS / bin_width))
    return (np.arange(n_bins) + 0.5) * bin_width


def top_peak_cents(
    hist: np.ndarray,
    bin_width: float = 7.5,
    exclude_around_tonic: float = 50.0,
) -> float:
    """Position (in cents) of the highest histogram bin that is NOT the tonic

    The tonic is almost always the biggest peak, so we hide every bin within
    `exclude_around_tonic` cents of 0 (or of 1200: the octave is a circle)
    """
    centers = bin_centers(bin_width)
    distance_to_tonic = np.minimum(centers, OCTAVE_CENTS - centers)  # circular distance
    masked = np.where(distance_to_tonic > exclude_around_tonic, hist, -np.inf)
    return float(centers[np.argmax(masked)])


def _circular_gaussian_smooth(hist: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Gaussian blur on a circular array, done with the FFT

    Multiplying in the frequency domain = circular convolution, so the
    wrap around at the octave boundary is handled for free
    """
    n = len(hist)
    offsets = np.arange(n)
    offsets = np.minimum(offsets, n - offsets)  # circular distance to bin 0
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    smoothed = np.real(np.fft.ifft(np.fft.fft(hist) * np.fft.fft(kernel)))
    # FFT round-off leaves tiny negatives (ar. -1e-18) in empty regions
    # A histogram can't be negative, and sqrt or log of a negative gives NaN later on
    return np.maximum(smoothed, 0.0)
