"""Turning a pitch track into a time sequence a neural network can read

A histogram forgets time. Here we keep it: the melody is cut into short steps
(0.2 s by default) and each step becomes a small folded pitch histogram
(48 bins of 25 cents). Stacked side by side they form a "pitch-class-gram",
a 2D array of shape (n_bins, n_steps), like a very coarse spectrogram

Everything here is plain numpy, so it can be tested without PyTorch
"""

from __future__ import annotations

import numpy as np

from makam.data import HOP_SECONDS
from makam.features import OCTAVE_CENTS


def pitch_class_gram(
    pitch_hz: np.ndarray,
    tonic_hz: float,
    step_seconds: float = 0.2,
    n_bins: int = 48,
) -> np.ndarray:
    """Shape (n_bins, n_steps): for each time step, the fraction of it spent on each pitch class

    Unvoiced frames count as nothing, so a silent step is a column of zeros and
    a half-silent step sums to 0.5. Folding makes the octave problem irrelevant
    """
    frames_per_step = max(1, int(round(step_seconds / HOP_SECONDS)))
    n_steps = len(pitch_hz) // frames_per_step  # the last incomplete step is dropped
    pitch = pitch_hz[: n_steps * frames_per_step]

    voiced = pitch > 0
    cents = np.zeros_like(pitch, dtype=np.float64)
    cents[voiced] = OCTAVE_CENTS * np.log2(pitch[voiced] / tonic_hz)
    # Bins centered on the tonic: bin 0 covers [-12.5, +12.5) cents, so a note sung
    # exactly on the tonic does not straddle a bin edge
    bin_width = OCTAVE_CENTS / n_bins
    bins = np.floor(np.mod(cents + bin_width / 2, OCTAVE_CENTS) / bin_width).astype(int)

    step_of_frame = np.arange(len(pitch)) // frames_per_step
    gram = np.zeros((n_bins, n_steps), dtype=np.float32)
    # add 1 at (bin, step) for every voiced frame, repeated pairs accumulate
    np.add.at(gram, (bins[voiced], step_of_frame[voiced]), 1.0)
    return gram / frames_per_step


def add_position_channel(gram: np.ndarray, start: float = 0.0, end: float = 1.0) -> np.ndarray:
    """Append one row going linearly from start to end: where we are in the piece

    For a whole recording this is 0 -> 1. For a crop it is the crop's own slice
    of that ramp, so the network knows whether it is looking at the opening or the ending
    """
    n_steps = gram.shape[1]
    position = np.linspace(start, end, n_steps, dtype=np.float32)[np.newaxis, :]
    return np.vstack([gram, position])


def random_crop(
    gram: np.ndarray, length: int, rng: np.random.Generator, region: float = 1.0
) -> np.ndarray:
    """A random window of `length` steps WITH its position channel, shape (n_bins + 1, length)

    region < 1 keeps the window inside the first `region` fraction of the piece
    (e.g. 1/3 = the opening). Positions stay relative to the WHOLE piece
    Recordings shorter than `length` are padded with silence at the end
    """
    n_steps = gram.shape[1]
    if n_steps <= length:
        padded = np.zeros((gram.shape[0], length), dtype=np.float32)
        padded[:, :n_steps] = gram
        return add_position_channel(padded, 0.0, length / max(n_steps, 1))
    limit = _region_limit(n_steps, length, region)
    start = int(rng.integers(0, limit - length + 1))
    crop = gram[:, start : start + length]
    return add_position_channel(crop, start / n_steps, (start + length) / n_steps)


def _region_limit(n_steps: int, length: int, region: float) -> int:
    """Last step a window may reach: the end of the region, but never shorter than one window"""
    return min(n_steps, max(length, int(round(n_steps * region))))


def sliding_crops(gram: np.ndarray, length: int, hop: int, region: float = 1.0) -> np.ndarray:
    """Overlapping windows covering the whole recording, shape (n_windows, n_bins + 1, length)

    Used at test time so the network sees exactly what it saw in training:
    windows of `length` steps, each with its own slice of the position ramp.
    The last window is aligned to the end of the region (the whole piece by
    default) so no part of it is skipped
    """
    n_steps = gram.shape[1]
    if n_steps <= length:
        return random_crop(gram, length, np.random.default_rng(0))[np.newaxis]  # one padded window
    limit = _region_limit(n_steps, length, region)
    starts = list(range(0, limit - length + 1, hop))
    if starts[-1] != limit - length:
        starts.append(limit - length)
    return np.stack([
        add_position_channel(gram[:, s : s + length], s / n_steps, (s + length) / n_steps)
        for s in starts
    ])


def full_sequence(gram: np.ndarray) -> np.ndarray:
    """The whole recording with its position channel, used at test time"""
    return add_position_channel(gram, 0.0, 1.0)
