"""YIN: estimating the pitch of a melody from audio (de Cheveigne and Kawahara, 2002)

Idea: a periodic signal looks like itself shifted by one period. For each short
frame and each candidate shift tau we measure how different the frame is from
itself shifted by tau. The smallest difference (after a normalization that
stops tau = 0 from always winning) gives the period, and sample_rate / period
is the frequency

Output format matches the recording dataset: one value every `hop` samples,
0 where no clear pitch is found
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def yin(
    audio: np.ndarray,
    sample_rate: int = 44100,
    frame: int = 2048,
    hop: int = 128,
    fmin: float = 60.0,
    fmax: float = 1000.0,
    threshold: float = 0.15,
    silence_db: float = -50.0,
    chunk: int = 2000,
) -> np.ndarray:
    """Pitch in Hz for each hop, 0 = unvoiced. With 44100 Hz and hop 128 this is the dataset's step"""
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) < frame:
        return np.zeros(0, dtype=np.float32)
    frames = sliding_window_view(audio, frame)[::hop]  # (n_frames, frame), no copy
    half = frame // 2
    tau_min = max(2, int(sample_rate / fmax))
    tau_max = min(half - 1, int(sample_rate / fmin))
    out = np.zeros(len(frames), dtype=np.float32)
    for start in range(0, len(frames), chunk):
        out[start : start + chunk] = _yin_chunk(
            frames[start : start + chunk], sample_rate, half, tau_min, tau_max, threshold, silence_db
        )
    return out


def _yin_chunk(x, sample_rate, half, tau_min, tau_max, threshold, silence_db):
    n = x.shape[1]
    a = x[:, :half]  # the integration window: first half of the frame

    # cross[tau] = sum_j a[j] * x[j + tau], all taus at once with the FFT
    size = 2 * n
    cross = np.fft.irfft(np.fft.rfft(x, size) * np.conj(np.fft.rfft(a, size)), size)[:, :half]

    # energies of the window and of the window shifted by tau, from a running sum
    sq = np.cumsum(np.pad(x**2, ((0, 0), (1, 0))), axis=1)
    energy_0 = sq[:, half][:, None]
    taus = np.arange(half)
    energy_tau = sq[:, taus + half] - sq[:, taus]
    diff = np.maximum(energy_0 + energy_tau - 2 * cross, 0.0)  # d(tau), clipped against round-off

    # cumulative mean normalized difference: d'(0) = 1, d'(tau) = d(tau) * tau / sum(d[1..tau])
    running = np.cumsum(diff[:, 1:], axis=1)
    cmnd = np.ones_like(diff)
    cmnd[:, 1:] = diff[:, 1:] * taus[1:] / np.maximum(running, 1e-12)

    search = cmnd[:, tau_min : tau_max + 1]
    below = search < threshold
    voiced = below.any(axis=1)
    tau = np.argmax(below, axis=1)  # first tau under the threshold
    # then slide down to the bottom of that dip
    rows = np.arange(len(x))
    for _ in range(search.shape[1]):
        nxt = np.minimum(tau + 1, search.shape[1] - 1)
        step = voiced & (search[rows, nxt] < search[rows, tau])
        if not step.any():
            break
        tau = np.where(step, nxt, tau)

    # parabolic interpolation around the minimum for sub-sample precision
    left = search[rows, np.maximum(tau - 1, 0)]
    mid = search[rows, tau]
    right = search[rows, np.minimum(tau + 1, search.shape[1] - 1)]
    denom = left - 2 * mid + right
    flat = np.abs(denom) < 1e-12  # a flat bottom has no parabola, keep tau as is
    shift = np.where(flat, 0.0, 0.5 * (left - right) / np.where(flat, 1.0, denom))
    period = tau + tau_min + np.clip(shift, -1, 1)

    loud = 10 * np.log10(np.mean(a**2, axis=1) + 1e-20) > silence_db
    f0 = np.where(voiced & loud, sample_rate / period, 0.0)
    return f0.astype(np.float32)
