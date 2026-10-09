import numpy as np
import pytest

from makam.features import (
    bin_centers,
    fold_to_octave,
    hz_to_cents,
    pitch_class_histogram,
)


def test_tonic_is_zero_cents_and_octave_is_1200():
    cents = hz_to_cents(np.array([220.0, 440.0, 110.0]), tonic_hz=220.0)
    np.testing.assert_allclose(cents, [0.0, 1200.0, -1200.0])


def test_unvoiced_samples_are_dropped():
    cents = hz_to_cents(np.array([0.0, 220.0, 0.0]), tonic_hz=220.0)
    assert len(cents) == 1


def test_bad_tonic_raises():
    with pytest.raises(ValueError):
        hz_to_cents(np.array([220.0]), tonic_hz=0.0)


def test_fold_wraps_negatives_and_higher_octaves():
    np.testing.assert_allclose(fold_to_octave(np.array([-100.0, 1300.0, 700.0])), [1100.0, 100.0, 700.0])


def test_histogram_sums_to_one_and_peaks_at_the_note():
    # 1000 samples exactly on a perfect fifth (702 cents), in two octaves
    cents = np.concatenate([np.full(500, 702.0), np.full(500, 702.0 + 1200)])
    hist = pitch_class_histogram(cents, bin_width=7.5, smoothing_cents=None)
    assert hist.sum() == pytest.approx(1.0)
    peak = bin_centers(7.5)[np.argmax(hist)]
    assert abs(peak - 702.0) <= 7.5


def test_smoothing_wraps_around_the_octave():
    # A note right at the tonic should leak into the LAST bins too (circularity)
    hist = pitch_class_histogram(np.full(100, 1.0), bin_width=7.5, smoothing_cents=15)
    assert hist[-1] > 0.01
