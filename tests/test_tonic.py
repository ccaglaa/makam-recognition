import numpy as np
import pytest

from makam.features import hz_to_cents, pitch_class_histogram
from makam.tonic import (
    REFERENCE_HZ,
    all_rotations,
    estimate_makam_and_tonic,
    estimate_tonic,
    tonic_cents_above_reference,
    tonic_error_cents,
)

BW = 7.5


def _fake_melody(tonic_hz, intervals_cents, weights):
    """Frequencies (Hz) of a melody that spends `weights` time on each interval above the tonic"""
    notes = [np.full(w, tonic_hz * 2 ** (c / 1200)) for c, w in zip(intervals_cents, weights)]
    return np.concatenate(notes)


def _template(intervals_cents, weights):
    """Tonic-relative histogram, built the same way as the real templates"""
    melody = _fake_melody(200.0, intervals_cents, weights)
    return pitch_class_histogram(hz_to_cents(melody, 200.0), BW)


def test_rotation_row_s_is_np_roll():
    h = np.arange(6.0)
    rot = all_rotations(h)
    assert rot.shape == (6, 6)
    np.testing.assert_array_equal(rot[2], [2, 3, 4, 5, 0, 1])


@pytest.mark.parametrize("tonic_hz", [196.0, 233.1, 311.1, 415.3])
def test_tonic_found_for_a_transposed_melody(tonic_hz):
    shape = ([0, 204, 385, 498, 702], [400, 150, 200, 250, 300])  # a Rast-like scale
    template = _template(*shape)
    melody = _fake_melody(tonic_hz, *shape)
    hist = pitch_class_histogram(hz_to_cents(melody, REFERENCE_HZ), BW)  # no tonic used here
    est = estimate_tonic(hist, template, "bhattacharyya", BW)
    assert abs(tonic_error_cents(est, tonic_cents_above_reference(tonic_hz))) <= BW


def test_joint_estimation_picks_the_right_makam_and_tonic():
    shape_a = ([0, 204, 385, 498, 702], [400, 150, 200, 250, 300])
    shape_b = ([0, 150, 294, 498, 702], [400, 150, 200, 300, 150])
    labels = np.array(["A", "B"])
    templates = np.stack([_template(*shape_a), _template(*shape_b)])
    melody = _fake_melody(260.0, *shape_b)
    hist = pitch_class_histogram(hz_to_cents(melody, REFERENCE_HZ), BW)
    makam, est = estimate_makam_and_tonic(hist, labels, templates, "bhattacharyya", BW)
    assert makam == "B"
    assert abs(tonic_error_cents(est, tonic_cents_above_reference(260.0))) <= BW


def test_tonic_error_wraps_around_the_octave():
    assert tonic_error_cents(1195.0, 5.0) == pytest.approx(-10.0)
    assert tonic_error_cents(5.0, 1195.0) == pytest.approx(10.0)
    assert tonic_error_cents(700.0, 0.0) == pytest.approx(-500.0)  # a fifth up = a fourth down


def test_reference_itself_is_zero_cents():
    assert tonic_cents_above_reference(REFERENCE_HZ) == pytest.approx(0.0)
    assert tonic_cents_above_reference(REFERENCE_HZ / 2) == pytest.approx(0.0)  # octave below
