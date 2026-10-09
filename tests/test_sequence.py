import numpy as np
import pytest

from makam.data import HOP_SECONDS
from makam.sequence import add_position_channel, full_sequence, pitch_class_gram, random_crop, sliding_crops

STEP = 0.2
FRAMES = int(round(STEP / HOP_SECONDS))  # frames per step, ar. 69


def test_gram_shape_and_silence():
    pitch = np.zeros(FRAMES * 10 + 5)  # 10 full steps of silence + an incomplete one
    gram = pitch_class_gram(pitch, tonic_hz=200.0, step_seconds=STEP)
    assert gram.shape == (48, 10)
    assert gram.sum() == 0


def test_tonic_and_fifth_land_in_the_right_bins_in_time_order():
    tonic, fifth = 200.0, 200.0 * 2 ** (700 / 1200)
    pitch = np.concatenate([np.full(FRAMES * 3, tonic), np.full(FRAMES * 3, fifth)])
    gram = pitch_class_gram(pitch, tonic_hz=200.0, step_seconds=STEP)
    assert np.argmax(gram[:, 0]) == 0  # first steps: tonic, bin 0
    assert np.argmax(gram[:, -1]) == 28  # last steps: 700 cents / 25 = bin 28
    np.testing.assert_allclose(gram.sum(axis=0), 1.0, atol=1e-6)  # fully voiced steps sum to 1


def test_slightly_flat_tonic_still_in_bin_0():
    pitch = np.full(FRAMES, 200.0 * 2 ** (-5 / 1200))  # 5 cents flat
    gram = pitch_class_gram(pitch, tonic_hz=200.0, step_seconds=STEP)
    assert gram[0, 0] > 0.99


def test_octaves_fold_together():
    a = pitch_class_gram(np.full(FRAMES, 200.0), 200.0, STEP)
    b = pitch_class_gram(np.full(FRAMES, 400.0), 200.0, STEP)
    np.testing.assert_allclose(a, b)


def test_position_channel_ramps():
    out = add_position_channel(np.zeros((48, 5), dtype=np.float32))
    assert out.shape == (49, 5)
    np.testing.assert_allclose(out[-1], [0, 0.25, 0.5, 0.75, 1.0])


def test_random_crop_shape_and_position_slice():
    gram = np.ones((48, 100), dtype=np.float32)
    crop = random_crop(gram, 30, np.random.default_rng(0))
    assert crop.shape == (49, 30)
    assert 0.0 <= crop[-1, 0] < crop[-1, -1] <= 1.0


def test_short_recording_is_padded():
    gram = np.ones((48, 10), dtype=np.float32)
    crop = random_crop(gram, 30, np.random.default_rng(0))
    assert crop.shape == (49, 30)
    assert crop[:48, 10:].sum() == 0  # padding is silence


def test_full_sequence_spans_0_to_1():
    out = full_sequence(np.ones((48, 7), dtype=np.float32))
    assert out[-1, 0] == 0.0 and out[-1, -1] == 1.0


def test_sliding_crops_cover_the_whole_recording():
    gram = np.arange(48 * 100, dtype=np.float32).reshape(48, 100)
    crops = sliding_crops(gram, length=30, hop=15)
    assert crops.shape[1:] == (49, 30)
    assert crops[0, -1, 0] == 0.0  # first window starts at the beginning
    assert crops[-1, -1, -1] == 1.0  # last window ends at the very end
    np.testing.assert_array_equal(crops[-1, :48], gram[:, 70:])  # aligned to the end


def test_sliding_crops_short_recording_gives_one_padded_window():
    crops = sliding_crops(np.ones((48, 10), dtype=np.float32), length=30, hop=15)
    assert crops.shape == (1, 49, 30)


def test_region_keeps_crops_in_the_opening_with_absolute_positions():
    gram = np.ones((48, 300), dtype=np.float32)
    rng = np.random.default_rng(0)
    for _ in range(50):
        crop = random_crop(gram, 30, rng, region=1 / 3)
        assert crop[-1, -1] <= 100 / 300 + 1e-6  # never past the first third
    windows = sliding_crops(gram, 30, 15, region=1 / 3)
    assert windows[-1, -1, -1] == pytest.approx(100 / 300)  # last window ends at the end of the opening
    assert windows[0, -1, 0] == 0.0


def test_region_never_smaller_than_one_window():
    gram = np.ones((48, 60), dtype=np.float32)  # a third would be 20 steps, shorter than a window
    windows = sliding_crops(gram, 30, 15, region=1 / 3)
    assert windows.shape == (1, 49, 30)
