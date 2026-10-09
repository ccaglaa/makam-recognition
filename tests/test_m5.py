import numpy as np
import pytest

from makam.data import HOP_SECONDS
from makam.pipeline import MakamRecognizer
from makam.pitch import yin
from makam.symbtr import KOMA_CENTS, Score, load_score, score_to_pitch_track, synthesize

SR = 44100


def _tone(f0, seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return sum(a * np.sin(2 * np.pi * f0 * (k + 1) * t) for k, a in enumerate([1.0, 0.5, 0.3, 0.2]))


@pytest.mark.parametrize("f0", [110.0, 233.08, 440.0])
def test_yin_finds_the_pitch_within_one_cent(f0):
    p = yin(_tone(f0), SR)
    assert abs(1200 * np.log2(np.median(p[p > 0]) / f0)) < 1.0


def test_yin_silence_is_unvoiced_and_hop_matches_dataset():
    p = yin(np.zeros(SR), SR)
    assert (p == 0).all()
    assert 128 / SR == pytest.approx(HOP_SECONDS)


def test_load_score_reads_notes_rests_and_tonic(tmp_path):
    text = "Sira\tKod\tNota53\tNotaAE\tKoma53\tKomaAE\tPay\tPayda\tMs\tLNS\tBas\tSoz1\tOffset\n"
    text += "1\t51\t\t\t0\t0\t10\t4\t0\t31\t0\tusul\t0\n"  # not a note, ignored
    rows = [(305, 500), (-1, 250)] + [(327, 250)] * 10 + [(305, 1000)]  # ends on the tonic, 305
    for i, (koma, ms) in enumerate(rows):
        text += f"{i + 2}\t9\tx\tx\t{koma}\t{koma}\t1\t4\t{ms}\t99\t96\t\t0\n"
    path = tmp_path / "hicaz--sarki--aksak--test--anon.txt"
    path.write_text(text, encoding="utf-8")
    score = load_score(path, {"hicaz": "Hicaz"})
    assert score.makam == "Hicaz"
    assert score.cents[0] == 0.0 and np.isnan(score.cents[1])  # tonic, then a rest
    assert score.cents[2] == pytest.approx(22 * KOMA_CENTS)  # 22 commas above the tonic
    assert score.seconds[0] == pytest.approx(0.5)


def _score(cents, seconds=0.5):
    cents = np.array(cents, dtype=float)
    return Score("x", "Test", cents, np.full(len(cents), seconds))


def test_pitch_track_has_dataset_hop_and_zero_rests():
    track = score_to_pitch_track(_score([0.0, np.nan, 700.0]), tonic_hz=200.0)
    frames = round(0.5 / HOP_SECONDS)
    assert len(track) == 3 * frames
    assert track[0] == pytest.approx(200.0) and track[frames] == 0.0
    assert track[-1] == pytest.approx(200.0 * 2 ** (700 / 1200), rel=1e-5)


def test_synthesized_score_is_tracked_by_yin():
    score = _score([0.0, 204.0, 385.0, 498.0, 702.0, 0.0])
    audio = synthesize(score, tonic_hz=220.0, max_seconds=None)
    assert len(audio) == pytest.approx(3.0 * SR, abs=10) and np.abs(audio).max() <= 0.81
    p, ref = yin(audio, SR), score_to_pitch_track(score, 220.0)
    n = min(len(p), len(ref))
    both = (p[:n] > 0) & (ref[:n] > 0)
    err = np.abs(1200 * np.log2(p[:n][both] / ref[:n][both]))
    assert np.median(err) < 1.0


def test_recognizer_finds_tonic_and_makam_of_a_transposed_melody():
    rng = np.random.default_rng(0)
    shapes = {"A": ([0, 204, 385, 498, 702], [5, 2, 3, 3, 4]), "B": ([0, 150, 294, 498, 702], [5, 2, 4, 4, 2])}

    def melody(name, tonic):
        cents, weights = shapes[name]
        seq = np.repeat(cents, np.array(weights) * 300) + rng.normal(0, 5, sum(weights) * 300)
        return (tonic * 2 ** (seq / 1200)).astype(np.float32)

    pitches, tonics, labels = [], [], []
    for name in shapes:
        for _ in range(12):
            t = rng.uniform(180, 300)
            pitches.append(melody(name, t))
            tonics.append(t)
            labels.append(name)
    model = MakamRecognizer().fit(pitches, np.array(tonics), np.array(labels))
    pred = model.predict(melody("B", 247.0))
    assert pred.ranking[0][0] == "B"
    off = 1200 * np.log2(pred.tonic_hz / 247.0)
    assert abs(off - 1200 * round(off / 1200)) < 15  # right note, any octave
