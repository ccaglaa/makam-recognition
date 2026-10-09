"""M5 demo: hear a makam, then let the model recognize it from the AUDIO alone

Usage:
    python scripts/m5_demo.py --data ../otmm_makam_recognition_dataset --symbtr ../SymbTr --makam Saba
    python scripts/m5_demo.py --data ../otmm_makam_recognition_dataset --symbtr ../SymbTr --random

Steps, exactly like a real recording would go:
  1. pick a SymbTr score the model has never seen (composition not in the recordings)
  2. synthesize the first 60 s at a random tonic -> demo/<name>.wav (open it and listen)
  3. extract the melody from the audio with YIN (no access to the score any more)
  4. find the tonic and rank the makams (trained on all 1000 recordings)
  5. compare with the truth, and plot the extracted melody -> demo/<name>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from makam.data import HOP_SECONDS, load_pitch, load_recordings
from makam.pipeline import MakamRecognizer
from makam.pitch import yin
from makam.symbtr import load_scores, score_to_pitch_track, synthesize
from makam.tonic import tonic_cents_above_reference, tonic_error_cents

SAMPLE_RATE = 44100


def plot(track, tonic_hz, title, out: Path) -> None:
    t = np.arange(len(track)) * HOP_SECONDS
    cents = np.full(len(track), np.nan)
    voiced = track > 0
    cents[voiced] = 1200 * np.log2(track[voiced] / tonic_hz)
    fig, ax = plt.subplots(figsize=(11, 4))
    for c in range(-1200, 2401, 100):
        ax.axhline(c, color="0.88", lw=0.6, ls=":", zorder=0)
    ax.axhline(0, color="C3", lw=1, label="estimated tonic")
    ax.plot(t, cents, lw=0.8, color="C0", label="melody extracted by YIN")
    ax.set(xlabel="time (s)", ylabel="cents above estimated tonic", title=title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--symbtr", required=True)
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--makam", help="pick a random unseen score of this makam")
    parser.add_argument("--score", help="exact SymbTr file name without .txt")
    parser.add_argument("--random", action="store_true", help="any unseen score")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=None, help="fix the choice and the tonic")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path("demo")
    out_dir.mkdir(exist_ok=True)

    print("Training on the 1000 recordings...", flush=True)
    recordings = load_recordings(args.data)
    pitches = [load_pitch(r, cache_dir=args.cache) for r in recordings]
    model = MakamRecognizer().fit(
        pitches, np.array([r.tonic_hz for r in recordings]), np.array([r.makam for r in recordings])
    )
    del pitches

    seen_works = {w for r in recordings for w in r.works}
    candidates = [s for s in load_scores(args.symbtr, sorted({r.makam for r in recordings})) if s.work not in seen_works]
    if args.score:
        candidates = [s for s in candidates if s.name == args.score]
    elif args.makam:
        candidates = [s for s in candidates if s.makam.lower() == args.makam.lower()]
    elif not args.random:
        raise SystemExit("choose --makam NAME, --score NAME or --random")
    if not candidates:
        raise SystemExit("no unseen score matches (the composition may be among the recordings)")
    score = candidates[rng.integers(len(candidates))]

    true_tonic = float(rng.uniform(180.0, 300.0))
    audio = synthesize(score, true_tonic, SAMPLE_RATE, max_seconds=args.seconds)
    wav_path = out_dir / f"{score.name}.wav"
    wavfile.write(wav_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    track = yin(audio, SAMPLE_RATE)  # from here on, only the audio is used
    reference = score_to_pitch_track(score, true_tonic, max_seconds=args.seconds)
    n = min(len(track), len(reference))
    both = (track[:n] > 0) & (reference[:n] > 0)
    pitch_error = np.abs(1200 * np.log2(track[:n][both] / reference[:n][both]))

    pred = model.predict(track)
    tonic_err = tonic_error_cents(tonic_cents_above_reference(pred.tonic_hz), tonic_cents_above_reference(true_tonic))

    print(f"\nScore     : {score.name}")
    print(f"Audio     : {wav_path} ({len(audio) / SAMPLE_RATE:.0f} s, open it to listen)")
    print(f"YIN       : median pitch error {np.median(pitch_error):.2f} cents vs the score")
    print(f"Tonic     : true {true_tonic:.1f} Hz, estimated {pred.tonic_hz:.1f} Hz "
          f"({'right' if abs(tonic_err) < 25 else 'WRONG'}, off by {tonic_err:+.0f} cents, octave ignored)")
    print(f"True makam: {score.makam}")
    print("Ranking   :")
    for i, (makam, p) in enumerate(pred.ranking[:5], 1):
        mark = "  <- true" if makam == score.makam else ""
        print(f"  {i}. {makam:<16} {p:6.1%}{mark}")

    png_path = out_dir / f"{score.name}.png"
    verdict = "right" if pred.ranking[0][0] == score.makam else "wrong"
    plot(track, pred.tonic_hz,
         f"{score.makam} (score), predicted {pred.ranking[0][0]} ({verdict})", png_path)
    print(f"Figure    : {png_path}")


if __name__ == "__main__":
    main()
