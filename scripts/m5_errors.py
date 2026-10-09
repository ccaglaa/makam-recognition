"""M5: where does the best simple model (F2, tonic known) go wrong on the recordings?

Usage:
    python scripts/m5_errors.py --data ../otmm_makam_recognition_dataset

Same grouped 10 folds as before. Every recording gets an out-of-fold prediction
and a confidence (the probability of the predicted makam). Accuracy is then
broken down by recording properties, to find WHERE the errors concentrate
"""

from __future__ import annotations

import argparse

import numpy as np

from makam.classify import logreg_proba
from makam.data import HOP_SECONDS, load_pitch, load_recordings
from makam.evaluate import composition_groups, cv_splits
from makam.features import hz_to_cents
from makam.pipeline import order_features


def breakdown(title, groups, correct) -> None:
    print(f"\n{title}")
    for g in sorted(set(groups), key=str):
        mask = np.array([x == g for x in groups])
        print(f"  {str(g):<36}{mask.sum():>5} recordings  {correct[mask].mean():6.1%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    recordings = load_recordings(args.data)
    pitches = [load_pitch(r, cache_dir=args.cache) for r in recordings]
    labels = np.array([r.makam for r in recordings])
    x = np.stack([order_features(p, r.tonic_hz) for p, r in zip(pitches, recordings)])
    groups = composition_groups(recordings)

    predicted = np.empty(len(labels), dtype=labels.dtype)
    confidence = np.zeros(len(labels))
    for train_idx, test_idx in cv_splits(labels, groups, seed=args.seed):
        classes, proba = logreg_proba(x[train_idx], labels[train_idx], x[test_idx], seed=args.seed)
        predicted[test_idx] = classes[proba.argmax(axis=1)]
        confidence[test_idx] = proba.max(axis=1)
    correct = predicted == labels
    print(f"F2, grouped 10-fold CV: {correct.mean():.1%} overall")

    breakdown("By instrumentation", [r.instrumentation or "unknown" for r in recordings], correct)

    minutes = np.array([len(p) * HOP_SECONDS / 60 for p in pitches])
    cuts = np.quantile(minutes, [1 / 3, 2 / 3])
    length = ["short (< %.1f min)" % cuts[0] if m < cuts[0] else
              "long (> %.1f min)" % cuts[1] if m > cuts[1] else "medium" for m in minutes]
    breakdown("By length (thirds of the dataset)", length, correct)

    medians = np.array([np.median(hz_to_cents(p, r.tonic_hz)) for p, r in zip(pitches, recordings)])
    octave_off = ["tonic octave matches melody" if -200 <= m < 1000 else "tonic octave off from melody"
                  for m in medians]
    breakdown("By octave of the annotated tonic (folded features should not care)", octave_off, correct)

    print("\nConfidence: when the model says it is X% sure, how often is it right?")
    edges = [0.0, 0.4, 0.6, 0.8, 0.95, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.any():
            print(f"  {lo:4.0%} to {min(hi, 1):4.0%} sure: {mask.sum():>4} recordings, right {correct[mask].mean():6.1%}")


if __name__ == "__main__":
    main()
