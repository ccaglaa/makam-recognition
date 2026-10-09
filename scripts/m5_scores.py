"""M5: does a model trained on PERFORMANCES recognize SCORES of pieces it never heard?

Usage:
    git clone https://github.com/MTG/SymbTr.git ../SymbTr
    python scripts/m5_scores.py --data ../otmm_makam_recognition_dataset --symbtr ../SymbTr

Train on all 1000 recordings (with their tonics). Test on every SymbTr score of the
same 20 makams whose composition is NOT among the recordings. Each score becomes a
pitch track at a random tonic between 180 and 330 Hz, so the tonic must be found
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from makam.data import load_pitch, load_recordings
from makam.pipeline import MakamRecognizer, order_features
from makam.symbtr import load_scores, score_to_pitch_track
from makam.tonic import tonic_cents_above_reference, tonic_error_cents

TONIC_TOLERANCE = 25.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--symbtr", required=True)
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    Path("results").mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    recordings = load_recordings(args.data)
    pitches = [load_pitch(r, cache_dir=args.cache) for r in recordings]
    model = MakamRecognizer().fit(
        pitches, np.array([r.tonic_hz for r in recordings]), np.array([r.makam for r in recordings]), seed=args.seed
    )
    del pitches

    makams = sorted({r.makam for r in recordings})
    seen_works = {w for r in recordings for w in r.works}
    scores = load_scores(args.symbtr, makams)
    unseen = [s for s in scores if s.work not in seen_works]
    print(f"{len(scores)} scores in these 20 makams, {len(scores) - len(unseen)} removed because the "
          f"composition is among the training recordings, {len(unseen)} left\n")

    labels, x_known, x_est, template_makams, tonic_err = [], [], [], [], []
    for s in unseen:
        tonic = rng.uniform(180.0, 330.0)
        track = score_to_pitch_track(s, tonic)
        est_tonic, template_makam = model.estimate_tonic(track)
        labels.append(s.makam)
        x_known.append(order_features(track, tonic))
        x_est.append(order_features(track, est_tonic))
        template_makams.append(template_makam)
        tonic_err.append(tonic_error_cents(tonic_cents_above_reference(est_tonic), tonic_cents_above_reference(tonic)))
    labels = np.array(labels)
    tonic_ok = np.abs(tonic_err) < TONIC_TOLERANCE

    known = np.array([p.ranking[0][0] for p in model.rank_features(np.stack(x_known), [0] * len(labels))])
    est_preds = model.rank_features(np.stack(x_est), [0] * len(labels))
    est = np.array([p.ranking[0][0] for p in est_preds])
    confidence = np.array([p.ranking[0][1] for p in est_preds])
    template = np.array(template_makams)

    balanced = np.mean([np.mean(est[labels == m] == m) for m in makams])  # every makam counts equally
    rows = [
        ("Tonic found (within 25 cents)", tonic_ok.mean(), "96.1%"),
        ("Makam, tonic given, F2", np.mean(known == labels), "77.4%"),
        ("Makam, tonic unknown, template only", np.mean(template == labels), "66.9%"),
        ("Makam, tonic unknown, template + F2", np.mean(est == labels), "-"),
        ("Makam and tonic both right", np.mean((est == labels) & tonic_ok), "65.9%"),
        ("Balanced accuracy, tonic unknown + F2", balanced, "-"),
    ]
    print(f"{'task':<40}{'scores':>9}{'recordings (CV)':>18}")
    for name, value, ref in rows:
        print(f"{name:<40}{value:>9.1%}{ref:>18}")
    print("Chance level with 20 makams: 5%. Score counts per makam are unequal (7 to 145), so")
    print("balanced accuracy, the mean of the 20 per-makam accuracies, is the fairer summary")

    print("\nPer makam: tonic found, makam with tonic given, makam with tonic unknown")
    print(f"  {'makam':<16}{'scores':>7}{'tonic':>8}{'given':>8}{'unknown':>9}")
    for m in makams:
        mask = labels == m
        print(f"  {m:<16}{mask.sum():>7}{tonic_ok[mask].mean():>8.0%}"
              f"{np.mean(known[mask] == m):>8.0%}{np.mean(est[mask] == m):>9.0%}")

    print("\nMost common mistakes (true -> predicted):")
    for (t, p), n in Counter((t, p) for t, p in zip(labels, est) if t != p).most_common(8):
        print(f"  {t:<16} -> {p:<16} {n}")

    print("\nConfidence on scores (tonic unknown): is the model still right when it is sure?")
    edges = [0.0, 0.4, 0.6, 0.8, 0.95, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.any():
            print(f"  {lo:4.0%} to {min(hi, 1):4.0%} sure: {mask.sum():>4} scores, right {np.mean(est[mask] == labels[mask]):6.1%}")

    with open("results/m5_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "scores", "recordings_cv"])
        for name, value, ref in rows:
            writer.writerow([name, f"{value:.4f}", ref])


if __name__ == "__main__":
    main()
