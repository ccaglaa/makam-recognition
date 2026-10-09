"""M4b: a 1D CNN that reads the melody in order, same grouped 10 folds as M2 and M4a

Usage:
    python scripts/m4b_cnn.py --data ../otmm_makam_recognition_dataset --quick
    python scripts/m4b_cnn.py --data ../otmm_makam_recognition_dataset
    python scripts/m4b_cnn.py --data ../otmm_makam_recognition_dataset --region 0.333
    python scripts/m4b_cnn.py --data ../otmm_makam_recognition_dataset --region 0.333 --ensemble

Run --quick first: 2 folds, 3 epochs, only checks that nothing crashes
Tonic known, to compare with M4a (F2 order features: 77.4%)

Printed for every fold
  train : accuracy on its own training recordings (same windows as test)
  test  : test recordings as averaged 30 s windows, like in training
  with --ensemble, also
  F2    : the M4a logistic regression on per-third histograms, same fold
  both  : average of the CNN and F2 probabilities
--region 0.333 trains and tests on windows from the first third only (the opening)
First run: 40 epochs whole piece gave train 83.8%, test 71.9% (overfitting,
and averaging all windows dilutes the informative opening)
"""

from __future__ import annotations

import argparse
import csv
import time
from functools import partial
from pathlib import Path

import numpy as np

from makam.classify import logreg_proba
from makam.cnn import pick_device, predict_proba, train_model
from makam.data import load_pitch, load_recordings
from makam.evaluate import composition_groups, cv_splits, per_class_accuracy
from makam.features import hz_to_cents, pitch_class_histogram, section_histograms
from makam.sequence import pitch_class_gram

HARD_MAKAMS = ("Ussak", "Muhayyer", "Beyati", "Rast", "Mahur", "Nihavent", "Sultaniyegah")


def f2_features(pitches, recordings) -> np.ndarray:
    """The M4a F2 input: folded 15-cent histograms of each third of the piece"""
    folded_15 = partial(pitch_class_histogram, bin_width=15.0, smoothing_cents=15.0)
    return np.stack([
        section_histograms(hz_to_cents(p, r.tonic_hz), 3, folded_15)
        for p, r in zip(pitches, recordings)
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--region", type=float, default=1.0,
                        help="fraction of the piece used, from the start (0.333 = opening third)")
    parser.add_argument("--ensemble", action="store_true", help="also run F2 and CNN + F2")
    parser.add_argument("--quick", action="store_true", help="2 folds, 3 epochs: crash test only")
    args = parser.parse_args()
    Path("results").mkdir(exist_ok=True)

    device = pick_device()
    print(f"device: {device} | region: first {args.region:.0%} of each piece")

    recordings = load_recordings(args.data)
    pitches = [load_pitch(r, cache_dir=args.cache) for r in recordings]
    grams = [pitch_class_gram(p, r.tonic_hz) for p, r in zip(pitches, recordings)]
    labels = np.array([r.makam for r in recordings])
    class_names, y = np.unique(labels, return_inverse=True)  # makam names <-> integers 0..19
    groups = composition_groups(recordings)
    f2 = f2_features(pitches, recordings) if args.ensemble else None

    epochs = 3 if args.quick else args.epochs
    n_folds_to_run = 2 if args.quick else 10
    columns = ["train", "test"] + (["F2", "both"] if args.ensemble else [])

    preds = {c: np.full(len(y), -1) for c in columns if c != "train"}
    rows = []
    print("  fold" + "".join(f"{c:>8}" for c in columns))
    for fold, (train_idx, test_idx) in enumerate(cv_splits(labels, groups, seed=args.seed)):
        if fold >= n_folds_to_run:
            break
        t0 = time.time()
        model = train_model(
            [grams[i] for i in train_idx], y[train_idx], n_classes=len(class_names),
            epochs=epochs, seed=args.seed * 100 + fold, device=device, region=args.region,
        )
        predict = partial(predict_proba, model, mode="crops", region=args.region)
        row = {"fold": fold + 1}
        row["train"] = np.mean(predict([grams[i] for i in train_idx]).argmax(axis=1) == y[train_idx])
        cnn_proba = predict([grams[i] for i in test_idx])
        fold_preds = {"test": cnn_proba.argmax(axis=1)}
        if args.ensemble:
            classes, f2_proba = logreg_proba(f2[train_idx], labels[train_idx], f2[test_idx], seed=args.seed)
            assert list(classes) == list(class_names), "class order must match to average"
            fold_preds["F2"] = f2_proba.argmax(axis=1)
            fold_preds["both"] = (cnn_proba + f2_proba).argmax(axis=1)  # equal-weight average
        for name, pred in fold_preds.items():
            preds[name][test_idx] = pred
            row[name] = np.mean(pred == y[test_idx])
        rows.append(row)
        print(f"  {fold + 1:>4}" + "".join(f"{row[c]:8.1%}" for c in columns)
              + f"   ({time.time() - t0:.0f} s)", flush=True)

    mean = {c: np.mean([r[c] for r in rows]) for c in columns}
    std = {c: np.std([r[c] for r in rows]) for c in columns}
    print("\n  mean" + "".join(f"{mean[c]:8.1%}" for c in columns) + f"   over {len(rows)} folds")
    print("  std " + "".join(f"{std[c]:8.1%}" for c in columns))
    if args.quick:
        print("(quick mode: these numbers mean nothing, they only show the code runs)")
        return
    print("References: M2 template 73.0%, M4a F2 77.4%, M4b first run (whole piece) 71.9%")

    print("\nHard makams:")
    shown = [c for c in columns if c != "train"]
    print(f"  {'makam':<14}" + "".join(f"{c:>8}" for c in shown))
    per_makam = {c: per_class_accuracy(labels, class_names[preds[c]]) for c in shown}
    for makam in HARD_MAKAMS:
        print(f"  {makam:<14}" + "".join(f"{per_makam[c][makam]:8.0%}" for c in shown))

    name = f"results/m4b_{epochs}ep_region{args.region:.2f}{'_ensemble' if args.ensemble else ''}.csv"
    with open(name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold"] + columns)
        for r in rows:
            writer.writerow([r["fold"]] + [f"{r[c]:.4f}" for c in columns])
        writer.writerow(["mean"] + [f"{mean[c]:.4f}" for c in columns])
        writer.writerow(["std"] + [f"{std[c]:.4f}" for c in columns])
    print(f"\nSaved {name}")


if __name__ == "__main__":
    main()
