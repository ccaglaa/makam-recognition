"""M4a: does the histogram lose something important? Register vs order

Usage:
    python scripts/m4a_order_register.py --data ../otmm_makam_recognition_dataset

Four inputs, same classifier (logistic regression), same grouped 10 folds as M2
  F0 folded pitch-class histogram         no order, no register (the M2 input)
  F1 3-octave histogram                   no order, register
  F2 folded histogram per third of piece  order, no register
  F3 3-octave histogram per third         order and register
Tonic known (as in M2). Takes ar. 5 to 10 minutes
"""

from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path

import numpy as np

from makam.classify import logreg_predict
from makam.data import load_pitch, load_recordings
from makam.evaluate import composition_groups, cross_validate, per_class_accuracy
from makam.features import (
    hz_to_cents,
    normalize_octave,
    pitch_class_histogram,
    register_histogram,
    section_histograms,
)

N_SECTIONS = 3
HARD_MAKAMS = ("Ussak", "Muhayyer", "Beyati", "Rast", "Mahur", "Nihavent", "Sultaniyegah")


def build_feature_sets(cents_per_recording):
    folded_15 = partial(pitch_class_histogram, bin_width=15.0, smoothing_cents=15.0)
    feature_fns = {
        "F0 folded (no order, no register)": lambda c: pitch_class_histogram(c, bin_width=7.5),
        "F1 3 octaves (register)": register_histogram,
        "F2 folded per third (order)": lambda c: section_histograms(c, N_SECTIONS, folded_15),
        "F3 3 octaves per third (both)": lambda c: section_histograms(c, N_SECTIONS, register_histogram),
    }
    return {name: np.stack([fn(c) for c in cents_per_recording]) for name, fn in feature_fns.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    Path("results").mkdir(exist_ok=True)

    recordings = load_recordings(args.data)
    cents = [normalize_octave(hz_to_cents(load_pitch(r, cache_dir=args.cache), r.tonic_hz)) for r in recordings]
    labels = np.array([r.makam for r in recordings])
    groups = composition_groups(recordings)
    features = build_feature_sets(cents)

    predictor = partial(logreg_predict, seed=args.seed)
    print("Logistic regression, grouped 10-fold CV (M2 template reference: 73.0%)\n")
    rows, per_makam = [], {}
    for name, x in features.items():
        acc, pred = cross_validate(x, labels, predictor, groups=groups, seed=args.seed)
        rows.append((name, x.shape[1], acc.mean(), acc.std()))
        per_makam[name] = per_class_accuracy(labels, pred)
        print(f"  {name:<38} {x.shape[1]:>4} features  {acc.mean():6.1%} +-{acc.std():.1%}", flush=True)

    with open("results/m4a_accuracy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["features", "n_features", "grouped_mean", "grouped_std"])
        for name, n, mean, std in rows:
            writer.writerow([name, n, f"{mean:.4f}", f"{std:.4f}"])

    names = list(features)
    print("\nAccuracy on the makams the histogram confuses most")
    print(f"  {'makam':<14}" + "".join(f"{n[:2]:>7}" for n in names))
    for makam in HARD_MAKAMS:
        print(f"  {makam:<14}" + "".join(f"{per_makam[n][makam]:>7.0%}" for n in names))


if __name__ == "__main__":
    main()
