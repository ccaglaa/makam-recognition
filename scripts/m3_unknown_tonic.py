"""M3: makam recognition WITHOUT the tonic, by rotating histograms

Usage:
    python scripts/m3_unknown_tonic.py --data ../otmm_makam_recognition_dataset

Three tasks, all with nearest template + Bhattacharyya (the best M2 method)
and the same 10 folds as M2
  A. tonic identification, makam known
  B. makam recognition, tonic unknown
  C. joint: makam AND tonic both right

The templates are still built from the training recordings with their
annotated tonic (that is training data). Only the TEST recordings lose it
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from makam.classify import fit_templates
from makam.data import load_pitch, load_recordings
from makam.evaluate import composition_groups, cv_splits
from makam.features import hz_to_cents, pitch_class_histogram
from makam.tonic import (
    REFERENCE_HZ,
    estimate_makam_and_tonic,
    estimate_tonic,
    tonic_cents_above_reference,
    tonic_error_cents,
)

METRIC = "bhattacharyya"
TONIC_TOLERANCE = 25.0  # cents, ar. one comma: closer than this counts as the right tonic


def run(rel_hists, abs_hists, labels, true_tonic, groups, bin_width, seed):
    """Out-of-fold predictions for the three tasks"""
    n = len(labels)
    tonic_given_makam = np.empty(n)  # task A
    makam_pred = np.empty(n, dtype=labels.dtype)  # task B
    tonic_joint = np.empty(n)  # task B/C
    for train_idx, test_idx in cv_splits(labels, groups, seed=seed):
        names, templates = fit_templates(rel_hists[train_idx], labels[train_idx])
        for i in test_idx:
            true_template = templates[list(names).index(labels[i])]
            tonic_given_makam[i] = estimate_tonic(abs_hists[i], true_template, METRIC, bin_width)
            makam_pred[i], tonic_joint[i] = estimate_makam_and_tonic(
                abs_hists[i], names, templates, METRIC, bin_width
            )
    err_a = np.array([tonic_error_cents(e, t) for e, t in zip(tonic_given_makam, true_tonic)])
    err_c = np.array([tonic_error_cents(e, t) for e, t in zip(tonic_joint, true_tonic)])
    return err_a, makam_pred, err_c


def plot_tonic_errors(err_a, err_c, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    bins = np.arange(-600, 601, 15)
    for ax, err, title in [
        (axes[0], err_a, "A. tonic estimated, makam known"),
        (axes[1], err_c, "C. tonic estimated, makam unknown"),
    ]:
        ax.hist(err, bins=bins, color="C0")
        ax.set_yscale("log")  # the 0 spike dwarfs everything else otherwise
        # On the folded circle a fifth up lands at -500, so +-500 marks "off by a fourth or a fifth"
        for c in (-500, 500):
            ax.axvline(c, color="C3", lw=0.8, ls="--")
        ax.text(-500, 1, " off by a 4th or 5th", color="C3", fontsize=8, va="bottom")
        ax.set_title(title)
        ax.set_ylabel("recordings (log scale)")
    axes[-1].set_xlabel("tonic error in cents, octave folded (0 = correct)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bin-width", type=float, default=7.5)
    args = parser.parse_args()
    Path("figures").mkdir(exist_ok=True)

    recordings = load_recordings(args.data)
    pitches = [load_pitch(r, cache_dir=args.cache) for r in recordings]
    bw = args.bin_width
    # Tonic-relative histograms: only used to build templates from TRAINING folds
    rel = np.stack([pitch_class_histogram(hz_to_cents(p, r.tonic_hz), bw) for r, p in zip(recordings, pitches)])
    # Reference-relative histograms: what a test recording looks like when the tonic is unknown
    absolute = np.stack([pitch_class_histogram(hz_to_cents(p, REFERENCE_HZ), bw) for p in pitches])
    labels = np.array([r.makam for r in recordings])
    true_tonic = np.array([tonic_cents_above_reference(r.tonic_hz) for r in recordings])
    groups = composition_groups(recordings)

    print(f"Nearest template | {METRIC}, tonic counted correct within {TONIC_TOLERANCE:.0f} cents\n")
    print(f"{'task':<40}{'stratified':>12}{'grouped':>12}")
    results = {}
    for setting, g in [("stratified", None), ("grouped", groups)]:
        err_a, makam_pred, err_c = run(rel, absolute, labels, true_tonic, g, bw, args.seed)
        tonic_ok_a = np.abs(err_a) < TONIC_TOLERANCE
        makam_ok = makam_pred == labels
        tonic_ok_c = np.abs(err_c) < TONIC_TOLERANCE
        results[setting] = {
            "A. tonic, makam known": tonic_ok_a.mean(),
            "B. makam, tonic unknown": makam_ok.mean(),
            "C. makam and tonic both right": (makam_ok & tonic_ok_c).mean(),
        }
        if setting == "grouped":
            kept = (err_a, err_c, makam_pred, makam_ok, tonic_ok_c)
    for task in results["stratified"]:
        print(f"{task:<40}{results['stratified'][task]:>12.1%}{results['grouped'][task]:>12.1%}")
    print(f"{'(M2 reference) makam, tonic GIVEN':<40}{'72.5%':>12}{'73.0%':>12}")

    err_a, err_c, makam_pred, makam_ok, tonic_ok_c = kept
    plot_tonic_errors(err_a, err_c, Path("figures/m3_tonic_errors.png"))

    wrong = np.abs(err_a) >= TONIC_TOLERANCE
    rounded = Counter(int(round(e / 50.0) * 50) for e in err_a[wrong])
    print(f"\nTask A, grouped: {wrong.sum()} wrong tonics. Most common errors (rounded to 50 cents):")
    for cents, count in rounded.most_common(5):
        print(f"  {cents:+5d} cents  x{count}")

    print("\nTask A, grouped: makams whose tonic is missed most often")
    print(f"  {'makam':<16}{'wrong':>6}{'-500':>7}{'+500':>7}{'other':>7}")
    # -500: estimate a fourth below the true tonic (same pitch class as a fifth above)
    # +500: estimate a fourth above the true tonic
    minus_500 = wrong & (np.abs(err_a + 500) < 50)
    plus_500 = wrong & (np.abs(err_a - 500) < 50)
    for makam, count in Counter(labels[wrong]).most_common(8):
        m = labels == makam
        n_minus, n_plus = (minus_500 & m).sum(), (plus_500 & m).sum()
        print(f"  {makam:<16}{count:>6}{n_minus:>7}{n_plus:>7}{count - n_minus - n_plus:>7}")

    print("\nTask B, grouped: what goes wrong when the makam is wrong")
    print(f"  makam wrong but tonic right : {(~makam_ok & tonic_ok_c).sum()}")
    print(f"  makam wrong and tonic wrong : {(~makam_ok & ~tonic_ok_c).sum()}")
    print(f"  makam right but tonic wrong : {(makam_ok & ~tonic_ok_c).sum()}")


if __name__ == "__main__":
    main()
