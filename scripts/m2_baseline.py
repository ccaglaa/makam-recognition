"""M2: baseline makam classifier from pitch-class histograms (tonic known)

Usage:
    python scripts/m2_baseline.py --data ../otmm_makam_recognition_dataset

Prints an accuracy table (every method and distance, both CV settings),
saves it to results/m2_accuracy.csv and draws a confusion matrix in figures/
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from makam.classify import METRICS, distance_matrix, fit_templates, knn_predict, template_predict
from makam.data import load_pitch, load_recordings
from makam.evaluate import composition_groups, cross_validate
from makam.features import hz_to_cents, pitch_class_histogram

BIN_WIDTH = 7.5
K_VALUES = (1, 3, 5, 10, 15)


def knn(train_h, train_y, test_h, *, k: int, metric: str):
    return knn_predict(distance_matrix(test_h, train_h, metric), train_y, k)


def template(train_h, train_y, test_h, *, metric: str):
    labels, templates = fit_templates(train_h, train_y)
    return template_predict(test_h, labels, templates, metric)


def all_configs():
    """Every (name, predictor) pair we want to compare"""
    for metric in METRICS:
        yield f"template | {metric}", partial(template, metric=metric)
        for k in K_VALUES:
            yield f"kNN k={k:<2} | {metric}", partial(knn, k=k, metric=metric)


def plot_confusion(labels, predictions, out: Path, title: str) -> None:
    names = sorted(set(labels))
    index = {m: i for i, m in enumerate(names)}
    mat = np.zeros((len(names), len(names)), dtype=int)
    for true, pred in zip(labels, predictions):
        mat[index[true], index[pred]] += 1

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(mat, cmap="Blues")
    for i in range(len(names)):
        for j in range(len(names)):
            if mat[i, j]:
                color = "white" if mat[i, j] > 25 else "black"
                ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(range(len(names)), names, rotation=90)
    ax.set_yticks(range(len(names)), names)
    ax.set(xlabel="predicted makam", ylabel="true makam", title=title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    Path("figures").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    recordings = load_recordings(args.data)
    hists = np.stack([
        pitch_class_histogram(hz_to_cents(load_pitch(r, cache_dir=args.cache), r.tonic_hz), BIN_WIDTH)
        for r in recordings
    ])
    labels = np.array([r.makam for r in recordings])
    groups = composition_groups(recordings)
    print(f"{len(labels)} recordings, {hists.shape[1]} histogram bins, "
          f"{len(np.unique(groups))} composition groups\n")

    rows = []
    print(f"{'method':<28}{'stratified':>16}{'grouped':>16}")
    for name, predictor in all_configs():
        acc_s, _ = cross_validate(hists, labels, predictor, groups=None, seed=args.seed)
        acc_g, pred_g = cross_validate(hists, labels, predictor, groups=groups, seed=args.seed)
        rows.append((name, acc_s.mean(), acc_s.std(), acc_g.mean(), acc_g.std(), pred_g))
        print(f"{name:<28}{acc_s.mean():>9.1%} +-{acc_s.std():.1%}"
              f"{acc_g.mean():>9.1%} +-{acc_g.std():.1%}")

    with open("results/m2_accuracy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "stratified_mean", "stratified_std", "grouped_mean", "grouped_std"])
        for name, ms, ss, mg, sg, _ in rows:
            writer.writerow([" ".join(name.split()), f"{ms:.4f}", f"{ss:.4f}", f"{mg:.4f}", f"{sg:.4f}"])

    # Best config by grouped accuracy (picking the best of many is a little optimistic, see M2 notes)
    best = max(rows, key=lambda r: r[3])
    name, _, _, acc_g, _, pred_g = best
    print(f"\nBest under grouped CV: {name.strip()} -> {acc_g:.1%}")
    plot_confusion(labels, pred_g, Path("figures/m2_confusion.png"),
                   f"Confusion matrix, {name.strip()}, grouped 10-fold CV ({acc_g:.1%})")

    confusions = Counter(
        tuple(sorted((t, p))) for t, p in zip(labels, pred_g) if t != p
    )
    print("Most confused pairs (both directions counted):")
    for (a, b), n in confusions.most_common(8):
        print(f"  {a:<16} <-> {b:<16} {n}")


if __name__ == "__main__":
    main()
