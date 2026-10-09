"""The two summary charts shown at the top of the README

Usage:
    python scripts/make_readme_figures.py

Numbers are copied from the experiment outputs (grouped 10-fold CV, tonic known):
  results/m2_accuracy.csv, results/m4a_accuracy.csv, M4a section experiments,
  results/m4b_*_ensemble.csv (CNN + F2, mean of seeds 0 to 2)
Rerun the experiments and update the numbers below if anything changes
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#86b6ef"       # our models
BLUE_KEY = "#1c5cab"   # the bar the chart is about
GRAY = "#b8b7b1"       # published reference


def bar_chart(rows, title, subtitle, out: Path, xmax: float = 100.0) -> None:
    """rows: (label, value, color) from top to bottom"""
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(rows) + 1.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = list(range(len(rows)))[::-1]
    for yi, (label, value, color) in zip(y, rows):
        ax.barh(yi, value, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
        ax.text(value + 1, yi, f"{value:.1f}%", va="center", ha="left", color=INK, fontsize=11)
    ax.set_yticks(y, [r[0] for r in rows], color=INK)
    ax.set_xlim(0, xmax)
    ax.set_xticks(range(0, int(xmax) + 1, 20), [f"{v}%" for v in range(0, int(xmax) + 1, 20)], color=INK_2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    fig.tight_layout(rect=(0, 0, 1, 0.86))  # leave room above the plot for the two title lines
    fig.text(0.02, 0.965, title, ha="left", va="top", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.02, 0.905, subtitle, ha="left", va="top", fontsize=10.5, color=INK_2)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    out = Path("figures")
    out.mkdir(exist_ok=True)
    bar_chart(
        [
            ("Published baseline (MORTY, 2016)", 71.8, GRAY),
            ("Pitch histogram, nearest template", 73.0, BLUE),
            ("1D CNN on the melody", 71.9, BLUE),
            ("Histogram per third of the piece", 77.4, BLUE),
            ("CNN + per-third histograms", 79.0, BLUE_KEY),
        ],
        "Makam recognition accuracy, 20 makams",
        "1000 recordings, 10-fold CV grouped by composition, tonic known, chance = 5%",
        out / "readme_results.png",
    )
    bar_chart(
        [
            ("Whole piece, one histogram", 72.6, BLUE),
            ("Last third only", 61.4, BLUE),
            ("First third only", 77.4, BLUE_KEY),
            ("All three thirds, in order", 77.4, BLUE),
        ],
        "The opening identifies the makam, the ending does not",
        "Same classifier on different parts of each piece (every makam ends on its tonic)",
        out / "readme_opening.png",
    )
    print(f"Saved {out}/readme_results.png and {out}/readme_opening.png")


if __name__ == "__main__":
    main()
