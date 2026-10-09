"""M1 — explore the dataset: stats, a melody contour, pitch-class histograms.

Usage:
    python scripts/m1_explore.py --data ../otmm_makam_recognition_dataset

Outputs go to figures/ and the stats are printed.
The first run parses 1,000 text files (a few minutes); later runs use the cache.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, no window
import matplotlib.pyplot as plt
import numpy as np

from makam.data import HOP_SECONDS, load_pitch, load_recordings
from makam.features import bin_centers, hz_to_cents, pitch_class_histogram

BIN_WIDTH = 7.5


def print_stats(recordings, pitches) -> None:
    print(f"Recordings: {len(recordings)}")
    per_makam = Counter(r.makam for r in recordings)
    print(f"Makams: {len(per_makam)} | recordings per makam: "
          f"min {min(per_makam.values())}, max {max(per_makam.values())}")

    durations_min = np.array([len(p) * HOP_SECONDS / 60 for p in pitches])
    voiced = np.array([(p > 0).mean() for p in pitches])
    print(f"Total audio: {durations_min.sum() / 60:.1f} h | "
          f"duration median {np.median(durations_min):.1f} min "
          f"(min {durations_min.min():.1f}, max {durations_min.max():.1f})")
    print(f"Voiced fraction: median {np.median(voiced):.0%}")

    # --- Leakage check: does the same composition / performer appear several times?
    work_counts = Counter(w for r in recordings for w in r.works)
    artist_counts = Counter(a for r in recordings for a in r.artists)
    repeated_works = {w: c for w, c in work_counts.items() if c > 1}
    recs_sharing_a_work = sum(any(w in repeated_works for w in r.works) for r in recordings)
    print(f"Distinct works: {len(work_counts)} | works recorded more than once: "
          f"{len(repeated_works)} (covering {recs_sharing_a_work} recordings)")
    print(f"Distinct artists: {len(artist_counts)} | top artists by #recordings: "
          f"{[c for _, c in artist_counts.most_common(5)]}")
    print(f"Recordings without work metadata: {sum(not r.works for r in recordings)}")

    instr = Counter(r.instrumentation or "unknown" for r in recordings)
    print("Instrumentation:", dict(instr.most_common()))


def plot_contour(rec, pitch, out: Path, start_s: float = 30, length_s: float = 20) -> None:
    """Melody of one recording in cents vs. time, with piano semitones as dotted lines."""
    i0, i1 = int(start_s / HOP_SECONDS), int((start_s + length_s) / HOP_SECONDS)
    segment = pitch[i0:i1]
    t = start_s + np.arange(len(segment)) * HOP_SECONDS
    cents = np.full(len(segment), np.nan)  # NaN = gap in the line where unvoiced
    voiced = segment > 0
    cents[voiced] = 1200 * np.log2(segment[voiced] / rec.tonic_hz)

    fig, ax = plt.subplots(figsize=(11, 4))
    for c in range(-1200, 2401, 100):
        ax.axhline(c, color="0.85", lw=0.6, ls=":", zorder=0)
    ax.axhline(0, color="C3", lw=1, label="tonic")
    ax.plot(t, cents, lw=0.8, color="C0")
    ax.set(xlabel="time (s)", ylabel="cents above tonic",
           title=f"{rec.makam}: {length_s:.0f} s of melody (dotted = piano semitones)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def mean_histograms(recordings, pitches) -> dict[str, np.ndarray]:
    """Average pitch-class histogram per makam."""
    per_makam: dict[str, list[np.ndarray]] = {}
    for rec, p in zip(recordings, pitches):
        h = pitch_class_histogram(hz_to_cents(p, rec.tonic_hz), bin_width=BIN_WIDTH)
        per_makam.setdefault(rec.makam, []).append(h)
    return {m: np.mean(hs, axis=0) for m, hs in sorted(per_makam.items())}


def plot_makam_comparison(means: dict[str, np.ndarray], makams: list[str], out: Path) -> None:
    x = bin_centers(BIN_WIDTH)
    fig, axes = plt.subplots(len(makams), 1, figsize=(11, 2.2 * len(makams)), sharex=True)
    for ax, m in zip(axes, makams):
        for c in range(0, 1201, 100):
            ax.axvline(c, color="0.85", lw=0.6, ls=":", zorder=0)
        ax.fill_between(x, means[m], color="C0", alpha=0.35)
        ax.plot(x, means[m], color="C0", lw=1)
        ax.set_ylabel(m, rotation=0, ha="right", va="center")
        ax.set_yticks([])
    axes[-1].set_xlabel("cents above tonic, folded into one octave (dotted = piano semitones)")
    axes[-1].set_xticks(range(0, 1201, 100))
    fig.suptitle("Average pitch-class histogram (where the melody spends its time)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_all_makams_heatmap(means: dict[str, np.ndarray], out: Path) -> None:
    names = list(means)
    mat = np.stack([means[m] / means[m].max() for m in names])  # each row scaled to max 1
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.imshow(mat, aspect="auto", cmap="magma", extent=(0, 1200, len(names), 0))
    ax.set_yticks(np.arange(len(names)) + 0.5, names)
    ax.set_xticks(range(0, 1201, 100))
    ax.set_xlabel("cents above tonic (folded)")
    ax.set_title("All 20 makams: average pitch-class histogram (brighter = more time)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to otmm_makam_recognition_dataset")
    parser.add_argument("--cache", default="cache", help="folder for fast .npy copies")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    recordings = load_recordings(args.data)
    pitches = [load_pitch(r, cache_dir=args.cache) for r in recordings]

    print_stats(recordings, pitches)

    example = next(i for i, r in enumerate(recordings) if r.makam == "Rast")
    plot_contour(recordings[example], pitches[example], out / "m1_contour_rast.png")

    means = mean_histograms(recordings, pitches)
    plot_makam_comparison(means, ["Rast", "Mahur", "Ussak", "Hicaz"], out / "m1_four_makams.png")
    plot_all_makams_heatmap(means, out / "m1_all_makams.png")
    print(f"Figures written to {out}/")


if __name__ == "__main__":
    main()
