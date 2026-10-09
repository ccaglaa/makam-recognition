"""Reading SymbTr scores and turning them into a pitch track or into sound

SymbTr (MTG, CC BY-NC-SA 4.0) stores each score as a tab-separated text file,
one row per event. The columns we use:
  Kod     event type, 9 = a note (other codes are lyrics, tempo, usul...)
  Koma53  pitch on a grid of 53 commas per octave (Turkish theory), <= 0 = rest
  Ms      duration in milliseconds
The file name starts with the makam: "hicaz--sarki--...txt"

Makam pieces end on their tonic (the karar), so the LAST note gives the tonic.
Scores are written at a nominal pitch, so we choose the tonic frequency ourselves
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from makam.data import HOP_SECONDS

KOMA_CENTS = 1200.0 / 53  # one Holdrian comma, ar. 22.6 cents
NOTE_CODE = "9"


@dataclass(frozen=True)
class Score:
    name: str                 # file name without .txt
    makam: str                # spelled like the recording dataset, e.g. "Ussak"
    cents: np.ndarray         # pitch of each event relative to the final note, NaN = rest
    seconds: np.ndarray       # duration of each event
    work: str = ""            # MusicBrainz work id, for leakage checks


def load_score(path: str | Path, makam_names: dict[str, str]) -> Score | None:
    """Parse one SymbTr .txt file. Returns None if it has no usable notes

    makam_names maps the lowercase file prefix to our spelling, e.g. {"ussak": "Ussak"}
    """
    path = Path(path)
    prefix = path.name.split("--")[0]
    if prefix not in makam_names:
        return None
    komas, durations = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("Kod") != NOTE_CODE:
                continue
            try:
                koma, ms = int(row["Koma53"]), float(row["Ms"])
            except (TypeError, ValueError):
                continue
            if ms <= 0:
                continue
            komas.append(koma if koma > 0 else np.nan)  # rests become NaN
            durations.append(ms / 1000.0)
    komas_arr = np.array(komas, dtype=float)
    sounding = ~np.isnan(komas_arr)
    if sounding.sum() < 10:
        return None
    tonic_koma = komas_arr[sounding][-1]  # the last sounding note is the karar
    return Score(
        name=path.stem,
        makam=makam_names[prefix],
        cents=(komas_arr - tonic_koma) * KOMA_CENTS,
        seconds=np.array(durations),
    )


def load_scores(symbtr_dir: str | Path, makams: list[str]) -> list[Score]:
    """All scores of the given makams, with their MusicBrainz work ids attached"""
    symbtr_dir = Path(symbtr_dir)
    makam_names = {m.lower(): m for m in makams}
    works = {}
    mbid_file = symbtr_dir / "symbTr_mbid.json"
    if mbid_file.exists():
        for entry in json.loads(mbid_file.read_text(encoding="utf-8")):
            works[entry["name"]] = entry["uuid"].rstrip("/").split("/")[-1]
    scores = []
    for path in sorted((symbtr_dir / "txt").glob("*.txt")):
        score = load_score(path, makam_names)
        if score is not None:
            scores.append(Score(score.name, score.makam, score.cents, score.seconds, works.get(score.name, "")))
    return scores


def score_to_pitch_track(score: Score, tonic_hz: float, max_seconds: float | None = None) -> np.ndarray:
    """The score as a pitch track in the SAME format as the recording dataset

    One frequency (Hz) every HOP_SECONDS, 0 for rests: what a perfect melody
    extractor would output if a musician played the score exactly
    """
    hz = np.where(np.isnan(score.cents), 0.0, tonic_hz * 2 ** (np.nan_to_num(score.cents) / 1200))
    frames = np.maximum(1, np.round(score.seconds / HOP_SECONDS).astype(int))
    track = np.repeat(hz, frames).astype(np.float32)
    if max_seconds is not None:
        track = track[: int(max_seconds / HOP_SECONDS)]
    return track


def synthesize(
    score: Score, tonic_hz: float, sample_rate: int = 44100, max_seconds: float | None = 60.0
) -> np.ndarray:
    """Render the score as audio: a soft tone with 4 harmonics per note, float32 in [-1, 1]

    Each note fades in and out over 10 ms so note changes do not click
    """
    harmonics = np.array([1.0, 0.5, 0.3, 0.2])
    pieces, total = [], 0.0
    fade = int(0.01 * sample_rate)
    for cents, seconds in zip(score.cents, score.seconds):
        if max_seconds is not None and total >= max_seconds:
            break
        n = int(round(seconds * sample_rate))
        total += seconds
        if np.isnan(cents) or n == 0:
            pieces.append(np.zeros(n, dtype=np.float32))
            continue
        f0 = tonic_hz * 2 ** (cents / 1200)
        t = np.arange(n) / sample_rate
        tone = sum(a * np.sin(2 * np.pi * f0 * (k + 1) * t) for k, a in enumerate(harmonics))
        envelope = np.ones(n)
        ramp = min(fade, n // 2)
        if ramp > 0:
            envelope[:ramp] = np.linspace(0, 1, ramp)
            envelope[-ramp:] = np.linspace(1, 0, ramp)
        pieces.append((tone * envelope).astype(np.float32))
    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    if max_seconds is not None:
        audio = audio[: int(max_seconds * sample_rate)]
    peak = np.abs(audio).max() if len(audio) else 0.0
    return audio / peak * 0.8 if peak > 0 else audio
