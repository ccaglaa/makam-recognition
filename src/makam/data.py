"""Loading the OTMM Makam Recognition Dataset

The dataset folder (cloned from GitHub) looks like:

    otmm_makam_recognition_dataset/
        annotations.json          # one entry per recording: mbid, makam, tonic (Hz)
        data/<Makam>/<id>.pitch   # predominant melody: one frequency (Hz) per line
        data/<Makam>/<id>.json    # MusicBrainz metadata (works, artists, form...)

A pitch value of 0.0 means "no melody detected here" (silence, percussion...)
Consecutive values are 0.0029 s apart (HOP_SECONDS)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HOP_SECONDS = 128 / 44100  # ar. 0.0029 s between two pitch samples


@dataclass(frozen=True)
class Recording:
    """Everything we know about one recording, except the (large) pitch track"""

    mbid: str            # MusicBrainz recording id = file name
    makam: str           # the label we want to predict
    tonic_hz: float      # annotated tonic frequency
    pitch_path: Path
    works: tuple[str, ...] = field(default=())    # composition ids 
    artists: tuple[str, ...] = field(default=())  # performer ids 
    instrumentation: str = ""


def load_recordings(dataset_dir: str | Path) -> list[Recording]:
    """Read annotations.json + per-recording metadata. Does NOT load pitch tracks"""
    dataset_dir = Path(dataset_dir)
    with open(dataset_dir / "annotations.json", encoding="utf-8") as f:
        annotations = json.load(f)

    recordings = []
    for ann in annotations:
        mbid = ann["mbid"].rstrip("/").split("/")[-1]  # keep only the id part of the URL
        makam = ann["makam"]
        folder = dataset_dir / "data" / makam
        meta_path = folder / f"{mbid}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        recordings.append(
            Recording(
                mbid=mbid,
                makam=makam,
                tonic_hz=float(ann["tonic"]),
                pitch_path=folder / f"{mbid}.pitch",
                works=tuple(w["mbid"] for w in meta.get("works", [])),
                artists=tuple(a["mbid"] for a in meta.get("artists", [])),
                instrumentation=meta.get("instrumentation_voicing", ""),
            )
        )
    return recordings


def load_pitch(recording: Recording, cache_dir: str | Path | None = None) -> np.ndarray:
    """Return the pitch track in Hz (float32)

    Parsing text is slow, so if cache_dir is given we save a binary .npy copy
    the first time and reuse it afterwards
    """
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{recording.mbid}.npy"
        if cache_path.exists():
            return np.load(cache_path)

    pitch = np.loadtxt(recording.pitch_path, dtype=np.float32)

    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, pitch)
    return pitch
