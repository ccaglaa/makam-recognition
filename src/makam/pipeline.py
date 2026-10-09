"""A complete recognizer: melody in, tonic and ranked makams out, no tonic given

Two stages, both trained on recordings with their annotated tonic
  1. tonic (M3): rotate the reference-relative histogram against every makam
     template and keep the best rotation
  2. makam (M4a F2): per-third histograms measured from that tonic, scored by
     logistic regression, which gives a probability for every makam

The CNN is left out on purpose: it needs PyTorch and adds ar. 1.7 points
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from makam.classify import fit_templates, logreg_proba
from makam.features import hz_to_cents, pitch_class_histogram, section_histograms
from makam.tonic import REFERENCE_HZ, estimate_makam_and_tonic

TEMPLATE_BIN = 7.5
METRIC = "bhattacharyya"
folded_15 = partial(pitch_class_histogram, bin_width=15.0, smoothing_cents=15.0)


def order_features(pitch_hz: np.ndarray, tonic_hz: float) -> np.ndarray:
    """F2: one folded histogram per third of the piece, measured from the tonic"""
    return section_histograms(hz_to_cents(pitch_hz, tonic_hz), 3, folded_15)


@dataclass
class Prediction:
    tonic_hz: float                 # estimated tonic, octave chosen near the melody's lower range
    ranking: list[tuple[str, float]]  # (makam, probability), best first
    template_makam: str             # what stage 1 alone would answer


class MakamRecognizer:
    def fit(self, pitches: list[np.ndarray], tonics_hz: np.ndarray, labels: np.ndarray, seed: int = 0):
        rel = np.stack([pitch_class_histogram(hz_to_cents(p, t), TEMPLATE_BIN) for p, t in zip(pitches, tonics_hz)])
        self.template_labels, self.templates = fit_templates(rel, labels)
        self.train_x = np.stack([order_features(p, t) for p, t in zip(pitches, tonics_hz)])
        self.train_y = np.asarray(labels)
        self.seed = seed
        return self

    def estimate_tonic(self, pitch_hz: np.ndarray) -> tuple[float, str]:
        """Stage 1: (tonic in Hz, makam of the best-matching template)"""
        absolute = pitch_class_histogram(hz_to_cents(pitch_hz, REFERENCE_HZ), TEMPLATE_BIN)
        template_makam, tonic_cents = estimate_makam_and_tonic(
            absolute, self.template_labels, self.templates, METRIC, TEMPLATE_BIN
        )
        return self._place_octave(REFERENCE_HZ * 2 ** (tonic_cents / 1200), pitch_hz), template_makam

    def predict(self, pitch_hz: np.ndarray, tonic_hz: float | None = None) -> Prediction:
        """tonic_hz=None: estimate it (the realistic case). Otherwise use the given tonic"""
        estimated, template_makam = self.estimate_tonic(pitch_hz)
        return self.predict_many([pitch_hz], [tonic_hz or estimated], [template_makam])[0]

    def predict_many(self, pitches, tonics_hz, template_makams=None) -> list[Prediction]:
        """Stage 2 for many melodies at once

        The logistic regression is fitted inside this call (a few seconds), so
        predicting a whole test set in one call fits it only once
        """
        x = np.stack([order_features(p, t) for p, t in zip(pitches, tonics_hz)])
        return self.rank_features(x, tonics_hz, template_makams)

    def rank_features(self, x: np.ndarray, tonics_hz, template_makams=None) -> list[Prediction]:
        """Stage 2 from precomputed order_features rows (saves memory on large test sets)"""
        classes, proba = logreg_proba(self.train_x, self.train_y, x, seed=self.seed)
        template_makams = template_makams or [""] * len(x)
        out = []
        for row, tonic, tm in zip(proba, tonics_hz, template_makams):
            order = np.argsort(-row)
            out.append(Prediction(float(tonic), [(str(classes[i]), float(row[i])) for i in order], tm))
        return out

    @staticmethod
    def _place_octave(tonic_hz: float, pitch_hz: np.ndarray) -> float:
        """Pick the octave so the tonic sits near the bottom of the melody (only for display)"""
        voiced = pitch_hz[pitch_hz > 0]
        if len(voiced) == 0:
            return tonic_hz
        low = np.percentile(voiced, 10)
        octaves = np.round(np.log2(low / tonic_hz))
        return float(tonic_hz * 2**octaves)
