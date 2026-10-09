"""Finding the tonic when nobody gives it to us

Without a tonic we measure every note relative to a fixed, arbitrary
REFERENCE_HZ instead. The histogram then has the right SHAPE but sits at an
unknown rotation around the octave circle

Idea: rotate it to every possible position and compare each rotation with
the makam templates (which were built relative to the true tonic). The best
match tells us both
  * which makam (which template fits best)
  * where the tonic is (how far we had to rotate)
"""

from __future__ import annotations

import numpy as np

from makam.classify import distance_matrix
from makam.features import OCTAVE_CENTS

REFERENCE_HZ = 440.0  # arbitrary, any value works since we try every rotation


def all_rotations(hist: np.ndarray) -> np.ndarray:
    """Matrix whose row s is the histogram rotated left by s bins, shape (n, n)

    Row s answers: "what would this histogram look like if the tonic were
    s bins above the reference?"
    """
    return np.stack([np.roll(hist, -s) for s in range(len(hist))])


def rotation_distances(hist: np.ndarray, templates: np.ndarray, metric: str) -> np.ndarray:
    """Distance of every rotation to every template, shape (n_rotations, n_templates)"""
    return distance_matrix(all_rotations(hist), templates, metric)


def estimate_tonic(hist: np.ndarray, template: np.ndarray, metric: str, bin_width: float) -> float:
    """Makam known: best rotation against that one template -> tonic in cents above reference"""
    dist = rotation_distances(hist, template[np.newaxis, :], metric)[:, 0]
    return float(np.argmin(dist) * bin_width)


def estimate_makam_and_tonic(
    hist: np.ndarray,
    template_labels: np.ndarray,
    templates: np.ndarray,
    metric: str,
    bin_width: float,
) -> tuple[str, float]:
    """Makam unknown too: best (rotation, template) pair over the whole table"""
    dist = rotation_distances(hist, templates, metric)
    rotation, template_index = np.unravel_index(np.argmin(dist), dist.shape)
    return str(template_labels[template_index]), float(rotation * bin_width)


def tonic_cents_above_reference(tonic_hz: float) -> float:
    """Where the true tonic sits on the octave circle, in [0, 1200)"""
    return float(np.mod(OCTAVE_CENTS * np.log2(tonic_hz / REFERENCE_HZ), OCTAVE_CENTS))


def tonic_error_cents(estimated: float, true: float) -> float:
    """Signed error on the octave circle, in [-600, 600)

    An estimate of 1195 for a true value of 5 is 10 cents off, not 1190
    Octave errors disappear on purpose: we only judge the pitch class
    """
    return float(np.mod(estimated - true + 600.0, OCTAVE_CENTS) - 600.0)
