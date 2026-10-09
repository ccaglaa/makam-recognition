"""Classifying recordings from their pitch-class histograms

Two classic methods, both based on "how far apart are two histograms":

  * k-nearest neighbours (kNN): compare the new recording to EVERY training
    recording, take the k closest, and let them vote
  * nearest template: average the training histograms of each makam into one
    template per makam, and pick the closest template

Everything works on distance matrices: D[i, j] = distance between test
recording i and training recording (or template) j
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.spatial.distance import cdist

METRICS = ("l1", "l2", "bhattacharyya")


def distance_matrix(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    """Distances between every row of a (n, d) and every row of b (m, d) -> (n, m)

    Rows must be histograms that sum to 1
      l1: sum of absolute differences (city-block distance)
      l2: usual straight-line (Euclidean) distance
      bhattacharyya: -log(sum sqrt(p * q)), a distance between probability
          distributions, 0 when they are identical
    """
    if metric == "l1":
        return cdist(a, b, metric="cityblock")
    if metric == "l2":
        return cdist(a, b, metric="euclidean")
    if metric == "bhattacharyya":
        # sum_k sqrt(p_k * q_k) for all pairs at once is a matrix product of the square roots
        overlap = np.sqrt(a) @ np.sqrt(b).T
        return -np.log(np.clip(overlap, 1e-12, None))  # clip avoids log(0) for disjoint histograms
    raise ValueError(f"unknown metric {metric!r}, choose from {METRICS}")


def knn_predict(dist: np.ndarray, train_labels: np.ndarray, k: int) -> np.ndarray:
    """Majority vote among the k nearest training recordings, for each test row

    Ties are broken in favour of the label whose member is closest
    """
    nearest = np.argsort(dist, axis=1)[:, :k]  # indices of the k smallest distances per row
    predictions = []
    for row in nearest:
        labels = train_labels[row]  # labels ordered from closest to farthest
        counts = Counter(labels)
        best = max(counts.values())
        # first label in distance order that reaches the best count
        predictions.append(next(lab for lab in labels if counts[lab] == best))
    return np.array(predictions)


def fit_templates(train_hists: np.ndarray, train_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One template per makam: the mean of its training histograms

    Returns (template_labels, templates) with templates of shape (n_makams, d)
    """
    template_labels = np.unique(train_labels)
    templates = np.stack([train_hists[train_labels == m].mean(axis=0) for m in template_labels])
    return template_labels, templates


def template_predict(
    test_hists: np.ndarray, template_labels: np.ndarray, templates: np.ndarray, metric: str
) -> np.ndarray:
    """Label of the closest template for each test histogram"""
    dist = distance_matrix(test_hists, templates, metric)
    return template_labels[np.argmin(dist, axis=1)]
