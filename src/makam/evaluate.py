"""Honest evaluation: cross-validation, with and without grouping by composition

Why two settings
  * stratified 10-fold: the standard protocol (the MORTY paper uses it), so
    our numbers are comparable to theirs
  * grouped 10-fold: two recordings of the SAME composition never end up on
    both sides of a split. Otherwise the model can "recognize the song"
    instead of the makam, and the score looks better than it really is
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from makam.data import Recording

# A predictor gets (train_hists, train_labels, test_hists) and returns predicted labels
Predictor = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def composition_groups(recordings: list[Recording]) -> np.ndarray:
    """Group id per recording: recordings sharing a composition get the same id

    A recording can list several works (e.g. a medley), so we merge groups
    that share any work (union-find). Recordings with no work metadata get
    a group of their own
    """
    parent = list(range(len(recordings)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path halving, keeps the trees shallow
            i = parent[i]
        return i

    first_seen: dict[str, int] = {}
    for i, rec in enumerate(recordings):
        for work in rec.works:
            if work in first_seen:
                parent[find(i)] = find(first_seen[work])  # merge the two groups
            else:
                first_seen[work] = i

    roots = [find(i) for i in range(len(recordings))]
    _, group_ids = np.unique(roots, return_inverse=True)  # relabel as 0, 1, 2, ...
    return group_ids


def cv_splits(
    labels: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 10,
    seed: int = 0,
):
    """Yield (train_idx, test_idx) for each fold

    groups=None -> stratified folds, otherwise grouped and stratified folds
    Same labels, groups and seed -> same folds, so different milestones are
    evaluated on exactly the same splits
    """
    dummy_x = np.zeros((len(labels), 1))  # the splitters only look at labels and groups
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from splitter.split(dummy_x, labels)
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from splitter.split(dummy_x, labels, groups)


def cross_validate(
    hists: np.ndarray,
    labels: np.ndarray,
    predictor: Predictor,
    groups: np.ndarray | None = None,
    n_splits: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run k-fold CV and return (accuracy per fold, out-of-fold predictions)

    groups=None -> stratified folds, otherwise grouped and stratified folds
    Every recording is predicted exactly once, by a model that never saw it
    """
    predictions = np.empty(len(labels), dtype=labels.dtype)
    fold_accuracies = []
    for train_idx, test_idx in cv_splits(labels, groups, n_splits, seed):
        pred = predictor(hists[train_idx], labels[train_idx], hists[test_idx])
        predictions[test_idx] = pred
        fold_accuracies.append(np.mean(pred == labels[test_idx]))
    return np.array(fold_accuracies), predictions


def per_class_accuracy(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """For each makam, the fraction of its recordings that were predicted correctly

    This is the diagonal of the confusion matrix divided by the row totals
    """
    return {
        str(makam): float(np.mean(predictions[labels == makam] == makam))
        for makam in np.unique(labels)
    }
