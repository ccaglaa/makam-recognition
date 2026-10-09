from pathlib import Path

import numpy as np

from makam.classify import distance_matrix, fit_templates, knn_predict, logreg_predict, template_predict
from makam.data import Recording
from makam.evaluate import composition_groups, cross_validate, per_class_accuracy

P = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])


def test_identical_histograms_have_zero_distance_for_every_metric():
    for metric in ("l1", "l2", "bhattacharyya"):
        d = distance_matrix(P, P, metric)
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-9)
        assert d.shape == (2, 2)


def test_l1_by_hand():
    # |0.5-0| + |0.5-0.5| + |0-0.5| = 1.0
    assert distance_matrix(P[:1], P[1:], "l1")[0, 0] == 1.0


def test_bhattacharyya_disjoint_histograms_is_large_but_finite():
    d = distance_matrix(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), "bhattacharyya")
    assert np.isfinite(d).all() and d[0, 0] > 20


def test_knn_majority_vote_and_tie_break():
    labels = np.array(["A", "B", "B", "A"])
    dist = np.array([[0.1, 0.2, 0.3, 0.4]])  # closest to farthest: A, B, B, A
    assert knn_predict(dist, labels, k=3)[0] == "B"  # B wins 2 to 1
    assert knn_predict(dist, labels, k=4)[0] == "A"  # 2 to 2, A has the closest member
    assert knn_predict(dist, labels, k=1)[0] == "A"


def test_template_is_the_class_mean_and_predicts_closest():
    hists = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    labels = np.array(["A", "A", "B"])
    names, templates = fit_templates(hists, labels)
    np.testing.assert_allclose(templates[list(names).index("A")], [0.9, 0.1])
    assert template_predict(np.array([[0.7, 0.3]]), names, templates, "l1")[0] == "A"


def _rec(mbid, works):
    return Recording(mbid=mbid, makam="X", tonic_hz=100.0, pitch_path=Path("."), works=works)


def test_groups_merge_recordings_that_share_a_work():
    recs = [_rec("a", ("w1",)), _rec("b", ("w2",)), _rec("c", ("w1", "w3")), _rec("d", ("w3",)), _rec("e", ())]
    g = composition_groups(recs)
    assert g[0] == g[2] == g[3]  # a-c share w1, c-d share w3 -> one group
    assert len({g[0], g[1], g[4]}) == 3  # b and e (no works) are on their own


def test_grouped_cv_never_splits_a_group():
    rng = np.random.default_rng(0)
    labels = np.repeat(["A", "B"], 40)
    hists = rng.random((80, 4))
    groups = np.arange(80) // 4  # groups of 4 recordings with the same label
    tested = []

    def row_index(row):
        return int(np.flatnonzero((hists == row).all(axis=1))[0])

    def spy(train_h, train_y, test_h):
        # a fake predictor that checks the split instead of predicting anything
        train_groups = {groups[row_index(r)] for r in train_h}
        test_groups = {groups[row_index(r)] for r in test_h}
        assert not train_groups & test_groups  # no group on both sides
        tested.extend(row_index(r) for r in test_h)
        return np.full(len(test_h), "A")

    acc, _ = cross_validate(hists, labels, spy, groups=groups, n_splits=5)
    assert len(acc) == 5
    assert sorted(tested) == list(range(80))  # every recording tested exactly once


def test_per_class_accuracy_by_hand():
    labels = np.array(["A", "A", "A", "B", "B"])
    preds = np.array(["A", "B", "A", "B", "B"])
    acc = per_class_accuracy(labels, preds)
    assert acc == {"A": 2 / 3, "B": 1.0}


def test_logreg_learns_two_easy_classes():
    rng = np.random.default_rng(0)
    a = rng.dirichlet([20, 1, 1], size=30)  # histograms peaked on bin 0
    b = rng.dirichlet([1, 1, 20], size=30)  # peaked on bin 2
    x = np.vstack([a, b])
    y = np.array(["A"] * 30 + ["B"] * 30)
    pred = logreg_predict(x, y, np.array([[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]]))
    assert list(pred) == ["A", "B"]
