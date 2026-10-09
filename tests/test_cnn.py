"""These tests only run where PyTorch is installed (they are skipped otherwise)"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from makam.cnn import MakamCNN, train_and_predict  # noqa: E402


def test_output_shape_does_not_depend_on_length():
    model = MakamCNN(n_inputs=49, n_classes=20).eval()
    for length in (150, 1031):
        out = model(torch.zeros(2, 49, length))
        assert out.shape == (2, 20)


def test_learns_two_easy_classes_on_cpu():
    # class 0 always sits on bin 0, class 1 on bin 28: trivially separable
    rng = np.random.default_rng(0)
    grams, labels = [], []
    for label, b in [(0, 0), (1, 28)]:
        for _ in range(20):
            g = np.zeros((48, 200), dtype=np.float32)
            g[b] = 1.0
            g += rng.random(g.shape).astype(np.float32) * 0.05
            grams.append(g)
            labels.append(label)
    pred = train_and_predict(
        grams, np.array(labels), grams[:1] + grams[-1:], n_classes=2,
        epochs=5, crop_steps=50, batch_size=8, device=torch.device("cpu"),
    )
    assert list(pred) == [0, 1]


def test_predict_proba_both_modes_give_probabilities():
    from makam.cnn import predict_proba

    model = MakamCNN(n_inputs=49, n_classes=3)
    grams = [np.random.default_rng(i).random((48, n)).astype(np.float32) for i, n in enumerate((40, 400))]
    for mode in ("full", "crops"):
        p = predict_proba(model, grams, mode=mode, crop_steps=150, device=torch.device("cpu"))
        assert p.shape == (2, 3)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
