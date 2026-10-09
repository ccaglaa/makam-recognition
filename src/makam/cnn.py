"""A small 1D convolutional network that reads a pitch-class-gram over time

Input shape (batch, channels, time): 48 pitch bins + 1 position channel
  * each Conv1d slides a window of 5 steps (ar. 1 s) along time and learns
    short melodic patterns, like "step down from the 4th to the 3rd"
  * MaxPool1d halves the time axis, so deeper layers see longer stretches
    (after 3 blocks one unit sees ar. 8 s of melody)
  * global pooling over time (mean and max) turns any length into a fixed
    vector, so the same network can read a 30 s crop or a whole 10 min piece

Training uses random 30 s crops (many different examples per recording)
Two ways to predict a whole recording:
  * "full": the whole recording in one pass (longer than anything seen in training)
  * "crops": overlapping 30 s windows, averaged (exactly what was seen in training)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from makam.sequence import full_sequence, random_crop, sliding_crops


class MakamCNN(nn.Module):
    def __init__(self, n_inputs: int, n_classes: int, width: int = 64, dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        channels = n_inputs
        for _ in range(3):
            layers += [
                nn.Conv1d(channels, width, kernel_size=5, padding=2),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            channels = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * width, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)  # (batch, width, time / 8)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)  # (batch, 2 * width)
        return self.head(pooled)  # (batch, n_classes) raw scores, softmax is inside the loss


def pick_device() -> torch.device:
    """Apple GPU (MPS) when available, otherwise CPU"""
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def train_model(
    train_grams: list[np.ndarray],
    train_y: np.ndarray,
    n_classes: int,
    epochs: int = 40,
    crop_steps: int = 150,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
    device: torch.device | None = None,
    region: float = 1.0,
) -> MakamCNN:
    """Train a fresh network on random crops. train_y holds integer class indices

    region < 1 draws crops only from the first `region` fraction of each piece
    """
    device = device or pick_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n_inputs = train_grams[0].shape[0] + 1  # + position channel
    model = MakamCNN(n_inputs, n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train_grams))  # new shuffle and new crops every epoch
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            x = np.stack([random_crop(train_grams[i], crop_steps, rng, region) for i in idx])
            xb = torch.from_numpy(x).to(device)
            yb = torch.from_numpy(train_y[idx].astype(np.int64)).to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def predict_proba(
    model: MakamCNN,
    grams: list[np.ndarray],
    mode: str = "crops",
    crop_steps: int = 150,
    device: torch.device | None = None,
    region: float = 1.0,
) -> np.ndarray:
    """Class probabilities for each recording, shape (n_recordings, n_classes)

    mode="full": one pass over the whole recording (region is ignored)
    mode="crops": windows of crop_steps with 50% overlap, probabilities averaged,
        only inside the first `region` fraction of the piece
    """
    device = device or next(model.parameters()).device
    model.eval()  # dropout off, batch norm uses its running averages
    probs = []
    with torch.no_grad():
        for gram in grams:
            if mode == "full":
                x = full_sequence(gram)[np.newaxis]  # batch of one
            elif mode == "crops":
                x = sliding_crops(gram, crop_steps, hop=crop_steps // 2, region=region)
            else:
                raise ValueError(f"mode must be 'full' or 'crops', got {mode!r}")
            p = torch.softmax(model(torch.from_numpy(x).to(device)), dim=1)
            probs.append(p.mean(dim=0).cpu().numpy())  # average over windows
    return np.stack(probs)


def train_and_predict(
    train_grams: list[np.ndarray],
    train_y: np.ndarray,
    test_grams: list[np.ndarray],
    n_classes: int,
    mode: str = "crops",
    **train_kwargs,
) -> np.ndarray:
    """Convenience wrapper: train on one fold, return predicted class indices for the test recordings"""
    model = train_model(train_grams, train_y, n_classes, **train_kwargs)
    crop_steps = train_kwargs.get("crop_steps", 150)
    region = train_kwargs.get("region", 1.0)
    return predict_proba(model, test_grams, mode, crop_steps, region=region).argmax(axis=1)
