"""Original geometric scenes and a public handwritten-digit split.

Alpaydin and Kaynak, Optical Recognition of Handwritten Digits
https://doi.org/10.24432/C50P49
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import metrics as met


def digits_split():
    ds = load_digits()
    return train_test_split(
        ds.images.astype("float32") / 16,
        ds.target,
        test_size=0.25,
        random_state=17,
        stratify=ds.target,
    )


def shapes(n, seed, size=32):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    images, masks = [], []
    for i in range(n):
        x, y = rng.integers(8, size - 7, 2)
        w, h = rng.integers(4, 8, 2)
        if i % 2:
            inside = ((xx - x) / w) ** 2 + ((yy - y) / h) ** 2 <= 1
        else:
            inside = (abs(xx - x) <= w) & (abs(yy - y) <= h)
        illumination = (
            0.12 + rng.uniform(-0.12, 0.12) * xx / size + rng.uniform(-0.12, 0.12) * yy / size
        )
        raw = illumination + rng.uniform(0.36, 0.73) * inside + rng.normal(0, 0.105, (size, size))
        images.append(np.clip(raw, 0, 1))
        masks.append(inside)
    return np.asarray(images, np.float32), np.asarray(masks)


def translated_texture(seed):
    rng = np.random.default_rng(seed)
    base = ndi.gaussian_filter(rng.normal(size=(72, 88)), sigma=1.0)
    lo, hi = np.quantile(base, [0.002, 0.998])
    frame = (255 * np.clip((base - lo) / (hi - lo), 0, 1)).astype(np.uint8)
    dx, dy = rng.integers(-3, 4, 2)
    if dx == dy == 0:
        dx = 2
    shifted = np.roll(frame, (dy, dx), axis=(0, 1))
    return frame, shifted, np.array([dx, dy])


def moving_clips(n, seed):
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % 4
    rng.shuffle(labels)
    size = 16
    yy, xx = np.mgrid[:size, :size]
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    clips = []
    for label in labels:
        x, y = rng.integers(0, size, 2)
        vx, vy = directions[label]
        radius = rng.uniform(1.5, 2.7)
        frames = []
        for t in range(6):
            dx = (xx - x - vx * t + size / 2) % size - size / 2
            dy = (yy - y - vy * t + size / 2) % size - size / 2
            disk = dx * dx + dy * dy <= radius * radius
            frames.append(np.clip(0.1 + 0.72 * disk + rng.normal(0, 0.09, (size, size)), 0, 1))
        clips.append(frames)
    return np.array(clips, dtype=np.float32), labels


def class_scores(y, pred, proba=None):
    print(
        f"Accuracy {met.accuracy_score(y,pred):.3f} | Balanced accuracy {met.balanced_accuracy_score(y,pred):.3f} | MCC {met.matthews_corrcoef(y,pred):.3f}"
    )
    print(met.classification_report(y, pred, digits=3, zero_division=0))
    if proba is not None:
        print(
            f"Log loss {met.log_loss(y,proba):.3f} | Macro ROC AUC (one-vs-rest) {met.roc_auc_score(y,proba,multi_class='ovr',average='macro'):.3f}"
        )
        indicator = np.eye(proba.shape[1])[y]
        print(
            f"Macro average precision {met.average_precision_score(indicator,proba,average='macro'):.3f} | Multiclass Brier {np.square(proba-indicator).sum(axis=1).mean():.3f}"
        )


def confusion(ax, y, pred):
    met.ConfusionMatrixDisplay.from_predictions(
        y, pred, ax=ax, cmap="Blues", colorbar=False, include_values=len(np.unique(y)) <= 5
    )
    ax.set_title("Held-out predictions")


def binary_mask_scores(truth, pred):
    truth, pred = np.asarray(truth, bool), np.asarray(pred, bool)
    axes = tuple(range(1, truth.ndim))
    inter = (truth & pred).sum(axis=axes)
    total = truth.sum(axis=axes) + pred.sum(axis=axes)
    div = lambda num, den: np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)
    return {
        "Mean IoU": div(inter, total - inter).mean(),
        "Mean Dice": div(2 * inter, total).mean(),
        "Pixel precision": np.sum(inter) / max(pred.sum(), 1),
        "Pixel recall": np.sum(inter) / max(truth.sum(), 1),
        "Pixel accuracy": (truth == pred).mean(),
    }
