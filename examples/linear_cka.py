"""Compare two feature spaces on the same ordered samples using linear CKA."""
import numpy as np


def linear_cka(x, y):
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Expected feature matrices with matching sample rows")
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    if denominator == 0:
        raise ValueError("Both feature spaces must vary across samples")
    return np.linalg.norm(x.T @ y) ** 2 / denominator


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    features = rng.standard_normal((64, 5))
    rotation, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    transformed = 3 * features @ rotation + 7
    aligned = linear_cka(features, transformed)
    shuffled = linear_cka(features, rng.permutation(transformed))
    assert np.isclose(aligned, 1) and shuffled < aligned
    print(f"Aligned rows: {aligned:.3f}; shuffled rows: {shuffled:.3f}")
