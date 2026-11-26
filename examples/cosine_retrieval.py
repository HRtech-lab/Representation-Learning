"""Evaluate paired embeddings by retrieving each query's partner with cosine similarity."""
import numpy as np


def unit_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cosine similarity requires non-zero vectors")
    return matrix / norms


if __name__ == "__main__":
    candidates = np.array([[0, 2], [-3, 0], [4, 0]], dtype=float)
    queries = np.array([[2, 0.1], [-1, 0.2], [0.1, 2]], dtype=float)
    partner_indices = np.array([2, 1, 0])
    similarities = unit_rows(queries) @ unit_rows(candidates).T
    nearest = similarities.argmax(axis=1)
    recall_at_one = np.mean(nearest == partner_indices)
    assert recall_at_one == 1.0
    print(f"Retrieved rows: {nearest.tolist()}, recall@1: {recall_at_one:.2f}")
