# Runnable examples

Run these commands from the repository root. The examples use NumPy.

## Cosine retrieval

Evaluate paired embeddings by retrieving each query's partner with cosine similarity.

```sh
python examples/cosine_retrieval.py
```

## Linear CKA

Each row represents one sample. Rotating, uniformly scaling and shifting the feature vectors preserves their centred geometry, so linear CKA gives `1.000` when the rows are aligned. Shuffling the rows of just one matrix breaks the sample pairing. With the fixed seed, the score falls to `0.122`.

```sh
python examples/linear_cka.py
```
