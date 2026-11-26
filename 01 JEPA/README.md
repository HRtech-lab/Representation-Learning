# JEPA

A predictor receives a partially observed digit and predicts the clean view in an embedding space. This small paired-view JEPA uses a moving-average target and a variance penalty to discourage collapse.

[Executed notebook](jepa.ipynb) · [Python script](jepa.py)

## Context and target

The two encoders start with the same weights. Only the online encoder and predictor receive gradients.

## Predict a clean embedding

Each coordinate is hidden independently with probability 0.32. The variance penalty follows VICReg. Digit labels stratify the initial partition and train the downstream classifier. The embedding objective uses only the images.

## Frozen linear probes

Identical probe settings compare raw pixels, the initial encoder and the learned encoder. PCA is fitted on training features.

## Recorded output

```text
Held-out latent MAE: 0.1697
Raw input: accuracy 0.969 | balanced accuracy 0.969 | macro F1 0.969
Random encoder: accuracy 0.956 | balanced accuracy 0.955 | macro F1 0.955
Trained encoder: accuracy 0.791 | balanced accuracy 0.790 | macro F1 0.787
Held-out mean feature SD: 2.1801 | Effective covariance rank: 2.09
```

## Reading the result

A small prediction loss alone can be achieved by collapsing the representation. Feature spread, effective rank and the random-encoder probe give the loss a scale.

![JEPA, figure 1](figure.svg)

## References

[LeCun, A Path Towards Autonomous Machine Intelligence (2022)](https://openreview.net/pdf?id=BZ5a1r-kVsf)

[Bardes, Ponce and LeCun, VICReg (2022)](https://arxiv.org/abs/2105.04906)

[Alpaydin and Kaynak, Optical Recognition of Handwritten Digits, UCI, CC BY 4.0](https://doi.org/10.24432/C50P49)

[Frozen probes and masked-prediction training](../gallery/representations.py)
