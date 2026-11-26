# Representation Learning

What should a model keep when part of an observation is hidden? A joint-embedding predictive architecture (JEPA) learns to predict a target representation from context. The image and video chapters follow this idea from masked spatial blocks to visual planning. Other chapters approach the same question through contrast, self-distillation, redundancy reduction and reconstruction.

## Worked examples

| Method | What you can test |
| --- | --- |
| [JEPA](01%20JEPA/README.md) | A predictor receives a partially observed digit and predicts the clean view in an embedding space. This small paired-view JEPA uses a moving-average target and a variance penalty to discourage collapse. |
| [I-JEPA](02%20I-JEPA/README.md) | An 8×8 digit is divided into sixteen 2×2 patches. A transformer predicts two held-out spatial blocks in the embedding space of a full-image target encoder. |
| [V-JEPA](03%20V-JEPA/README.md) | A moving disk is divided into non-overlapping 2-frame tubelets of 4×4 pixels. The target is a contextual feature vector for each hidden tubelet. |
| [V-JEPA 2 action-conditioned planning](06%20V-JEPA%202%20action-conditioned%20planning/README.md) | Two-stage action-free pretraining and action-conditioned latent planning on a seven-by-seven board. |
| [V-JEPA 2.1 dense video features](07%20V-JEPA%202.1%20dense%20video%20features/README.md) | An ablation of dense, multi-level self-supervision on moving-point videos, with a frozen foreground probe. |
| [LeJEPA and Gaussian projections](11%20LeJEPA%20and%20Gaussian%20projections/README.md) | Two-view agreement with a characteristic-function test on random one-dimensional projections. |
| [SimCLR](04%20SimCLR/README.md) | Two independently corrupted versions of the same digit form a positive pair. Embeddings from other digits in the batch supply negatives without consulting their labels. |
| [Denoising Autoencoder](05%20Denoising%20Autoencoder/README.md) | An encoder maps a corrupted digit to sixteen numbers. Its decoder reconstructs clean pixel intensities from that bottleneck. |
| [Barlow Twins](08%20Barlow%20Twins/README.md) | Cross-view feature correlation and redundancy reduction on handwritten digits. |
| [DINO self-distillation](09%20DINO%20self-distillation/README.md) | Centred teacher probabilities, a sharpened softmax and cross-view student prediction. |
| [Masked autoencoding](10%20Masked%20autoencoding/README.md) | Reconstruct randomly hidden patches and evaluate a frozen patch encoder. |

## Reading the experiments

The JEPA models are small teaching implementations trained locally. Each lesson states which parts of its source method it implements. No external checkpoint is required. A raw-input or persistence baseline is especially helpful on these deliberately simple datasets.

### Original frozen-probe experiments

Within each row, the three feature sets use the same linear classifier settings. Accuracy is the fraction of correct held-out labels. The video example has four motion classes, and the digit examples have ten classes.

| Encoder | Test task | Raw inputs | Random encoder | Trained encoder | Effective rank after training |
|---|---|---:|---:|---:|---:|
| JEPA | Digits | 0.969 | 0.956 | 0.791 | 2.09 |
| I-JEPA | Digits | 0.969 | 0.980 | 0.980 | 19.19 |
| V-JEPA | Motion | 0.362 | 0.300 | 0.338 | 100.59 |
| SimCLR | Digits | 0.969 | 0.951 | 0.947 | 7.67 |
| Denoising Autoencoder | Digits | 0.969 | 0.880 | 0.916 | 6.91 |

The paired-view JEPA encoder had 32 coordinates and an effective rank of 2.09 on held-out digits. Many coordinates varied together, and its probe lost accuracy. The I-JEPA probe tied its random-encoder baseline on this split. These results describe the small experiments in this collection.

### Action-conditioned planning

The dynamics model holds out 36 distinct state-action pairs. Planning then tests 80 reachable start-goal pairs on the same board with a horizon of three moves. A random plan reaches the goal in 5.0% of trials and the latent planner in 100.0%. These figures refer to the simulated board. [Run and inspect the plans](06%20V-JEPA%202%20action-conditioned%20planning/README.md)

![Learning curve, first test plan and measured success](06%20V-JEPA%202%20action-conditioned%20planning/figure.svg)

### Dense video and digit features

The dense-video exercise probes a labelled foreground at each tubelet with a new set of clips. Barlow Twins, DINO, masked autoencoding and LeJEPA use a common 25% digit holdout with the same training-only linear probe recipe. Their random-encoder and raw-pixel baselines are recorded alongside each trained model. The older digit examples above used a different image split.

## Run the examples

Use Python 3.12, install `requirements.txt` in this folder and open a notebook in its own example directory. All examples also have a matching script. The `gallery` directory contains the shared data and model code.

```sh
python -m pip install -r requirements.txt
python "02 I-JEPA/i_jepa.py"
```

The optical digits are an image-level teaching split of the UCI data bundled with scikit-learn. Images are scaled to [0, 1]. Writer identifiers are unavailable. The moving disks and point trajectories are generated by the scripts. [UCI optical digits](https://doi.org/10.24432/C50P49) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [MIT code licence](../LICENSE)
