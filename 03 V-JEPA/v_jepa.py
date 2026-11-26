"""V-JEPA

A moving disk is divided into non-overlapping 2-frame tubelets of 4×4 pixels. The target is a contextual feature vector for each hidden tubelet.

Bardes et al., Revisiting Feature Prediction for Learning Visual Representations from Video (2024)
https://arxiv.org/abs/2404.08471"""

from pathlib import Path
import sys

start = Path(__file__).resolve().parent
for parent in (start, *start.parents):
    if (parent / "gallery" / "__init__.py").is_file():
        sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Open this example inside the Representation Learning folder.")


import copy
import matplotlib.pyplot as plt
import torch
from torch import nn
from gallery.vision import digits_split
from gallery.neural import seed_cpu
from gallery.representations import probe_encoders, update_teacher

from gallery.vision import moving_clips
from gallery.representations import PatchEncoder, MaskPredictor, train_masked, mask_test


def patchify(tensor):
    return tensor.reshape(-1, 3, 2, 4, 4, 4, 4).permute(0, 1, 3, 5, 2, 4, 6).reshape(-1, 48, 32)


def main():
    seed_cpu()
    train, y = moving_clips(576, 17)
    test, y_test = moving_clips(160, 79)
    X = patchify(torch.tensor(train))
    T = patchify(torch.tensor(test))
    encoder = PatchEncoder(X.shape[2], X.shape[1])
    initial = copy.deepcopy(encoder)
    predictor = MaskPredictor(X.shape[1])
    teacher, history = train_masked(encoder, predictor, X, times=3, epochs=14)
    mask_test(encoder, predictor, teacher, T, times=3)

    probe_encoders(initial, encoder, X, T, y, y_test, history)
    plt.show()


if __name__ == "__main__":
    main()
