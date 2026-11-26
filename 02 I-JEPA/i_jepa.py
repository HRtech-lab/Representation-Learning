"""I-JEPA

An 8×8 digit is divided into sixteen 2×2 patches. A transformer predicts two held-out spatial blocks in the embedding space of a full-image target encoder.

Assran et al., Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (2023)
https://arxiv.org/abs/2301.08243
Alpaydin and Kaynak, Optical Recognition of Handwritten Digits, UCI, CC BY 4.0
https://doi.org/10.24432/C50P49"""

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
from gallery.representations import PatchEncoder, MaskPredictor, train_masked, mask_test


def patchify(tensor):
    return tensor.reshape(-1, 4, 2, 4, 2).permute(0, 1, 3, 2, 4).reshape(-1, 16, 4)


def main():
    seed_cpu()
    train, test, y, y_test = digits_split()
    X = patchify(torch.tensor(train))
    T = patchify(torch.tensor(test))
    encoder = PatchEncoder(X.shape[2], X.shape[1])
    initial = copy.deepcopy(encoder)
    predictor = MaskPredictor(X.shape[1])
    teacher, history = train_masked(encoder, predictor, X, times=1, epochs=16)
    mask_test(encoder, predictor, teacher, T, times=1)

    probe_encoders(initial, encoder, X, T, y, y_test, history)
    plt.show()


if __name__ == "__main__":
    main()
