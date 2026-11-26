"""Frozen-representation evaluation and small public JEPA building blocks.

Assran et al., I-JEPA
https://arxiv.org/abs/2301.08243
Bardes et al., V-JEPA
https://arxiv.org/abs/2404.08471
"""

import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def features(encoder, X):
    encoder.eval()
    return torch.cat([encoder(x).flatten(1).cpu() for x in X.split(96)]).numpy()


def compare_features(train, test, y, y_test, history):
    for name, (a, b) in zip(("Raw input", "Random encoder", "Trained encoder"), zip(train, test)):
        probe = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=1500, random_state=17)
        )
        probe.fit(a, y)
        pred = probe.predict(b)
        print(
            f"{name}: accuracy {accuracy_score(y_test,pred):.3f} | balanced accuracy {balanced_accuracy_score(y_test,pred):.3f} | macro F1 {f1_score(y_test,pred,average='macro'):.3f}"
        )
    a, b = train[-1], test[-1]
    # Eigenvalues of the centred feature covariance, obtained from singular values.
    singular = np.linalg.svd(b - b.mean(0), compute_uv=False)
    variance = singular**2
    weights = variance[variance > 1e-12] / max(variance.sum(), 1e-12)
    rank = np.exp(-np.sum(weights * np.log(weights))) if len(weights) else 0.0
    print(
        f"Held-out mean feature SD: {b.std(0).mean():.4f} | Effective covariance rank: {rank:.2f}"
    )
    z = PCA(2, svd_solver="randomized", random_state=17).fit(a).transform(b)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.1, 3.2), constrained_layout=True)
    ax.plot(np.arange(1, len(history) + 1), history, color="#42728a")
    ax.set(title="Self-supervised training", xlabel="Epoch", ylabel="Mean training loss")
    sc = ax2.scatter(*z.T, s=12, c=y_test, cmap="tab10", vmin=-0.5, vmax=9.5, alpha=0.8)
    fig.colorbar(sc, ax=ax2, ticks=np.unique(y_test), label="Held-out class")
    ax2.set(title="Frozen test representations", xlabel="Training PC 1", ylabel="Training PC 2")


def probe_encoders(initial, trained, X, T, y, y_test, history):
    tr = [X.flatten(1).numpy(), features(initial, X), features(trained, X)]
    te = [T.flatten(1).numpy(), features(initial, T), features(trained, T)]
    compare_features(tr, te, y, y_test, history)


@torch.no_grad()
def update_teacher(teacher, student, momentum=0.985):
    for shadow, live in zip(teacher.parameters(), student.parameters(), strict=True):
        shadow.lerp_(live, 1 - momentum)


class PatchEncoder(nn.Module):
    def __init__(self, patch_width, tokens, width=32):
        super().__init__()
        self.embed = nn.Linear(patch_width, width)
        self.position = nn.Parameter(torch.randn(1, tokens, width) * 0.12)
        self.layer = nn.TransformerEncoderLayer(
            width, 4, dim_feedforward=72, dropout=0, batch_first=True
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, patches, indices=None):
        z = self.embed(patches) + self.position
        if indices is not None:
            z = z[:, indices]
        return self.norm(self.layer(z))


class MaskPredictor(nn.Module):
    def __init__(self, tokens, width=32):
        super().__init__()
        self.mask = nn.Parameter(torch.randn(1, 1, width) * 0.04)
        self.position = nn.Parameter(torch.randn(1, tokens, width) * 0.12)
        self.layer = nn.TransformerEncoderLayer(
            width, 4, dim_feedforward=72, dropout=0, batch_first=True
        )
        self.head = nn.Linear(width, width)

    def forward(self, context, visible, hidden):
        queries = (self.mask + self.position[:, hidden]).expand(context.shape[0], -1, -1)
        seq = torch.cat([context + self.position[:, visible], queries], dim=1)
        return self.head(self.layer(seq)[:, -len(hidden) :])


def block_mask(times):
    # Two disjoint 2×2 regions of a 4×4 spatial grid. In video a mask spans time.
    selected = torch.randperm(4)[:2]
    grid = torch.arange(times * 16).reshape(times, 4, 4)
    hidden = torch.cat(
        [
            grid[:, 2 * (b // 2) : 2 * (b // 2) + 2, 2 * (b % 2) : 2 * (b % 2) + 2].flatten()
            for b in selected
        ]
    )
    keep = torch.ones(times * 16, dtype=torch.bool)
    keep[hidden] = False
    return torch.arange(times * 16)[keep], hidden


def train_masked(online, predictor, X, times, epochs):
    teacher = copy.deepcopy(online).eval().requires_grad_(False)
    opt = torch.optim.AdamW(
        [*online.parameters(), *predictor.parameters()], lr=0.0015, weight_decay=0.02
    )
    history = []
    for _ in range(epochs):
        total = 0.0
        for ids in torch.randperm(len(X)).split(48):
            visible, hidden = block_mask(times)
            batch = X[ids]
            with torch.no_grad():
                target = teacher(batch)[:, hidden]
                target = nn.functional.layer_norm(target, (target.shape[-1],))
            prediction = predictor(online(batch, visible), visible, hidden)
            loss = nn.functional.smooth_l1_loss(prediction, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_teacher(teacher, online)
            total += loss.item() * len(ids)
        history.append(total / len(X))
    return teacher, history


@torch.no_grad()
def mask_test(online, predictor, teacher, T, times):
    torch.manual_seed(113)
    online.eval()
    predictor.eval()
    errors = []
    for batch in T.split(48):
        visible, hidden = block_mask(times)
        target = teacher(batch)[:, hidden]
        target = nn.functional.layer_norm(target, (target.shape[-1],))
        errors.append(
            (predictor(online(batch, visible), visible, hidden) - target).abs().mean().item()
            * len(batch)
        )
    print(f"Held-out masked latent MAE: {sum(errors)/len(T):.4f}")
