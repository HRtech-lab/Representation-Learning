"""Small CPU training loops used by the image and video examples.

PyTorch, AdamW
https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
"""

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def seed_cpu(seed=17):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def train_classifier(net, X, y, epochs, lr=0.002):
    data = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(
        data, batch_size=48, shuffle=True, generator=torch.Generator().manual_seed(17)
    )
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        net.train()
        total = 0.0
        for batch, target in loader:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(net(batch), target)
            loss.backward()
            opt.step()
            total += loss.item() * len(target)
        history.append(total / len(data))
    net.eval()
    return history
