"""Small image sequences, tubelet tokens and encoders used in the video JEPA labs.

All clips are rendered from a point moving on a seven-by-seven grid. The four
actions are right, down, left and up. A move beyond the edge stays at the edge.
"""

import torch
import torch.nn.functional as F
from torch import nn


MOVES = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]])


def advance(xy, actions):
    return (xy + MOVES[actions]).clamp(0, 6)


def frame(xy):
    grid = torch.arange(12)
    dx = grid[None, None, :] - (xy[:, 0, None, None] + 2)
    dy = grid[None, :, None] - (xy[:, 1, None, None] + 2)
    return torch.exp(-(dx.square() + dy.square()) / 2.0)


def clips(n, seed):
    rng = torch.Generator().manual_seed(seed)
    xy = torch.randint(7, (n, 2), generator=rng)
    result = [frame(xy)]
    for _ in range(3):
        xy = advance(xy, torch.randint(4, (n,), generator=rng))
        result.append(frame(xy))
    return torch.stack(result, dim=1)


def tubelets(video):
    """Return 32 tokens in (time, row, column) order, each with 2 x 3 x 3 values."""
    return video.reshape(-1, 2, 2, 4, 3, 4, 3).permute(0, 1, 3, 5, 2, 4, 6).reshape(-1, 32, 18)


def region_mask(rng):
    """Keep the same randomly placed two-by-two hidden region in both tubelets."""
    r, c = torch.randint(3, (2,), generator=rng)
    mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    mask[:, r : r + 2, c : c + 2] = True
    return (~mask).flatten().nonzero().flatten(), mask.flatten().nonzero().flatten()


def layer(width):
    return nn.TransformerEncoderLayer(
        width, 4, dim_feedforward=2 * width, dropout=0, batch_first=True, activation="gelu"
    )


class VideoEncoder(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.embed = nn.Linear(18, width)
        self.position = nn.Parameter(torch.randn(1, 32, width) * 0.10)
        self.layers = nn.ModuleList([layer(width), layer(width)])

    def forward(self, patches, indices=None):
        h = self.embed(patches) + self.position
        if indices is not None:
            h = h[:, indices]
        levels = []
        for block in self.layers:
            h = block(h)
            levels.append(F.layer_norm(h, (h.shape[-1],)))
        return levels


class VideoPredictor(nn.Module):
    def __init__(self, multi_level=False, width=24):
        super().__init__()
        self.multi_level = multi_level
        self.fuse = (nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
                     if multi_level else nn.Identity())
        self.mask = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.randn(1, 32, width) * 0.12)
        self.block = layer(width)
        self.heads = nn.ModuleList(nn.Linear(width, width) for _ in range(2 if multi_level else 1))

    def forward(self, levels, keep, hide):
        context = self.fuse(torch.cat(levels, -1)) if self.multi_level else levels[-1]
        order = torch.cat([keep, hide])
        slots = torch.cat([context, self.mask.expand(len(context), len(hide), -1)], 1)
        hidden = self.block(slots + self.position[:, order])
        return [head(hidden) for head in self.heads]


@torch.no_grad()
def update_teacher(teacher, student, momentum=0.985):
    for target, online in zip(teacher.parameters(), student.parameters()):
        target.lerp_(online, 1.0 - momentum)
