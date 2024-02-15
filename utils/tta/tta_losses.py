import torch
import torch.nn as nn


@torch.jit.script
def entropy_loss(x):
    return torch.mean(-(x.softmax(1) * x.log_softmax(1)).sum(1), 0)


@torch.jit.script
def ema_entropy_loss(x, x_ema):
    return torch.mean((x.softmax(1) - x_ema.softmax(1)).pow(2).sum(1), 0)


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        return entropy_loss(x)


class EmaEntropyLoss(nn.Module):
    def __init__(self):
        super(EmaEntropyLoss, self).__init__()

    def forward(self, x, x_ema):
        return ema_entropy_loss(x, x_ema)
