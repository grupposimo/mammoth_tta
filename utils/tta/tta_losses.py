import torch
import torch.nn as nn


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        return torch.mean(-(x.softmax(1) * x.log_softmax(1)).sum(1), 0)


class EmaEntropyLoss(nn.Module):
    def __init__(self):
        super(EmaEntropyLoss, self).__init__()

    def forward(x, x_ema):
        return torch.mean(-(x_ema.softmax(1) * x.log_softmax(1)).sum(1), 0)
