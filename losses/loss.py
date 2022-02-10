import torch
import torch.nn as nn
import pytorch_lightning as pl

class PMLoss(pl.LightningModule):
    def __init__(
        self,
    ):
        super(PMLoss, self).__init__()

    def forward(self, x, y):
        return torch.mean((x - y) ** 2)
