import torch
import torch.nn as nn
from models import BaseModel
import pytorch_lightning as pl
import losses

class SimpleModel(BaseModel):
    def __init__(self, cfg, optim_cfg, loss_cfg) -> None:
        super(SimpleModel, self).__init__(optim_cfg, loss_cfg)
        self.conv1 = nn.Conv2d(cfg['in_channels'], 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(50176, 10)
        self.linear2 = nn.Linear(10, 1)
        self.act = getattr(nn, cfg['activation'])()
    
    def forward(self, x) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.flatten(x)
        x = self.act(self.linear1(x))
        x = self.linear2(x)
        return x
