import torch
import losses
import torch.nn as nn
import pytorch_lightning as pl

class BaseModel(pl.LightningModule):
    def __init__(self, optim_cfg, loss_cfg) -> None:
        super(BaseModel, self).__init__()
        self.optim_cfg = optim_cfg
        self.loss_fn = getattr(losses, loss_cfg.pop('type'))(**loss_cfg)
    
    def forward(self, x) -> torch.Tensor:
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return getattr(torch.optim, self.optim_cfg.pop('type'))(
            self.parameters(), **self.optim_cfg
        )
    