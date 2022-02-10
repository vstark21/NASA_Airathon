import torch
import models
import yaml
import argparse
import pytorch_lightning as pl

from utils import AttrDict
from datasets import PMDataModule
import warnings
warnings.filterwarnings('ignore')

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config', type=str, help='Path to the config file', default='configs/config.yml')
args = parser.parse_args()

if __name__ == '__main__':

	with open(args.config, "r") as f:
		config = AttrDict(yaml.safe_load(f))

	dm = PMDataModule(
		config.DATA_DIR, config.USE_BANDS, 
		config.TRAIN_DATALOADER_CFG, config.VAL_DATALOADER_CFG, 
		config.TRAIN_TRANSFORMS, config.VAL_TRANSFORMS
	)
	model = getattr(models, config.MODEL.pop('type'))(
		config.MODEL['cfg'], config.OPTIMIZER_CFG, config.LOSS_CFG
	)
	trainer = pl.Trainer(**config.TRAINER_CFG)
	trainer.fit(model, dm)
	trainer.test(datamodule=dm)
