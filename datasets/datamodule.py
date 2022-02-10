import os
import torch
import albumentations as A
import pytorch_lightning as pl
from dataset import PMDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class PMDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_dir,
        use_bands,
        tdataloader_cfg,
        vdataloader_cfg,
        ttransforms,
        vtransforms,
    ) -> None:
        super(PMDataModule, self).__init__()
        self.data_dir = data_dir
        self.tdataloader_cfg = tdataloader_cfg
        self.vdataloader_cfg = vdataloader_cfg
        self.ttransforms = ttransforms
        self.vtransforms = vtransforms
        self.use_bands = use_bands
    
    def prepare_data(self) -> None:
        if self.ttransforms:
            self.ttransforms = A.Compose([
                getattr(A, k)(**v) for k, v in self.ttransforms.items()
            ])
        if self.vtransforms:
            self.vtransforms = A.Compose([
                getattr(A, k)(**v) for k, v in self.vtransforms.items()
            ])
    
    def setup(self, stage) -> None:
        files = [
            os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)
        ]
        if stage == 'fit' or stage is None:
            train_files, val_files = train_test_split(files, test_size=0.2)
            self.train_dataset = PMDataset(
                train_files, transforms=self.ttransforms, use_bands=self.use_bands
            )
            self.val_dataset = PMDataset(
                val_files, transforms=self.vtransforms, use_bands=self.use_bands
            )
        elif stage == 'test':
            self.test_dataset = PMDataset(
                files, transforms=self.vtransforms, use_bands=self.use_bands
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self.tdataloader_cfg)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self.vdataloader_cfg)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, **self.vdataloader_cfg)
        