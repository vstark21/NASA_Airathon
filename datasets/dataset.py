import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

class PMDataset(Dataset):
    def __init__(
        self,
        files: list,
        use_bands: list,
        transforms=None
    ):
        self.files = files
        self.use_bands = use_bands
        self.transforms = transforms
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        feat_path = self.files[idx]

        _feat = np.load(feat_path)
        data = []
        for i, key in enumerate(self.use_bands):
            _band = _feat[key].transpose(1, 2, 0)[:, :, :2]
            if self.transforms:
                _band = self.transforms(image=_band)['image']
            data.append(_band)
        label = None
        if 'label' in _feat.keys():
            label = _feat['label']
        data = np.concatenate(data, axis=-1)
        data = data.transpose(2, 0, 1)
        return data.astype(np.float32), label
            