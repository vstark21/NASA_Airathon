import os
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from src.utils import check_existance
from typing import Tuple

def _load_features(path_dir: str, total: int) -> pd.DataFrame:
    """
    Load features from .npz files.
    """
    features = defaultdict(lambda:[])
    for idx in tqdm(range(total)):
        filename = os.path.join(path_dir, f"{idx}.npz")
        if not os.path.exists(filename):
            continue
        data = np.load(filename)
        for key in data.keys():
            if key == 'label':
                features[key].append(data[key])
                continue
            _band = data[key].ravel()
            _band = _band[_band != 0]
            mean, std = _band.mean(), _band.std()
            features[key + '_mean'].append(mean)
            features[key + '_std'].append(std)
    return pd.DataFrame(features)

def _impute(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute missing values in train and test data using grid wise mean data.
    """
    feat_columns = train_df.columns
    train_df['grid_id'] = train_metadata['grid_id']
    test_df['grid_id'] = test_metadata['grid_id'].values[:len(test_df)]

    for grid_id in train_metadata['grid_id'].unique():
        for col in feat_columns:
            indices = train_df[train_df['grid_id'] == grid_id].index
            mean_val = train_df.loc[indices, col].mean()
            train_df.loc[indices, col] = train_df.loc[indices, col].fillna(mean_val)
            
            indices = test_df[test_df['grid_id'] == grid_id].index
            test_df.loc[indices, col] = test_df.loc[indices, col].fillna(mean_val)

    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    train_df = train_df.drop(columns=['grid_id'])
    test_df = test_df.drop(columns=['grid_id'])

    return train_df, test_df

def make_dataset(
    work_dir: str, train_dir: str, test_dir: str, train_metafile: str, test_metafile: str,
    filename: str='features.csv.tmp'
) -> None:
    """
    Build features from data.
    """
    tfilename = os.path.join(work_dir, f"train_{filename}")
    tefilename = os.path.join(work_dir, f"test_{filename}")
    try:
        if check_existance(tfilename) and check_existance(tefilename):
            return tfilename, tefilename
    except:
        pass

    train_dir = os.path.join(work_dir, train_dir)
    check_existance(train_dir)
    test_dir = os.path.join(work_dir, test_dir)
    check_existance(test_dir)
    train_metafile = os.path.join(work_dir, train_metafile)
    check_existance(train_metafile)
    test_metafile = os.path.join(work_dir, test_metafile)
    check_existance(test_metafile)

    train_metadata = pd.read_csv(train_metafile)
    train_df = _load_features(train_dir, total=len(train_metadata))

    test_metadata = pd.read_csv(test_metafile)
    test_df = _load_features(test_dir, total=len(test_metadata))

    train_df, test_df = _impute(train_df, test_df, train_metadata, test_metadata)

    train_metadata['mean_value'] = train_metadata.groupby('grid_id')['value'].transform('mean')
    train_df['mean_value'] = train_metadata['mean_value'].values
    test_df['mean_value'] = test_metadata['value'].values[:len(test_df)]

    train_df.to_csv(tfilename, index=False)
    test_df.to_csv(tefilename, index=False)

    return tfilename, tefilename