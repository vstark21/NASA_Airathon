# General imports
import os
import sys
import cv2
import time
import glob
import json
import yaml
import scipy
import random
import warnings
warnings.filterwarnings('ignore')
import datetime
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from src.data import make_dataset
from src.models import run_kfold
from xgboost import XGBRegressor

from src.utils import *
from loguru import logger
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config', type=str, help='Path to the config file', default='configs/config.yml')
args = parser.parse_args()

if __name__ == '__main__':
	# Config
    with open(args.config, "r") as f:
        config = AttrDict(yaml.safe_load(f))
    config.OUTPUT_PATH = os.path.join(config.OUTPUT_PATH, TIMESTAMP)
    if not os.path.exists(config.OUTPUT_PATH):
        os.makedirs(config.OUTPUT_PATH)
    logger.add(os.path.join(config.OUTPUT_PATH, 'logs.log'))
    logger.info(f"Config:{str(config)}")
    
    tfilename, tefilename = make_dataset(
        '', config.TRAIN_DIR, config.TEST_DIR, 
        config.TRAIN_METAFILE, config.TEST_METAFILE, config.GRID_METAFILE
    )
    train_df = pd.read_csv(tfilename)
    test_df = pd.read_csv(tefilename)

    train_labels = train_df['label'].to_numpy()
    train_df = train_df.drop(['label'], axis=1)
    test_df = test_df.drop(['label'], axis=1)

    train_features = train_df.to_numpy()
    test_features = test_df.to_numpy()

    logger.info(f"Using following features: {train_df.columns}")
    logger.info(f"Found {len(train_features)} training instances")
    train_preds, test_preds, oof_preds, oof_labels, feat_importances = run_kfold(
        train_features, train_labels, test_features, config.N_FOLDS,
        XGBRegressor, config.XGB_PARAMS, config.OUTPUT_PATH, name='xgb'
    )

    metrics = compute_metrics(train_preds, train_labels)
    for k, v in metrics.items():
        logger.info(f"Average train_{k}: {np.mean(v)}")
    metrics = compute_metrics(np.array(oof_preds), np.array(oof_labels))
    for k, v in metrics.items():
        logger.info(f"Average eval_{k}: {np.mean(v)}")

    submission = pd.read_csv(config.TEST_METAFILE)
    submission['value'] = np.concatenate(
        (test_preds, submission['value'].values[len(test_df):])
    )
    submission.to_csv(os.path.join(config.OUTPUT_PATH, f'submission_{TIMESTAMP}.csv'), index=False)
    submission.head()

    plt.barh(train_df.columns, feat_importances)
    plt.yticks(fontsize='xx-small')
    plt.savefig(os.path.join(config.OUTPUT_PATH, 'feature_importance.png'))
    plt.show()
