from loguru import logger
import numpy as np
from sklearn.model_selection import KFold
from src.utils import Timer
from tqdm import tqdm
import pickle
import os

def run_kfold(
	train_features, train_labels, test_features,
    n_folds, model, model_params,
	save_dir, name='model', seed=42
):  
    kf = KFold(n_splits=n_folds)
    if seed:
        kf = KFold(
            n_splits=n_folds, 
            shuffle=True,
            random_state=seed
        )
    oof_preds = []
    oof_labels = []
    train_preds = np.zeros((len(train_labels)))
    test_preds = np.zeros((len(test_features)))
    feat_importances = np.zeros((len(train_features[0])))
    tim = Timer()

    logger.info(f"Training model with {n_folds} folds...")
    bar = tqdm(total=n_folds)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
        x_train, x_val = train_features[train_idx], train_features[val_idx]
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]
        reg = model(**model_params)
        reg.fit(x_train, y_train)
        
        # Prediction on train data
        preds = reg.predict(x_train)
        train_preds[train_idx] += preds
        
        # Prediction on val data
        preds = reg.predict(x_val)
        oof_preds.extend(preds.tolist())
        oof_labels.extend(y_val.tolist())
        
        # Prediction on test data
        preds = reg.predict(test_features)
        test_preds += preds

        # Feature importance
        feat_importances += reg.feature_importances_

        pickle.dump(
            reg, 
            open(os.path.join(save_dir, f"{name}_fold-{fold + 1}.pkl"), "wb")
        )
        bar.update()
        
    train_preds /= (n_folds - 1)
    test_preds /= n_folds
    feat_importances /= n_folds

    logger.info(tim.beep("Time taken for training {} folds: ".format(n_folds)))
    return train_preds, test_preds, oof_preds, oof_labels, feat_importances
