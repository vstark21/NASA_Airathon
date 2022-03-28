import os
import pandas as pd
import numpy as np
import random
import time
from src.utils import Timer
from tqdm import tqdm

test_data = pd.read_csv('data/submission_format.csv')
original = pd.read_csv("submission/2022-03-26-17-31/submission_2022-03-26-17-31.csv")
prev = pd.read_csv("submission/2022-03-22-03-33/submission_2022-03-22-03-33.csv")
tim = Timer()
random.seed(42)
for i in tqdm(range(10)):
    idx = random.randint(0, len(test_data)-1)
    dt = test_data.iloc[idx]['datetime']
    grid_id = test_data.iloc[idx]['grid_id']
    print(f"Predicting on {dt} {grid_id}")
    tim.reset()
    os.system(f"python predict.py --observation_start_time {dt} --grid_id {grid_id}")
    os.system(f"rm -rf *.grib2")
    os.system(f"rm -rf *.hdf")
    os.system(f"rm -rf *.nc")
    print(f"Original Value: {original['value'].values[idx]}")
    print(f"Prev Value: {prev['value'].values[idx]}")
    print(tim.beep())
