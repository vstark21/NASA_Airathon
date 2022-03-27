import os
import pandas as pd
import numpy as np
import random
import time
from src.utils import Timer

test_data = pd.read_csv('data/submission_format.csv')
original = pd.read_csv("submission/2022-03-26-17-31/submission_2022-03-26-17-31.csv")
tim = Timer()
random.seed(42)
for i in range(10):
    idx = random.randint(0, len(test_data)-1)
    dt = test_data.iloc[idx]['datetime']
    grid_id = test_data.iloc[idx]['grid_id']
    print(f"Predicting on {dt} {grid_id}")
    tim.reset()
    os.system(f"python predict.py --observation_start_time {dt} --grid_id {grid_id}")
    print(f"Original Value: {original['value'].values[idx]}")
    print(tim.beep())