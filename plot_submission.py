import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns;sns.set()
import os
import datetime
from loguru import logger

OUTPUT_PATH = 'submission'
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
OUTPUT_PATH = os.path.join(OUTPUT_PATH, TIMESTAMP)
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

logger.add(os.path.join(OUTPUT_PATH, 'logs.log'))

files = [
    # "D:/Repositories/NASA_Airathon/logs/2022-03-07-19-47/submission_2022-03-07-19-47.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-12-10-21/submission_2022-03-12-10-21.csv",
#     "D:/Repositories/NASA_Airathon/logs/2022-03-08-17-19/submission_2022-03-08-17-19.csv",
#     "D:/Repositories/NASA_Airathon/logs/2022-02-23-12-59/submission_2022-02-23-12-59.csv",
#     "D:/Repositories/NASA_Airathon/logs/2022-03-13-19-14/submission_2022-03-13-19-14.csv",
#     "D:/Repositories/NASA_Airathon/logs/2022-03-13-19-31/submission_2022-03-13-19-31.csv",
#     "D:/Repositories/NASA_Airathon/logs/2022-03-13-19-40/submission_2022-03-13-19-40.csv",
#     "D:/Repositories/NASA_Airathon/logs/2022-03-13-23-28/submission_2022-03-13-23-28.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-14-10-56/submission_2022-03-14-10-56.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-14-11-19/submission_2022-03-14-11-19.csv",
    "D:/Repositories/NASA_Airathon/logs/2022-03-14-11-41/submission_2022-03-14-11-41.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-14-22-37/submission_2022-03-14-22-37.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-14-22-53/submission_2022-03-14-22-53.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-15-19-26/submission_2022-03-15-19-26.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-15-19-44/submission_2022-03-15-19-44.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-15-19-58/submission_2022-03-15-19-58.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-15-21-21/submission_2022-03-15-21-21.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-15-23-13/submission_2022-03-15-23-13.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-16-00-38/submission_2022-03-16-00-38.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-16-11-08/submission_2022-03-16-11-08.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-16-12-31/submission_2022-03-16-12-31.csv",
    # "D:/Repositories/NASA_Airathon/logs/2022-03-16-13-37/submission_2022-03-16-13-37.csv",
    "D:/Repositories/NASA_Airathon/submission.csv"
]

scores = [
    # 0.6967,
    # 0.6866,
    # 0.6954,
    # 0.6963,
    # "N/A",
    # "N/A",
    # "N/A",
    # "N/A",
    # "N/A",
    # "N/A",
    0.7453,
    # "N/A",
    # "N/A",
    # 0.7429,
    # "N/A",
    # "N/A",
    # 0.7385,
    # "N/A",
    # "N/A",
    # 0.6890,
    # "N/A",
    # "N/A",
    "N/A"
]

preds = 0
for i, filename in enumerate(files):
    logger.info(f"Reading {filename} which has score {scores[i]}...")
    sub = pd.read_csv(filename)
    sns.distplot(
        sub['value'], 
        label=f"{os.path.basename(filename)}_{str(scores[i])}",
        hist=False
    )
    preds += (sub['value'] / len(files))

sub['value'] = preds
sub.to_csv(os.path.join(OUTPUT_PATH, f'submission_{TIMESTAMP}.csv'), index=False)

sns.distplot(
    sub['value'], 
    label="Average",
    hist=False)
plt.legend()
plt.show()
