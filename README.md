# NASA Airathon

Username: [vstark21](https://www.drivendata.org/users/vstark21/)

## Repo structure

```bash
.
├── README.md 
├── assets
├── configs # Contains config files for training and inference
    ├── predict.yml
    ├── pipeline_0.yml
    ├── model_0.yml
    └── ...
├── data
    ├── backup # Contains backup files
    ├── proc # Contains processed data
    ├── raw # Contains raw, unprocessed data
    ├── train_labels.csv
    ├── grid_metadata.csv
    └── ...
├── notebooks # Contains raw-data processing and EDA notebooks
├── src
    ├── data # Contains data pre-processing and feature engineering functions
    ├── models # Contains modelling functions
    ├── inference # Contains data downloading functions for inference
    ├── visualization # Contains visualization functions
    └── utils # Contains utility functions
├── requirements.txt
├── predict.py # Contains inference code for a single data point
├── train.py # Contains single model training code
├── train_locwise.py # Contains location wise single model training code
├── train_pipeline.py # Contains pipeline training code 
└── train_locwise_pipeline.py # Contains location wise pipeline training code
    
```