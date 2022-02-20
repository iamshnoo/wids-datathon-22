import pandas as pd
import numpy as np

from flaml import AutoML
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

import ray
ray.init(num_cpus=16, num_gpus=1,_temp_dir='ray_results/')


kf = KFold(n_splits=5, shuffle=True, random_state=42)

INPUT = "data/"
ID = "id"
TARGET = "site_eui"
SUBMISSION_PATH = "submission.csv"

if __name__ == '__main__':

    df_train = pd.read_csv(f'{INPUT}train.csv')
    df_test = pd.read_csv(f'{INPUT}test.csv')
    df_submission = pd.read_csv(f'{INPUT}sample_solution.csv')

    train = df_train.drop([TARGET, ID], axis=1)
    test = df_test.drop( [ID], axis=1)
    target = df_train.site_eui

    print(f'Train set has {train.shape[0]} rows and {train.shape[1]} columns.')
    print(f'Test set has {test.shape[0]} rows and {test.shape[1]} columns.')

    automl = AutoML()

    # Specify automl goal and constraint
    automl_settings = {
        "time_budget": 60,  # in seconds
        #"max_iter": 2,
        "estimator_list": ['lgbm','catboost', 'xgboost'],
        "eval_method": "cv",
        "split_type" : kf,
        "n_jobs": 4,
        "n_concurrent_trials": 4,
        "metric": 'rmse',
        "task": 'regression',
        "log_file_name": "flaml.log"}

    # Train with labeled input data
    automl.fit(train, target, **automl_settings)

    print('Best ML leaner:', automl.best_estimator)
    print('Best hyperparmeter config:', automl.best_config)
    print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))

    # Make prediction
    y_pred= automl.predict(test)
    print(y_pred)

    df_submission.site_eui = y_pred
    df_submission.to_csv(SUBMISSION_PATH,index=False)
