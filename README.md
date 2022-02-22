# Experiments for WIDS datathon 22

```main.py``` has a FLAML based AutoML solution. I got a best score of 30
something using the original data and some set of hyperparameters that I dont
remember + forgot to track.

```feature_engineering.py``` adds a couple of simple features based on a kaggle
EDA notebook that I liked.

```feature_processing.py``` tries some processing steps on the dataset.

```automl.py``` has a H2O based AutoML solution. I got a best score of about 50
something using this with the new data files generated from the feature
engineering process.

```submission.csv``` represents the latest submission from my end (not
necessarily the best one).
