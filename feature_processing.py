## Based on https://www.kaggle.com/angiengkh/wids-2022-feature-importance-selection
## along with some minor modifications based on a separate discussion thread.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

INPUT = "new_data/"
df_train = pd.read_csv(f"{INPUT}train.csv")
df_test = pd.read_csv(f"{INPUT}test.csv")

y_train = df_train[["site_eui"]]
X_train = df_train.copy()
X_train = X_train.drop(["site_eui"], axis=1)

## 1. Process numerical variables

# 1.1 Treatment of outlier & perfectly correlated features
to_drop = [
    "year_built",
    "direction_max_wind_speed",
    "direction_peak_wind_speed",
    "facility_type",
    "building_class",
]

X_train = X_train.drop(to_drop, axis=1)
df_test = df_test.drop(to_drop, axis=1)

# 1.2 Remove negative elevation
X_train["ELEVATION"] = X_train["ELEVATION"].apply(lambda x: max(x, 0))
df_test["ELEVATION"] = df_test["ELEVATION"].apply(lambda x: max(x, 0))

non_num = list(
    X_train.select_dtypes(exclude=[np.number]).columns
)  # List all non-numeric attributes
num = list(
    X_train.select_dtypes(include=[np.number]).columns
)  # List all numeric attributes
X_train_non_num = X_train[non_num]
X_train_num = X_train[num]
test_non_num = df_test[non_num]
test_num = df_test[num]

# 1.3 Log transform
def log_transform(data, to_log):
    X = data.copy()
    for item in to_log:
        # Add 1 to the data to prevent infinity values
        X[item] = np.log(1 + X[item])
    return X


to_log = []  # No log transformation is done at the moment
X_train_num_l = log_transform(X_train_num, to_log)
to_log = []  # No log transformation is done at the moment
test_num_l = log_transform(test_num, to_log)

# 1.4 Scale and center the data
scaler = StandardScaler()
scaler.fit(X_train_num_l)
X_train_num_l_n = scaler.transform(X_train_num_l)
X_train_num_l_n = pd.DataFrame(
    data=X_train_num_l_n, index=X_train_num_l.index, columns=X_train_num_l.columns
)
test_num_l_n = scaler.transform(test_num_l)
test_num_l_n = pd.DataFrame(
    data=test_num_l_n, index=test_num_l.index, columns=test_num_l.columns
)

## 2. Process categorical variables

# 2.1 One-hot encode categorical variables
def one_hot_encode(data):
    df = data.copy()
    df = pd.get_dummies(data=df)
    # Drop 1 dummy from each categorial feature to avoid dummy variable trap
    to_drop = ["building_class_group_Commercial_others", "State_Factor_State_1"]
    df = df.drop(to_drop, axis=1)
    return df


X_train_e = one_hot_encode(X_train)
test_e = one_hot_encode(df_test)

# 2.2 Add state 6 to test
test_e["State_Factor_State_6"] = 0

# 2.3 Log transformation
to_log = []  # No log transformation is done at the moment
X_train_e_l = log_transform(X_train_e, to_log)
to_log = []  # No log transformation is done at the moment
test_e_l = log_transform(test_e, to_log)

# 2.4 Scale and center the data
scaler.fit(X_train_e_l)
tmp_train = scaler.transform(X_train_e_l)
X_train_e_l_n = pd.DataFrame(
    data=tmp_train, index=X_train_e.index, columns=X_train_e.columns
)
tmp_test = scaler.transform(test_e_l)
test_e_l_n = pd.DataFrame(data=tmp_test, index=test_e.index, columns=test_e.columns)

## Export train(X,y) and test(X)

X_train_e_l_n.to_csv(f"{INPUT}X.csv", encoding="utf-8", header=True)
y_train.to_csv(f"{INPUT}Y.csv", encoding="utf-8", header=True)
test_e_l_n.to_csv(f"{INPUT}test_new2.csv", encoding="utf-8", header=True)
