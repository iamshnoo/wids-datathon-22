## Based on the notebook https://www.kaggle.com/angiengkh/wids-2022-feature-importance-selection
## along with some minor modifications based on a separate discussion thread.

import pandas as pd

INPUT = "data/"
df_train = pd.read_csv(f"{INPUT}train.csv")
df_test = pd.read_csv(f"{INPUT}test.csv")
df_submission = pd.read_csv(f"{INPUT}sample_solution.csv")

OUTPUT = "new_data/"
new_df_train_path = f"{OUTPUT}train.csv"
new_df_test_path = f"{OUTPUT}test.csv"

# Train - replace nan values with mean
df_train["year_built"] = df_train["year_built"].fillna(df_train["year_built"].mean())
df_train["energy_star_rating"] = df_train["energy_star_rating"].fillna(
    df_train["energy_star_rating"].mean()
)
df_train["direction_max_wind_speed"] = df_train["direction_max_wind_speed"].fillna(
    df_train["direction_max_wind_speed"].mean()
)
df_train["direction_peak_wind_speed"] = df_train["direction_peak_wind_speed"].fillna(
    df_train["direction_peak_wind_speed"].mean()
)
df_train["max_wind_speed"] = df_train["max_wind_speed"].fillna(
    df_train["max_wind_speed"].mean()
)
df_train["days_with_fog"] = df_train["days_with_fog"].fillna(
    df_train["days_with_fog"].mean()
)

# Test - replace nan values with mean
df_test["year_built"] = df_test["year_built"].fillna(df_test["year_built"].mean())
df_test["energy_star_rating"] = df_test["energy_star_rating"].fillna(
    df_test["energy_star_rating"].mean()
)
df_test["direction_max_wind_speed"] = df_test["direction_max_wind_speed"].fillna(
    df_test["direction_max_wind_speed"].mean()
)
df_test["direction_peak_wind_speed"] = df_test["direction_peak_wind_speed"].fillna(
    df_test["direction_peak_wind_speed"].mean()
)
df_test["max_wind_speed"] = df_test["max_wind_speed"].fillna(
    df_test["max_wind_speed"].mean()
)
df_test["days_with_fog"] = df_test["days_with_fog"].fillna(
    df_test["days_with_fog"].mean()
)

# Age of building
def age(df):
    if type(df["year_built"]) == float and pd.isna(df["year_built"]):
        return 0
    else:
        return 2022 - df["year_built"]


df_train["age"] = df_train.apply(age, axis=1)
df_test["age"] = df_test.apply(age, axis=1)

# Commercial buildings differentiated into 2 types
def building_class_group(d):
    if d["building_class"] == "Residential":
        return "Residential"
    elif d["facility_type"] in (
        "Data_Center",
        "Laboratory",
        "Grocery_store_or_food_market",
        "Health_Care_Inpatient",
        "Health_Care_Uncategorized",
        "Health_Care_Outpatient_Uncategorized",
        "Food_Service_Restaurant_or_cafeteria",
    ):
        return "Commercial_24_7"
    else:
        return "Commercial_others"


df_train["building_class_group"] = df_train.apply(building_class_group, axis=1)
df_test["building_class_group"] = df_test.apply(building_class_group, axis=1)

df_train.to_csv(new_df_train_path, encoding="utf-8", header=True)
df_test.to_csv(new_df_test_path, encoding="utf-8", header=True)
