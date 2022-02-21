import pandas as pd
import numpy as np

import h2o
from h2o.automl import H2OAutoML

h2o.init()

INPUT = "new_data/"
train = h2o.import_file(f"{INPUT}train.csv")
test = h2o.import_file(f"{INPUT}test.csv")

x = train.columns
y = "site_eui"
x.remove(y)
x.remove("id")  # removing id

aml = H2OAutoML(max_runtime_secs=3600, seed=42, project_name="wids-datathon-22", nfolds=5)
aml.train(x=x, y=y, training_frame=train)

lb = aml.leaderboard
print(lb.head())

# # Get model ids for all models in the AutoML Leaderboard
# model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])

submission_results = pd.read_csv("data/sample_solution.csv")


# def top_k_avg_predict(
#     k,
#     leaderboard,
# ):
#     lb = leaderboard.as_data_frame()
#     ans = submission_results["site_eui"]
#     for i in range(k):
#         model = lb.loc[i]["model_id"]
#         pred = h2o.get_model(model).predict(test)
#         pred = pred.as_data_frame()
#         ans += np.expm1(pred["predict"]) / k
#     return ans


preds = aml.predict(test)
submission_results["site_eui"] = h2o.as_list(preds)["predict"]
# submission_results.iloc["site_eui"] = top_k_avg_predict(8, aml.leaderboard)
submission_results.to_csv("submission.csv", index=None)
submission_results.head()

h2o.save_model(aml.leader, path="models/")
