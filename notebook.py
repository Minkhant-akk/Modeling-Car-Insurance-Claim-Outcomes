import pandas as pd
import numpy as np
from statsmodels.formula.api import logit
df = pd.read_csv('car_insurance.csv')
df['credit_score'].fillna(df['credit_score'].mean(),inplace=True)
df['annual_mileage'].fillna(df['annual_mileage'].mean(),inplace=True)

print(df.info())

modelList = []
drop_columns = df.drop(columns=["id","outcome"]).columns
for col in drop_columns:
    model = logit(f"outcome ~ {col}",data=df).fit()
    modelList.append(model)

accuracies = []

for models in range(0,len(modelList)):
    conf_matrix = modelList[models].pred_table()
    TN = conf_matrix[0,0]
    TP = conf_matrix[1,1]
    FN = conf_matrix[1,0]
    FP = conf_matrix[0,1]
    accuracy = (TN+TP) / (TN+FN+TP+FP)
    accuracies.append(accuracy)
    print(accuracies)
    
bestPerformModel = drop_columns[accuracies.index(max(accuracies))]

print(bestPerformModel)

best_feature_df = pd.DataFrame({"best_feature":bestPerformModel,
                                "best_accuracy": max(accuracies)},index=[0])
print(best_feature_df)