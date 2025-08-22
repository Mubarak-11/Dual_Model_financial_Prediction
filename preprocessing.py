#Load dataset, preprocess and into tensors.
#Target column is "isfraud" for classification and "amount" for regression

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#need a quick function for log1p clip calculation, can't use lambda in pkl files
def log1p_clip(v):
        return np.log1p(np.clip(v, 0, None))


def preprocess(df:pd.DataFrame, task: str = "classification"):

    df = df.copy()
    target_column = None

    #choose which task!
    if task == 'classification':
        if "isFraud" not in df.columns:
                raise ValueError("column isFraud not found for classification.")
        target_column = "isFraud"
        y = df[target_column].astype("float32")
    
    elif task == 'regression':
        if "amount" not in df.columns:
            raise ValueError("column amount not found in classification.")
        target_column = "amount"
        y = np.log1p(df[target_column].astype("float32")) #apply log1p for stability

    else:
         raise ValueError(f"Unknown Task: {task}")

    base_cols = [
        "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"
    ]

    present = [c for c in base_cols if c in df.columns] #keep only the columns that exist

    #For regression, exclude the column "amount":
    if task == "regression":
        feature_cols = [c for c in present if c != "amount"]
    else:
        feature_cols = present #classification gets to keep amount


    #split:
    X = df[feature_cols].copy()

    #onehot encode column: Type
    cat_feature = ['type']

    # money columns to compress loss and help gradients calculation be smooth
    money_cols_all = ['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']

    cat_cols_all = [col for col in cat_feature if col in X.columns]
    money_cols = [col for col in money_cols_all if col in X.columns]
    other_num = [col for col in feature_cols if col not in cat_cols_all + money_cols] 

    #log1p will help clip extreme ranges in data(monetary fields)
    log1p_tf = FunctionTransformer(log1p_clip, validate=False)

    #preprocess now
    transformers = []
    if cat_cols_all:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols_all))
    if money_cols:
        transformers.append(("money", Pipeline([("log1p", log1p_tf), ("scale", StandardScaler())]), money_cols))
    if other_num:
        transformers.append(("num", StandardScaler(), other_num))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return X,y, preprocessor
