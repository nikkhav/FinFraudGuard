import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / 'models' / 'model_randomforest.pkl')
with open(BASE_DIR / 'models' / 'feature_columns.json') as f:
    feature_cols = json.load(f)

def predict_transaction(input_dict: dict):
    input_df = pd.DataFrame([input_dict])

    input_df["Hour"] = (input_df["Time"] / 3600) % 24
    input_df["is_night"] = ((input_df["Hour"] < 6) | (input_df["Hour"] > 22)).astype(int)

    input_df["log_Amount"] = np.log1p(input_df["Amount"])
    input_df["is_high_amount"] = (input_df["Amount"] > 1000).astype(int)

    input_df = input_df.drop(columns=["Time"])

    input_df = input_df[feature_cols]

    proba = model.predict_proba(input_df)[0][1]
    label = proba > 0.5

    return {
        "fraud": bool(label),
        "probability": round(float(proba), 4)
    }