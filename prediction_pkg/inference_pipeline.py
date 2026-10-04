import pickle
import json
import pandas as pd
import os
import numpy as np


def load_model():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, "model.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_scaler():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    scaler_path = os.path.join(dir_path, "scaler.pkl")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def load_params():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    params_path = os.path.join(dir_path, "params.json")
    with open(params_path, "r") as f:
        return json.load(f)


def preprocess_data(raw_data, params, scaler):
    df = raw_data.copy()

    # Tell Pandas which columns should be numeric.
    numeric_columns = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Replace any '?' strings with Pandas NaN
    df = df.replace('?', pd.NA)

    # Fill numeric columns with 0
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(0)

    # Fill categorical columns with 'Unknown'
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')

    if 'capital.gain' in df.columns:
        df['capital_gain_log'] = np.log1p(df['capital.gain'])
        df['is_capital_maxed'] = (df['capital.gain'] == 99999).astype(int)

    df = pd.get_dummies(df)

    # Ensure exactly the right columns from params.json remain
    df = df.reindex(columns=params["expected_columns"], fill_value=0)

    X_scaled = scaler.transform(df)

    return X_scaled


def predict_new_data(raw_data):
    model = load_model()
    scaler = load_scaler()
    params = load_params()

    X_final = preprocess_data(raw_data, params, scaler)

    return model.predict(X_final)