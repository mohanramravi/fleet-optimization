import time
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from utils.features import add_basic_features


def load_training_data():
    """Load local training data for local run."""
    local_path = os.path.join("data", "train.csv")

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Could not find training file at {local_path}")

    print(f"[LOCAL] Loading dataset from {local_path}")
    return pd.read_csv(local_path)


def main():
    # ------------------------------
    # 1. Load dataset
    # ------------------------------
    start = time.time()
    df = load_training_data()

    # Remove garbage durations
    df = df[(df["duration_sec"] > 20) & (df["duration_sec"] < 7200)]

    # ------------------------------
    # 2. Feature engineering
    # ------------------------------
    df = add_basic_features(df)

    feature_cols = [
        "distance_km",
        "distance_log",
        "lat_diff",
        "lng_diff",
        "bearing",
        "hour_sin",
        "hour_cos",
        "weekday",
        "is_peak",
        "is_weekend",
        "departure_hour",
    ]

    X = df[feature_cols]

    # LOG-transform target for stability
    y = np.log1p(df["duration_sec"])

    # ------------------------------
    # 3. Train/Val split
    # ------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------
    # 4. Train model (XGBoost)
    # ------------------------------
    print("[TRAIN] Training XGBoost model...")

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)

    # ------------------------------
    # 5. Evaluate
    # ------------------------------
    preds = np.expm1(model.predict(X_val))
    true_vals = np.expm1(y_val)

    mae = mean_absolute_error(true_vals, preds)
    print(f"[VAL] MAE: {mae:.2f} seconds")

    # ------------------------------
    # 6. Save model locally
    # ------------------------------
    os.makedirs("model_local", exist_ok=True)
    os.makedirs("model_local", exist_ok=True)
    path = "model_local/travel_time_xgb_2.json"
    model.get_booster().save_model(path)
    print(f"[SAVE] Model saved to: {path}")
    end = time.time()

    print("Raw inference latency:", (end - start)*1000, "ms")



if __name__ == "__main__":
    main()
