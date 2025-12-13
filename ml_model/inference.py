import time
start = time.time()
import xgboost as xgb
import pandas as pd
import numpy as np

from utils.features import add_basic_features

# -------------------------------------------------------
# 1. HARDCODED CARRIER LOCATIONS
# -------------------------------------------------------
CARRIERS = [
    {"id": "C1", "lat": 39.0082, "lng": -76.9597},
    {"id": "C2", "lat": 39.0150, "lng": -76.9401},
    {"id": "C3", "lat": 39.0205, "lng": -76.9305},
    {"id": "C4", "lat": 39.0165, "lng": -76.9273},
    {"id": "C5", "lat": 38.9845, "lng": -76.9676},
]

FEATURE_COLS = [
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

# -------------------------------------------------------
# 2. Load XGBoost Booster (local)
# -------------------------------------------------------
def load_model(path="model_local/travel_time_xgb.json"):
    booster = xgb.Booster()
    booster.load_model(path)
    print(f"[LOAD] Loaded XGBoost model from {path}")
    return booster


# -------------------------------------------------------
# 3. Predict travel time for ALL carriers
# -------------------------------------------------------
def predict_for_all_carriers(
    booster,
    dest_lat,
    dest_lng,
    departure_hour,
    weekday,
):
    results = []

    for c in CARRIERS:
        df = pd.DataFrame([{
            "origin_lat": c["lat"],
            "origin_lng": c["lng"],
            "dest_lat": dest_lat,
            "dest_lng": dest_lng,
            "departure_hour": departure_hour,
            "weekday": weekday,
        }])

        # Feature engineering
        df = add_basic_features(df)

        # DMatrix
        dmatrix = xgb.DMatrix(df[FEATURE_COLS])

        # Predict log(duration)
        pred_log = booster.predict(dmatrix)[0]
        pred_sec = float(np.expm1(pred_log))

        results.append({
            "carrier_id": c["id"],
            "origin_lat": c["lat"],
            "origin_lng": c["lng"],
            "dest_lat": dest_lat,
            "dest_lng": dest_lng,
            "departure_hour": departure_hour,
            "weekday": weekday,
            "predicted_time_sec": pred_sec,
            "predicted_time_min": pred_sec / 60.0,
        })

    return pd.DataFrame(results)


# -------------------------------------------------------
# 4. MAIN (Local Test)
# -------------------------------------------------------
if __name__ == "__main__":
    

    booster = load_model("model_local/travel_time_xgb.json")

    predictions_df = predict_for_all_carriers(
        booster=booster,
        dest_lat=38.99,
        dest_lng=-76.95,
        departure_hour=17,
        weekday=2,
    )

   

    print("\n=== Carrier Travel Time Predictions ===")
    print(predictions_df[[
        "carrier_id",
        "predicted_time_min"
    ]])

 


end = time.time()

print(f"\nRaw inference latency: {(end - start) * 1000:.2f} ms")
