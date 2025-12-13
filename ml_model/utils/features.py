import numpy as np
import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate rich feature set for travel-time regression."""
    
    # --- Distance ---
    R = 6371  # km

    lat1 = np.radians(df["origin_lat"])
    lon1 = np.radians(df["origin_lng"])
    lat2 = np.radians(df["dest_lat"])
    lon2 = np.radians(df["dest_lng"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df["distance_km"] = R * c

    # --- Bearing (direction angle) ---
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    df["bearing"] = np.degrees(np.arctan2(y, x))
    df["bearing"] = df["bearing"].fillna(0)

    # --- Basic diffs ---
    df["lat_diff"] = np.abs(df["origin_lat"] - df["dest_lat"])
    df["lng_diff"] = np.abs(df["origin_lng"] - df["dest_lng"])

    # --- Time features ---
    df["is_peak"] = df["departure_hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["departure_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["departure_hour"] / 24)

    df["distance_log"] = np.log1p(df["distance_km"])

    return df
