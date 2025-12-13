import time
start = time.time()
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# -------------------------------------------------------
# HARD-CODED CARRIER HOURS (same as Lambda)
# -------------------------------------------------------
CARRIER_HOURS = {
    "C1": 1.2,
    "C2": 2.1,
    "C3": 3.4,
    "C4": 8.9,
    "C5": 10.0,
}

# -------------------------------------------------------
# Utilities
# -------------------------------------------------------
def to_python(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# -------------------------------------------------------
# Core optimization logic (UNCHANGED)
# -------------------------------------------------------
def assign_jobs(df: pd.DataFrame, max_hours=9.0):
    assignments = []
    available_carriers = set(df["carrier_id"].unique())

    for job_id, job_group in df.groupby("job_id"):

        job_group = job_group[
            job_group["carrier_id"].isin(available_carriers)
        ].copy()

        job_group["can_work"] = (
            job_group["carrier_hours_worked"]
            + (job_group["p90_time_min"] / 60.0)
        ) <= max_hours

        valid = job_group[job_group["can_work"]]

        if valid.empty:
            assignments.append({
                "job_id": int(job_id),
                "carrier_id": None,
                "reason": "No eligible carriers under 9-hour limit",
            })
            continue

        best_row = valid.loc[valid["p90_time_min"].idxmin()]

        assignments.append({
            "job_id": int(best_row["job_id"]),
            "carrier_id": best_row["carrier_id"],
            "p90_time_min": float(best_row["p90_time_min"]),
            "carrier_hours_before": float(best_row["carrier_hours_worked"]),
        })

        available_carriers.remove(best_row["carrier_id"])

    return assignments


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main(csv_path: str, out_dir: str):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] Reading predictions from {csv_path}")
    df = pd.read_csv(csv_path)

    # ---------------------------------------------------
    # Step 1 — Auto-generate job IDs
    # ---------------------------------------------------
    carriers_per_job = df["carrier_id"].nunique()
    df["job_id"] = df.index // carriers_per_job

    # ---------------------------------------------------
    # Step 2 — Apply hard-coded carrier hours
    # ---------------------------------------------------
    df["carrier_id_str"] = df["carrier_id"].astype(str)
    df["carrier_hours_worked"] = (
        df["carrier_id_str"].map(CARRIER_HOURS).fillna(0.0)
    )

    # Match Lambda naming
    df["p90_time_min"] = df["predicted_time_min"]

    # ---------------------------------------------------
    # Step 3 — Run optimization
    # ---------------------------------------------------
    print("[OPT] Running assignment optimization...")
    assignments = assign_jobs(df)

    # ---------------------------------------------------
    # Step 4 — Save output locally
    # ---------------------------------------------------
    out_df = pd.DataFrame(assignments)
    ts = datetime.utcnow().isoformat().replace(":", "-")
    out_path = out_dir / f"optimized_{ts}.csv"

    out_df.to_csv(out_path, index=False)

    # ---------------------------------------------------
    # Step 5 — Print summary
    # ---------------------------------------------------
    print("\n=== Optimization Result ===")
    print(out_df)

    print(f"\n[SAVE] Optimized CSV written to: {out_path}")

    # Optional JSON-style preview (Lambda-like)
    print("\n=== JSON Preview ===")
    print(json.dumps(
        [{k: to_python(v) for k, v in row.items()} for row in assignments],
        indent=2
    ))


# -------------------------------------------------------
# CLI entry
# -------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Local carrier-job optimizer")
    ap.add_argument(
        "--csv",
        required=True,
        help="Path to prediction CSV (from local inference)",
    )
    ap.add_argument(
        "--out_dir",
        default="optimized_local",
        help="Directory to save optimized CSV",
    )

    args = ap.parse_args()
    main(args.csv, args.out_dir)


end = time.time()

print(f"\nRaw inference latency: {(end - start) * 1000:.2f} ms")