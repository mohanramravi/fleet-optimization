import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

s3 = boto3.client("s3")

BUCKET = "fleet-optimization-data-himal"
PREDICTIONS_PREFIX = "predictions/"
OPTIMIZED_PREFIX = "optimized/"

def to_python(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# Fetch the most recent prediction CSV from S3
def get_latest_csv():
    """Return the key of the most recent CSV in predictions/."""
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREDICTIONS_PREFIX)

    if "Contents" not in resp:
        raise FileNotFoundError("No files found in predictions/ folder.")

    csvs = [obj for obj in resp["Contents"] if obj["Key"].endswith(".csv")]
    if not csvs:
        raise FileNotFoundError("No CSV files found in predictions/ folder.")

    latest = sorted(csvs, key=lambda x: x["LastModified"], reverse=True)[0]
    return latest["Key"]

# Embedded Optimizer Function
def assign_jobs(df: pd.DataFrame, max_hours=9.0):
    """
    Assigns the best carrier for each job.

    Logic:
      - For each job:
          * Filter carriers still available
          * Check hours constraint
          * Select the smallest p90 time
          * Remove selected carrier (1 job per carrier)
    """

    assignments = []
    available_carriers = set(df["carrier_id"].unique())

    for job_id, job_group in df.groupby("job_id"):

        # Only carriers still unassigned
        job_group = job_group[job_group["carrier_id"].isin(available_carriers)].copy()

        # Hours constraint
        job_group["can_work"] = (
            job_group["carrier_hours_worked"] + (job_group["p90_time_min"] / 60.0)
        ) <= max_hours

        valid = job_group[job_group["can_work"]]

        if valid.empty:
            assignments.append({
                "job_id": int(job_id),
                "carrier_id": None,
                "reason": "No eligible carriers under 9-hour limit"
            })
            continue

        # Choose carrier with minimum time
        best_row = valid.loc[valid["p90_time_min"].idxmin()]

        assignments.append({
            "job_id": int(best_row["job_id"]),
            "carrier_id": best_row["carrier_id"],
            "p90_time_min": float(best_row["p90_time_min"])
        })

        # Remove this carrier (one job only)
        available_carriers.remove(best_row["carrier_id"])

    return assignments

# Main Lambda Handler

def lambda_handler(event, context):

    try:
        key = get_latest_csv()
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_csv(obj["Body"])
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed loading predictions: {str(e)}"})
        }

    # Step 2  Auto-generate job_id (groups of carriers_per_job)
    carriers_per_job = df["carrier_id"].nunique()
    df["job_id"] = df.index // carriers_per_job

    # Step 3  Convert to optimizer format
    df["p90_time_min"] = df["predicted_time_sec"] / 60.0
    df["carrier_hours_worked"] = 0.0

    # Step 4  Run optimization
    try:
        assignments = assign_jobs(df)
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Optimization failed: {str(e)}"})
        }

    # Step 5  Save optimized results to S3
    out_df = pd.DataFrame(assignments)
    out_key = f"{OPTIMIZED_PREFIX}optimized_{datetime.utcnow().isoformat()}.csv"

    buf = StringIO()
    out_df.to_csv(buf, index=False)

    try:
        s3.put_object(Bucket=BUCKET, Key=out_key, Body=buf.getvalue())
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed writing optimized output: {str(e)}"})
        }

    # Step 6  DELETE the original CSV file
    try:
        s3.delete_object(Bucket=BUCKET, Key=key)
        deleted = True
    except Exception as e:
        deleted = False

    # Step 7  Return results to frontend
    return {
        "statusCode": 200,
        "headers": {"Access-Control-Allow-Origin": "*"},
        "body": json.dumps({
            "optimized_s3_key": out_key,
            "assignments": [
                {k: to_python(v) for k, v in row.items()}
                for row in assignments
            ],
            "deleted_input_file": deleted,
            "deleted_key": key
        })
    }
