# import boto3
# import pandas as pd
# from io import StringIO
#
#
# # ---------------------------------------------------------
# # 1. Load the ML predictions CSV from S3
# # ---------------------------------------------------------
# def load_predictions_from_s3(bucket: str, key: str) -> pd.DataFrame:
#     """
#     Reads the (carrier → job) ML predictions from S3.
#     The file must contain:
#         carrier_id, job_id, p90_time_min, carrier_hours_worked
#     """
#     s3 = boto3.client("s3")
#     obj = s3.get_object(Bucket=bucket, Key=key)
#     csv_data = obj["Body"].read().decode("utf-8")
#     df = pd.read_csv(StringIO(csv_data))
#     return df
#
#
# # ---------------------------------------------------------
# # 2. Assign the best carrier for each job
# # ---------------------------------------------------------
# def assign_jobs(df: pd.DataFrame, max_hours=9.0):
#     """
#     df must contain:
#       carrier_id, job_id, p90_time_min, carrier_hours_worked
#
#     Logic:
#       - For each job_id:
#           * Filter carriers that violate: hours_worked + p90_time <= max_hours
#           * Choose the carrier with minimum p90_time
#           * Assign carrier to the job
#           * Remove carrier from pool (carrier can handle only 1 job)
#     """
#
#     assignments = []  # list of {carrier_id, job_id, p90_time_min}
#
#     # Track which carriers are still free
#     available_carriers = set(df["carrier_id"].unique())
#
#     # Group by jobs
#     for job_id, job_group in df.groupby("job_id"):
#
#         # Filter only carriers still available
#         job_group = job_group[job_group["carrier_id"].isin(available_carriers)]
#
#         # Constraint: hours_worked + p90_time <= max_hours
#         job_group["can_work"] = (
#             job_group["carrier_hours_worked"] + (job_group["p90_time_min"] / 60.0)
#         ) <= max_hours
#
#         valid = job_group[job_group["can_work"]]
#
#         if valid.empty:
#             # No valid carriers → job cannot be assigned
#             assignments.append({
#                 "job_id": job_id,
#                 "carrier_id": None,
#                 "reason": "No eligible carriers under 9-hour limit"
#             })
#             continue
#
#         # Choose carrier with minimum P90 time
#         best_row = valid.loc[valid["p90_time_min"].idxmin()]
#
#         assignments.append({
#             "job_id": best_row["job_id"],
#             "carrier_id": best_row["carrier_id"],
#             "p90_time_min": best_row["p90_time_min"]
#         })
#
#         # Remove assigned carrier from pool
#         available_carriers.remove(best_row["carrier_id"])
#
#     return assignments
#
#
# # ---------------------------------------------------------
# # 3. Wrapper: Load → Assign → Return
# # ---------------------------------------------------------
# def run_optimizer(bucket: str, key: str):
#     df = load_predictions_from_s3(bucket, key)
#     results = assign_jobs(df)
#     return results
#
#
# # ---------------------------------------------------------
# # 4. Example Usage
# # ---------------------------------------------------------
# if __name__ == "__main__":
#     bucket_name = "your-s3-bucket"
#     file_key = "ml_outputs/predictions.csv"
#
#     assignments = run_optimizer(bucket_name, file_key)
#
#     print("\n=== FINAL ASSIGNMENTS ===")
#     for a in assignments:
#         print(a)


import pandas as pd


# ---------------------------------------------------------
# Load the predictions locally (no S3 needed)
# ---------------------------------------------------------
def load_predictions_local(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


# ---------------------------------------------------------
# Assign jobs based on constraints
# ---------------------------------------------------------
def assign_jobs(df: pd.DataFrame, max_hours=9.0):
    assignments = []
    available_carriers = set(df["carrier_id"].unique())

    for job_id, job_group in df.groupby("job_id"):

        # Only carriers that haven't been assigned yet
        job_group = job_group[job_group["carrier_id"].isin(available_carriers)]

        # Check work-hour constraint
        job_group = job_group.copy()
        job_group["can_work"] = (
            job_group["carrier_hours_worked"] + (job_group["p90_time_min"] / 60.0)
        ) <= max_hours

        valid = job_group[job_group["can_work"]]

        if valid.empty:
            assignments.append({
                "job_id": job_id,
                "carrier_id": None,
                "reason": "No eligible carriers under 9-hour limit"
            })
            continue

        # Pick fastest carrier
        best_row = valid.loc[valid["p90_time_min"].idxmin()]

        assignments.append({
            "job_id": best_row["job_id"],
            "carrier_id": best_row["carrier_id"],
            "p90_time_min": best_row["p90_time_min"]
        })

        # Remove carrier from future assignments
        available_carriers.remove(best_row["carrier_id"])

    return assignments


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    df = load_predictions_local("optimizertest.csv")
    results = assign_jobs(df)

    print("\n=== FINAL ASSIGNMENTS ===")
    for r in results:
        print(r)
