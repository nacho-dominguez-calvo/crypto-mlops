import os
import io
import logging
import pandas as pd
from src.data.storage import S3Storage
from src.data.features.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_latest_ingestion_run(s3: S3Storage) -> str:
    """
    Find the latest ingestion_<id> run available in the S3 bucket.
    Assumes objects are stored under prefixes like 'ingestion_<id>/'.
    """
    all_files = s3.list_files(prefix="ingestion_")
    if not all_files:
        raise RuntimeError("No ingestion runs found in S3")

    runs = {f.split("/")[0] for f in all_files}
    latest_run = sorted(runs)[-1]  # pick the last one lexicographically
    logger.info(f"Using latest ingestion run: {latest_run}")
    return latest_run


def main():
    # Load configuration from environment with sensible defaults
    bucket_name = os.getenv("AWS_BUCKET_NAME")
    region_name = os.getenv("AWS_REGION")

    s3 = S3Storage(bucket_name, region_name)

    # 1. Find the latest ingestion run
    latest_run = get_latest_ingestion_run(s3)

    # 2. Read historical data from that run
    historical_key = f"{latest_run}/historical_data.parquet"
    try:
        raw_bytes = s3.read_bytes(historical_key)
    except Exception as e:
        logger.error(f"Failed to read {historical_key}: {e}")
        return

    historical_df = pd.read_parquet(io.BytesIO(raw_bytes))
    logger.info(f"Loaded historical data with shape {historical_df.shape}")

    # 3. Run feature pipeline
    feature_pipeline = FeaturePipeline()
    features_df = feature_pipeline.run(historical_df)
    logger.info(f"Generated features with shape {features_df.shape}")

    # 4. Save features back to S3 under 'features/' for that ingestion run
    output_key = f"{latest_run}/features/feature_data.parquet"
    buffer = io.BytesIO()
    features_df.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3.write_bytes(output_key, buffer.read())
    logger.info(f"Saved feature data to s3://{bucket_name}/{output_key}")


if __name__ == "__main__":
    main()
