# src/scripts/run_ingestion.py
from src.data import DataIngestionPipeline, DataProcessor
from src.data.storage import S3Storage
import pandas as pd

def main():
    # ----------------------------
    # 1️⃣ Initialize ingestion
    # ----------------------------
    pipeline = DataIngestionPipeline(
        coins=['bitcoin', 'ethereum']  # monedas a probar
    )

    # ----------------------------
    # 2️⃣ Initialize processor
    # ----------------------------
    processor = DataProcessor()

    # ----------------------------
    # 3️⃣ Initialize S3 storage
    # ----------------------------
    s3_bucket = "mlops-terraform-state-ignacio-480415624749"
    storage = S3Storage(bucket_name=s3_bucket)

    try:
        # ----------------------------
        # 4️⃣ Run ingestion
        # ----------------------------
        raw_results = pipeline.run_ingestion(include_historical=True, historical_days=7)

        # ----------------------------
        # 5️⃣ Process results
        # ----------------------------
        processed_data = processor.process_ingestion_results(raw_results)

        # ----------------------------
        # 6️⃣ Print summary
        # ----------------------------
        print("\n=== INGESTION RESULTS ===")
        print(f"Pipeline ID: {raw_results['pipeline_run_id']}")
        print(f"Status: {raw_results['status']}")
        print(f"Coins: {raw_results['coins']}")

        print("\n=== PROCESSED DATA ===")
        for key, df in processed_data.items():
            print(f"\n{key.upper()}:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Sample:\n{df.head(2)}")

        # ----------------------------
        # 7️⃣ Save processed data to S3
        # ----------------------------
        for key, df in processed_data.items():
            s3_key = f"{raw_results['pipeline_run_id']}/{key}.parquet"
            # Convert DataFrame to bytes and save
            csv_bytes = df.to_parquet(index=False)
            storage.save_bytes(csv_bytes, s3_key, content_type="application/octet-stream")

        print("\n✅ All processed data saved to S3 successfully!")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")


if __name__ == "__main__":
    main()
