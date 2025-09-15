# src/data/features/feature_pipeline.py

import logging
import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime
from src.data.features.technical import TechnicalFeaturePipeline
from src.data.features.sentiment import SentimentFeaturePipeline
from src.data.features.market_features import MarketFeaturePipeline
# Import additional feature pipelines here (e.g., On-chain, Macro)

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """
    Main orchestrator for generating all features.
    Combines technical, sentiment, market, on-chain, and other features.
    """
    def __init__(self):
        self.technical_pipeline = TechnicalFeaturePipeline()
        self.sentiment_pipeline = SentimentFeaturePipeline()
        self.market_pipeline = MarketFeaturePipeline()
        # Initialize additional feature pipelines here

    def run(self, historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the feature pipeline.
        
        Args:
            historical_df: DataFrame with historical price and volume data.
        
        Returns:
            DataFrame enriched with all calculated features.
        """
        if historical_df.empty:
            logger.warning("Input DataFrame is empty. Features cannot be generated.")
            return pd.DataFrame()

        # 1. Generate technical features
        logger.info("Generating technical features...")
        technical_df = self.technical_pipeline.generate_features(historical_df)

        # 2. Generate market features
        logger.info("Generating market features...")
        market_df = self.market_pipeline.generate_features(historical_df)

        # 3. Generate sentiment features
        logger.info("Generating sentiment features...")
        sentiment_df = self.sentiment_pipeline.generate_sentiment_features(
            fear_greed_days=90,
            trends_timeframe="today 3-m"
        )

        # 4. Merge dataframes
        combined_df = technical_df.merge(
            market_df,
            on='timestamp',
            how='left'
        )

        final_df = combined_df.merge(
            sentiment_df,
            on='timestamp',
            how='left'
        )

        logger.info(f"Feature pipeline completed. Final DataFrame shape: {final_df.shape}")
        
        return final_df
    def save_to_s3(self, features_df, bucket: str, prefix: str = "features"):
        if features_df.empty:
            logger.warning("Features DataFrame is empty. Nothing to save.")
            return None

        storage = S3Storage(bucket_name=bucket)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        key = f"{prefix}/features_{timestamp}.parquet"
        local_path = f"/tmp/features_{timestamp}.parquet"

        features_df.to_parquet(local_path, index=False)
        storage.upload_file(local_path, key)

        logger.info(f"Features saved to s3://{bucket}/{key}")
        return key


if __name__ == "__main__":
    # Example usage. In a real pipeline, the input DataFrame
    # would come from the preprocessing step (DataProcessor).
    
    # Create a sample DataFrame for testing
    data = {
        'coin_id': ['bitcoin'] * 100,
        'timestamp': pd.to_datetime(pd.date_range(end=datetime.now(), periods=100)),
        'price_usd': np.random.rand(100) * 10000,
        'volume_24h': np.random.rand(100) * 1e9,
        'market_cap': np.random.rand(100) * 1e11
    }
    sample_df = pd.DataFrame(data)
    
    # Run the feature pipeline
    feature_pipeline = FeaturePipeline()
    final_features_df = feature_pipeline.run(sample_df)
    
    if not final_features_df.empty:
        print("\n=== FINAL DATASET STRUCTURE ===")
        print(final_features_df.info())
        print("\n=== FIRST ROWS ===")
        print(final_features_df.head())
