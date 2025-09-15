# src/scripts/show_features.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.data.storage import S3Storage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_features(market: pd.DataFrame, technical: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    df = market.copy()

    # Market features
    df["return"] = df["price_usd"].pct_change()
    df["volatility_7d"] = df["return"].rolling(7).std()

    # Merge technical and sentiment
    df = df.merge(technical, on=["coin_id", "timestamp"], how="left")
    df = df.merge(sentiment, on=["coin_id", "timestamp"], how="left")

    # Fill missing values
    df = df.fillna(method="ffill").dropna()

    # Scale features (excluding identifiers)
    feature_cols = df.columns.drop(["coin_id", "timestamp", "price_usd"])
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df

def main():
    bucket_name = "mlops-terraform-state-ignacio-480415624749"
    storage = S3Storage(bucket_name=bucket_name)

    # Archivos a leer
    files_to_load = {
        "market": "ingestion_1756563643/market_data.parquet",
        "technical": "ingestion_1756563643/technical_data.parquet",
        "sentiment": "ingestion_1756563643/historical_data.parquet"
    }

    dfs = {}
    for key, s3_key in files_to_load.items():
        local_file = f"tmp_{key}.parquet"
        if storage.download_file(s3_key, local_file):
            dfs[key] = pd.read_parquet(local_file)
            logger.info(f"{key} data loaded: {dfs[key].shape}")
        else:
            logger.error(f"Failed to load {key} from S3")
            return

    # Construir features
    df_features = build_features(dfs["market"], dfs["technical"], dfs["sentiment"])
    logger.info(f"Feature engineering completed. Result shape: {df_features.shape}")

    # Mostrar DataFrame final
    print("\n=== Final Features DataFrame ===")
    print(df_features)

if __name__ == "__main__":
    main()
