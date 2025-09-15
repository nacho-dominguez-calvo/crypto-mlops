# src/data/sources/sentiment.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import logging
import time
from pytrends.request import TrendReq

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FearGreedIndexClient:
    """Client for fetching Fear and Greed Index data from Alternative.me API."""

    def __init__(self):
        self.base_url = "https://api.alternative.me"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'crypto-mlops-pipeline/1.0'
        })

    def get_current_fear_greed(self) -> Dict:
        """Get current Fear and Greed Index value."""
        try:
            response = self.session.get(f"{self.base_url}/fng/")
            response.raise_for_status()
            data = response.json()

            if 'data' in data and len(data['data']) > 0:
                current = data['data'][0]
                logger.info(f"Current Fear & Greed Index: {current['value']} ({current['value_classification']})")
                return current
            else:
                raise ValueError("No current data available")
        except Exception as e:
            logger.error(f"Failed to fetch current Fear & Greed Index: {e}")
            raise

    def get_historical_fear_greed(self, days: int = 30) -> List[Dict]:
        """Get historical Fear and Greed Index data."""
        try:
            days = min(days, 200)
            params = {'limit': days, 'format': 'json'}
            response = self.session.get(f"{self.base_url}/fng/", params=params)
            response.raise_for_status()

            data = response.json()
            if 'data' in data:
                logger.info(f"Fetched {len(data['data'])} days of Fear & Greed Index data")
                return data['data']
            else:
                raise ValueError("No historical data available")
        except Exception as e:
            logger.error(f"Failed to fetch historical Fear & Greed Index: {e}")
            raise

    def process_fear_greed_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """Convert raw Fear & Greed data to DataFrame."""
        processed = []
        for entry in raw_data:
            processed.append({
                'timestamp': pd.to_datetime(entry['timestamp'], unit='s'),
                'fear_greed_value': int(entry['value']),
                'fear_greed_classification': entry['value_classification'],
                'fear_greed_time_until_update': entry.get('time_until_update', None)
            })

        df = pd.DataFrame(processed).sort_values('timestamp')
        df['fear_greed_normalized'] = df['fear_greed_value'] / 100.0

        classification_map = {
            'Extreme Fear': 0,
            'Fear': 1,
            'Neutral': 2,
            'Greed': 3,
            'Extreme Greed': 4
        }
        df['fear_greed_category'] = df['fear_greed_classification'].map(classification_map)

        df['fear_greed_sma_7'] = df['fear_greed_value'].rolling(window=7, min_periods=1).mean()
        df['fear_greed_sma_14'] = df['fear_greed_value'].rolling(window=14, min_periods=1).mean()
        df['fear_greed_change_1d'] = df['fear_greed_value'].diff(1)
        df['fear_greed_change_7d'] = df['fear_greed_value'].diff(7)

        return df


class GoogleTrendsClient:
    """Client for fetching Google Trends data using pytrends."""

    def __init__(self, hl='en-US', tz=360):
        self.pytrends = TrendReq(hl=hl, tz=tz, timeout=(10, 25))
        self.rate_limit_delay = 2.0
        self._last_request_time = 0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def get_trends_data(self, keywords: List[str], timeframe: str = 'today 3-m', geo: str = '') -> pd.DataFrame:
        """Get Google Trends data for specified keywords."""
        try:
            self._rate_limit()
            keywords = keywords[:5]

            logger.info(f"Fetching Google Trends for: {keywords}")
            self.pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
            trends_data = self.pytrends.interest_over_time()

            if trends_data.empty:
                logger.warning(f"No trends data returned for keywords: {keywords}")
                return pd.DataFrame()

            if 'isPartial' in trends_data.columns:
                trends_data = trends_data.drop('isPartial', axis=1)

            return trends_data.reset_index().rename(columns={'date': 'timestamp'})
        except Exception as e:
            logger.error(f"Failed to fetch Google Trends data: {e}")
            return pd.DataFrame()

    def get_crypto_trends(self, timeframe: str = 'today 3-m') -> pd.DataFrame:
        """Get predefined crypto-related trends."""
        top5_keywords = ['Bitcoin', 'Ethereum', 'Tether', 'Binance Coin', 'Solana']
        generic_keywords = ['cryptocurrency', 'crypto', 'blockchain', 'DeFi', 'NFT']

        top5_df = self.get_trends_data(top5_keywords, timeframe)
        generic_df = self.get_trends_data(generic_keywords, timeframe)

        if not top5_df.empty and not generic_df.empty:
            return pd.merge(top5_df, generic_df, on='timestamp', how='outer', suffixes=('', '_generic'))
        return top5_df if not top5_df.empty else generic_df

    def get_related_queries(self, keyword: str) -> Dict:
        """Get related queries for a keyword."""
        try:
            self._rate_limit()
            self.pytrends.build_payload([keyword])
            return self.pytrends.related_queries().get(keyword, {})
        except Exception as e:
            logger.error(f"Failed to fetch related queries for {keyword}: {e}")
            return {}


class SentimentFeaturePipeline:
    """Main pipeline for generating sentiment-based features from multiple sources."""

    def __init__(self):
        self.fear_greed_client = FearGreedIndexClient()
        self.trends_client = GoogleTrendsClient()

    def fetch_fear_greed_features(self, days: int = 90) -> pd.DataFrame:
        try:
            raw_data = self.fear_greed_client.get_historical_fear_greed(days)
            return self.fear_greed_client.process_fear_greed_data(raw_data) if raw_data else pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch Fear & Greed features: {e}")
            return pd.DataFrame()

    def fetch_trends_features(self, timeframe: str = 'today 3-m') -> pd.DataFrame:
        try:
            trends_df = self.trends_client.get_crypto_trends(timeframe)
            return self._process_trends_data(trends_df) if not trends_df.empty else pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch Google Trends features: {e}")
            return pd.DataFrame()

    def _process_trends_data(self, trends_df: pd.DataFrame) -> pd.DataFrame:
        if trends_df.empty:
            return pd.DataFrame()

        df = trends_df.copy()
        column_mapping = {col: f"trends_{col.lower().replace(' ', '_')}" for col in df.columns if col != 'timestamp'}
        df = df.rename(columns=column_mapping)

        trend_cols = [col for col in df.columns if col.startswith('trends_')]
        for col in trend_cols:
            base = col.replace('trends_', '')
            df[f'trends_{base}_normalized'] = df[col] / 100.0
            df[f'trends_{base}_sma_7'] = df[col].rolling(window=7, min_periods=1).mean()
            df[f'trends_{base}_sma_30'] = df[col].rolling(window=30, min_periods=1).mean()
            df[f'trends_{base}_change_7d'] = df[col].diff(7)
            df[f'trends_{base}_change_30d'] = df[col].diff(30)
            df[f'trends_{base}_volatility_7d'] = df[col].rolling(window=7).std()

        return df

    def generate_sentiment_features(self, fear_greed_days: int = 90, trends_timeframe: str = 'today 3-m') -> pd.DataFrame:
        fear_greed_df = self.fetch_fear_greed_features(fear_greed_days)
        trends_df = self.fetch_trends_features(trends_timeframe)
        all_features = [df for df in [fear_greed_df, trends_df] if not df.empty]

        if not all_features:
            return pd.DataFrame()

        result_df = all_features[0]
        for df in all_features[1:]:
            result_df = pd.merge(result_df, df, on='timestamp', how='outer', suffixes=('', '_dup'))
            result_df = result_df.drop(columns=[c for c in result_df.columns if c.endswith('_dup')])

        result_df = result_df.sort_values('timestamp')
        return self._add_sentiment_interactions(result_df)

    def _add_sentiment_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if 'fear_greed_normalized' in df.columns and 'trends_bitcoin' in df.columns:
            df['sentiment_fg_x_bitcoin'] = df['fear_greed_normalized'] * df['trends_bitcoin']
        if 'fear_greed_category' in df.columns and 'trends_ethereum' in df.columns:
            df['sentiment_fgcat_x_eth'] = df['fear_greed_category'] * df['trends_ethereum']
        return df


if __name__ == "__main__":
    pipeline = SentimentFeaturePipeline()
    features_df = pipeline.generate_sentiment_features(fear_greed_days=90, trends_timeframe="today 3-m")

    if not features_df.empty:
        print(features_df.head())
        output_path = f"data/processed/sentiment_features_{datetime.now().strftime('%Y%m%d')}.parquet"
        features_df.to_parquet(output_path, index=False)
        logger.info(f"Sentiment features saved to {output_path}")
    else:
        logger.warning("No sentiment features generated, nothing was saved.")
