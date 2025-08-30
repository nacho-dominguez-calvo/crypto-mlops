# src/data/sources/sentiment.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
import time
from pytrends.request import TrendReq

logger = logging.getLogger(__name__)

class FearGreedIndexClient:
    """
    Client for fetching Fear and Greed Index data from Alternative.me API.
    This is a completely free API that provides crypto market sentiment data.
    """
    
    def __init__(self):
        self.base_url = "https://api.alternative.me"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'crypto-mlops-pipeline/1.0'
        })
    
    def get_current_fear_greed(self) -> Dict:
        """Get current Fear and Greed Index value"""
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
        """Get historical Fear and Greed Index data"""
        try:
            # API supports up to 200 days of historical data
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
        """Convert raw Fear & Greed data to DataFrame"""
        processed = []
        
        for entry in raw_data:
            processed.append({
                'timestamp': pd.to_datetime(entry['timestamp'], unit='s'),
                'fear_greed_value': int(entry['value']),
                'fear_greed_classification': entry['value_classification'],
                'fear_greed_time_until_update': entry.get('time_until_update', None)
            })
        
        df = pd.DataFrame(processed)
        df = df.sort_values('timestamp')
        
        # Add derived features
        df['fear_greed_normalized'] = df['fear_greed_value'] / 100.0  # Normalize to 0-1
        
        # Categorical encoding for classification
        classification_map = {
            'Extreme Fear': 0,
            'Fear': 1, 
            'Neutral': 2,
            'Greed': 3,
            'Extreme Greed': 4
        }
        df['fear_greed_category'] = df['fear_greed_classification'].map(classification_map)
        
        # Rolling averages for smoothing
        df['fear_greed_sma_7'] = df['fear_greed_value'].rolling(window=7, min_periods=1).mean()
        df['fear_greed_sma_14'] = df['fear_greed_value'].rolling(window=14, min_periods=1).mean()
        
        # Momentum features
        df['fear_greed_change_1d'] = df['fear_greed_value'].diff(1)
        df['fear_greed_change_7d'] = df['fear_greed_value'].diff(7)
        
        return df


class GoogleTrendsClient:
    """
    Client for fetching Google Trends data using pytrends library.
    This provides insights into public interest in cryptocurrency terms.
    """
    
    def __init__(self, hl='en-US', tz=360):
        self.pytrends = TrendReq(hl=hl, tz=tz, timeout=(10,25))
        self.rate_limit_delay = 2.0  # Seconds between requests to avoid rate limiting
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Apply rate limiting to avoid being blocked by Google"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    def get_trends_data(self, keywords: List[str], 
                       timeframe: str = 'today 3-m',
                       geo: str = '') -> pd.DataFrame:
        """
        Get Google Trends data for specified keywords.
        
        Args:
            keywords: List of search terms (max 5)
            timeframe: Time range (e.g., 'today 3-m', 'today 12-m', 'today 5-y')
            geo: Geographic location ('' for worldwide, 'US' for United States, etc.)
        """
        try:
            self._rate_limit()
            
            # Limit to 5 keywords max (Google Trends limitation)
            keywords = keywords[:5]
            
            logger.info(f"Fetching Google Trends for: {keywords}")
            logger.debug(f"Timeframe: {timeframe}, Geography: {geo or 'Worldwide'}")
            
            self.pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
            
            # Get interest over time
            trends_data = self.pytrends.interest_over_time()
            
            if trends_data.empty:
                logger.warning(f"No trends data returned for keywords: {keywords}")
                return pd.DataFrame()
            
            # Remove 'isPartial' column if it exists
            if 'isPartial' in trends_data.columns:
                trends_data = trends_data.drop('isPartial', axis=1)
            
            # Reset index to make date a column
            trends_data = trends_data.reset_index()
            trends_data.rename(columns={'date': 'timestamp'}, inplace=True)
            
            logger.info(f"Retrieved {len(trends_data)} data points for trends")
            return trends_data
            
        except Exception as e:
            logger.error(f"Failed to fetch Google Trends data: {e}")
            return pd.DataFrame()
    
    def get_crypto_trends(self, timeframe: str = 'today 3-m') -> pd.DataFrame:
        """Get predefined cryptocurrency-related trends"""
        crypto_keywords = ['Bitcoin', 'Ethereum', 'Cryptocurrency', 'Crypto', 'BTC']
        return self.get_trends_data(crypto_keywords, timeframe)
    
    def get_related_queries(self, keyword: str) -> Dict:
        """Get related queries for a keyword (useful for feature expansion)"""
        try:
            self._rate_limit()
            
            self.pytrends.build_payload([keyword])
            related_queries = self.pytrends.related_queries()
            
            return related_queries.get(keyword, {})
            
        except Exception as e:
            logger.error(f"Failed to fetch related queries for {keyword}: {e}")
            return {}


class SentimentFeaturePipeline:
    """
    Main pipeline for generating sentiment-based features from multiple sources.
    """
    
    def __init__(self):
        self.fear_greed_client = FearGreedIndexClient()
        self.trends_client = GoogleTrendsClient()
    
    def fetch_fear_greed_features(self, days: int = 90) -> pd.DataFrame:
        """Fetch and process Fear & Greed Index features"""
        try:
            logger.info("Fetching Fear & Greed Index data...")
            raw_data = self.fear_greed_client.get_historical_fear_greed(days)
            
            if not raw_data:
                logger.warning("No Fear & Greed data available")
                return pd.DataFrame()
            
            df = self.fear_greed_client.process_fear_greed_data(raw_data)
            logger.info(f"Processed Fear & Greed features: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch Fear & Greed features: {e}")
            return pd.DataFrame()
    
    def fetch_trends_features(self, timeframe: str = 'today 3-m') -> pd.DataFrame:
        """Fetch and process Google Trends features"""
        try:
            logger.info("Fetching Google Trends data...")
            
            # Get crypto-related trends
            trends_df = self.trends_client.get_crypto_trends(timeframe)
            
            if trends_df.empty:
                logger.warning("No Google Trends data available")
                return pd.DataFrame()
            
            # Process trends data
            processed_df = self._process_trends_data(trends_df)
            logger.info(f"Processed Google Trends features: {processed_df.shape}")
            return processed_df
            
        except Exception as e:
            logger.error(f"Failed to fetch Google Trends features: {e}")
            return pd.DataFrame()
    
    def _process_trends_data(self, trends_df: pd.DataFrame) -> pd.DataFrame:
        """Process raw Google Trends data into features"""
        if trends_df.empty:
            return pd.DataFrame()
        
        df = trends_df.copy()
        
        # Rename columns to be more descriptive
        column_mapping = {}
        for col in df.columns:
            if col != 'timestamp':
                new_col = f'trends_{col.lower().replace(" ", "_")}'
                column_mapping[col] = new_col
        
        df = df.rename(columns=column_mapping)
        
        # Add derived features for each trend
        trend_cols = [col for col in df.columns if col.startswith('trends_')]
        
        for col in trend_cols:
            base_name = col.replace('trends_', '')
            
            # Normalized values (0-1)
            df[f'trends_{base_name}_normalized'] = df[col] / 100.0
            
            # Rolling averages
            df[f'trends_{base_name}_sma_7'] = df[col].rolling(window=7, min_periods=1).mean()
            df[f'trends_{base_name}_sma_30'] = df[col].rolling(window=30, min_periods=1).mean()
            
            # Momentum features  
            df[f'trends_{base_name}_change_7d'] = df[col].diff(7)
            df[f'trends_{base_name}_change_30d'] = df[col].diff(30)
            
            # Volatility
            df[f'trends_{base_name}_volatility_7d'] = df[col].rolling(window=7).std()
        
        return df
    
    def generate_sentiment_features(self, fear_greed_days: int = 90,
                                   trends_timeframe: str = 'today 3-m') -> pd.DataFrame:
        """
        Generate comprehensive sentiment features from all sources.
        
        Args:
            fear_greed_days: Number of days of Fear & Greed data
            trends_timeframe: Timeframe for Google Trends data
        
        Returns:
            DataFrame with all sentiment features
        """
        
        logger.info("Starting sentiment feature generation...")
        
        all_features = []
        
        # Fetch Fear & Greed features
        fear_greed_df = self.fetch_fear_greed_features(fear_greed_days)
        if not fear_greed_df.empty:
            all_features.append(fear_greed_df)
        
        # Fetch Google Trends features  
        trends_df = self.fetch_trends_features(trends_timeframe)
        if not trends_df.empty:
            all_features.append(trends_df)
        
        if not all_features:
            logger.warning("No sentiment features could be generated")
            return pd.DataFrame()
        
        # Merge all features on timestamp if multiple sources available
        if len(all_features) == 1:
            result_df = all_features[0]
        else:
            result_df = all_features[0]
            for df in all_features[1:]:
                result_df = pd.merge(
                    result_df, df, 
                    on='timestamp', 
                    how='outer', 
                    suffixes=('', '_dup')
                )
                
                # Remove duplicate columns
                dup_cols = [col for col in result_df.columns if col.endswith('_dup')]
                result_df = result_df.drop(columns=dup_cols)
        
        # Sort by timestamp
        result_df = result_df.sort_values('timestamp')
        
        # Add cross-feature interactions
        result_df = self._add_sentiment_interactions(result_df)
        
        sentiment_features = [col for col in result_df.columns if 
                            col.startswith(('fear_greed', 'trends_', 'sentiment_'))]
        
        logger.info(f"Generated {len(sentiment_features)} sentiment features")
        logger.debug(f"Features: {sentiment_features[:10]}...")
    def _add_sentiment_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between Fear & Greed and Google Trends.
        """
        if df.empty:
            return df

        df = df.copy()

        # Example: interaction between normalized fear/greed and Bitcoin trend
        if 'fear_greed_normalized' in df.columns and 'trends_bitcoin' in df.columns:
            df['sentiment_fg_x_bitcoin'] = df['fear_greed_normalized'] * df['trends_bitcoin']

        # Example: interaction between fear/greed category and Ethereum trend
        if 'fear_greed_category' in df.columns and 'trends_ethereum' in df.columns:
            df['sentiment_fgcat_x_eth'] = df['fear_greed_category'] * df['trends_ethereum']

        return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

    pipeline = SentimentFeaturePipeline()

    # Fetch sentiment features
    features_df = pipeline.generate_sentiment_features(
        fear_greed_days=90,
        trends_timeframe="today 3-m"
    )

    if not features_df.empty:
        output_path = f"data/processed/sentiment_features_{datetime.now().strftime('%Y%m%d')}.parquet"
        features_df.to_parquet(output_path, index=False)
        logger.info(f"Sentiment features saved to {output_path}")
    else:
        logger.warning("No sentiment features generated, nothing was saved.")
