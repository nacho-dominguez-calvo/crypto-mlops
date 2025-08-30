import requests
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CoinGeckoConfig:
    """Configuration for CoinGecko API client"""
    api_key: Optional[str] = None
    base_url: str = "https://api.coingecko.com/api/v3"
    rate_limit_delay: float = 6.0  # seconds between requests (free tier: 10-50 calls/min)
    max_retries: int = 3
    timeout: int = 30

class CoinGeckoClient:
    """Client for CoinGecko API (free tier, no API key needed)"""
    
    def __init__(self, config: Optional[CoinGeckoConfig] = None):
        self.config = config or CoinGeckoConfig()
        self.session = requests.Session()
        self._last_request_time = 0
        
    def _get_headers(self) -> Dict[str, str]:
        return {'accept': 'application/json', 'User-Agent': 'crypto-ml-pipeline/1.0'}
    
    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - elapsed
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        url = f"{self.config.base_url}/{endpoint}"
        for attempt in range(self.config.max_retries):
            try:
                self._rate_limit()
                response = self.session.get(url, params=params, headers=self._get_headers(), timeout=self.config.timeout)
                
                if response.status_code == 429:
                    wait_time = 60 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        
        raise Exception("Max retries exceeded")
    
    def get_current_prices(self, coin_ids: List[str]) -> Dict:
        params = {
            'ids': ','.join(coin_ids),
            'vs_currencies': 'usd',
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true',
            'include_last_updated_at': 'true'
        }
        data = self._make_request('simple/price', params)
        logger.info(f"Fetched current prices for {len(data)} coins")
        return data
    
    def get_market_data(self, coin_ids: List[str], vs_currency: str = 'usd') -> List[Dict]:
        params = {
            'ids': ','.join(coin_ids),
            'vs_currency': vs_currency,
            'order': 'market_cap_desc',
            'per_page': len(coin_ids),
            'page': 1,
            'sparkline': 'false',
            'price_change_percentage': '1h,24h,7d'
        }
        data = self._make_request('coins/markets', params)
        logger.info(f"Fetched market data for {len(data)} coins")
        return data
    
    def get_historical_data(self, coin_id: str, days: int = 30) -> Dict:
        """Fetch historical price data using market_chart/range (avoids 401)"""
        # Timestamps Unix para rango
        from_time = int((datetime.now() - timedelta(days=days)).timestamp())
        to_time = int(datetime.now().timestamp())
        
        endpoint = f"coins/{coin_id}/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': from_time,
            'to': to_time
        }
        
        data = self._make_request(endpoint, params)
        logger.info(f"Fetched {days} days of historical data for {coin_id} using range endpoint")
        return data


class DataIngestionPipeline:
    """Main pipeline for data ingestion"""
    
    def __init__(self, coins: List[str] = None):
        self.client = CoinGeckoClient()
        self.coins = coins or ['bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana']
        
    def validate_config(self):
        """Validate configuration before running pipeline"""
        if not os.getenv('COINGECKO_API_KEY'):
            logger.warning("No API key found. Using free tier limits.")
        
        logger.info(f"Pipeline configured for coins: {', '.join(self.coins)}")
        return True
    
    def fetch_current_data(self) -> Dict:
        """Fetch current market data"""
        try:
            # Fetch current prices
            price_data = self.client.get_current_prices(self.coins)
            
            # Fetch detailed market data
            market_data = self.client.get_market_data(self.coins)
            
            logger.info(f"Successfully fetched current data for {len(self.coins)} coins")
            return {
                'prices': price_data,
                'markets': market_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch current data: {e}")
            raise
    
    def fetch_historical_data(self, days: int = 30) -> Dict:
        """Fetch historical data for all coins"""
        historical_data = {}
        
        for coin_id in self.coins:
            try:
                raw_data = self.client.get_historical_data(coin_id, days)
                historical_data[coin_id] = raw_data
                
            except Exception as e:
                logger.error(f"Failed to fetch historical data for {coin_id}: {e}")
                continue
        
        if historical_data:
            logger.info(f"Successfully fetched {days} days of historical data")
            return {
                'data': historical_data,
                'days': days,
                'timestamp': datetime.now().isoformat()
            }
        else:
            raise Exception("No historical data could be fetched")
    
    def run_ingestion(self, include_historical: bool = True, historical_days: int = 90) -> Dict:
        """Run complete data ingestion pipeline"""
        logger.info("Starting data ingestion pipeline")
        
        # Validate configuration
        self.validate_config()
        
        results = {
            'pipeline_run_id': f"ingestion_{int(datetime.now().timestamp())}",
            'coins': self.coins,
            'status': 'running'
        }
        
        try:
            # Fetch current data
            logger.info("Fetching current market data...")
            results['current_data'] = self.fetch_current_data()
            
            # Fetch historical data if requested
            if include_historical:
                logger.info(f"Fetching {historical_days} days of historical data...")
                results['historical_data'] = self.fetch_historical_data(historical_days)
            
            results['status'] = 'completed'
            results['completed_at'] = datetime.now().isoformat()
            
            logger.info("Data ingestion completed successfully")
            return results
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            results['failed_at'] = datetime.now().isoformat()
            logger.error(f"Data ingestion failed: {e}")
            raise