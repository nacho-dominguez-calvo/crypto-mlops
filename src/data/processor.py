# src/data/processor.py

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

try:
    import pandera as pa
    from pandera import Check, Column, DataFrameSchema
except ImportError:
    pa = None
    logger.warning("Pandera not installed. Validation will be skipped.")

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process raw API data into structured DataFrames and validate with Pandera"""

    # Esquemas de validación con Pandera
    HISTORICAL_SCHEMA = DataFrameSchema({
        "coin_id": Column(str, nullable=False, checks=Check.str_length(min_value=1)),
        "timestamp": Column("datetime64[ns]", nullable=False),
        "price_usd": Column(float, nullable=False, checks=Check.greater_than_or_equal_to(1e-10)),
        "volume_24h": Column(float, nullable=True, checks=Check.greater_than_or_equal_to(0)),
        "market_cap": Column(float, nullable=True, checks=Check.greater_than_or_equal_to(0)),
    })

    CURRENT_PRICES_SCHEMA = DataFrameSchema({
        "coin_id": Column(str, nullable=False),
        "timestamp": Column("datetime64[ns]", nullable=False),
        "price_usd": Column(float, nullable=True),
        "market_cap": Column(float, nullable=True),
        "volume_24h": Column(float, nullable=True),
        "price_change_24h": Column(float, nullable=True),
        "last_updated": Column("datetime64[ns]", nullable=True),
    })

    MARKET_DATA_SCHEMA = DataFrameSchema({
        "coin_id": Column(str, nullable=False),
        "symbol": Column(str, nullable=True),
        "name": Column(str, nullable=True),
        "timestamp": Column("datetime64[ns]", nullable=False),
        "price_usd": Column(float, nullable=True),
        "market_cap": Column(float, nullable=True),
        "market_cap_rank": Column(int, nullable=True),
        "volume_24h": Column(float, nullable=True),
        "price_change_1h": Column(float, nullable=True),
        "price_change_24h": Column(float, nullable=True),
        "price_change_7d": Column(float, nullable=True),
        "circulating_supply": Column(float, nullable=True),
        "total_supply": Column(float, nullable=True),
        "max_supply": Column(float, nullable=True),
        "last_updated": Column("datetime64[ns]", nullable=True),
    })

    @staticmethod
    def process_current_prices(raw_data: Optional[Dict]) -> pd.DataFrame:
        """Convert current price data to DataFrame"""
        processed = []
        timestamp = datetime.now()
        
        if not raw_data:
            return pd.DataFrame()

        for coin_id, data in raw_data.items():
            try:
                last_updated = datetime.fromtimestamp(data['last_updated_at']) if data.get('last_updated_at') else None
            except (KeyError, TypeError, ValueError):
                last_updated = None

            processed.append({
                'coin_id': coin_id,
                'timestamp': timestamp,
                'price_usd': data.get('usd'),
                'market_cap': data.get('usd_market_cap'),
                'volume_24h': data.get('usd_24h_vol'),
                'price_change_24h': data.get('usd_24h_change'),
                'last_updated': last_updated
            })
        
        df = pd.DataFrame(processed)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
        return df

    @staticmethod
    def process_market_data(raw_data: Optional[List[Dict]]) -> pd.DataFrame:
        """Convert market data to DataFrame"""
        processed = []
        timestamp = datetime.now()
        
        if not raw_data:
            return pd.DataFrame()

        for coin in raw_data:
            try:
                last_updated = pd.to_datetime(coin.get('last_updated'), errors='coerce')
            except (ValueError, TypeError):
                last_updated = None

            processed.append({
                'coin_id': coin.get('id'),
                'symbol': coin.get('symbol'),
                'name': coin.get('name'),
                'timestamp': timestamp,
                'price_usd': coin.get('current_price'),
                'market_cap': coin.get('market_cap'),
                'market_cap_rank': coin.get('market_cap_rank'),
                'volume_24h': coin.get('total_volume'),
                'price_change_1h': coin.get('price_change_percentage_1h_in_currency'),
                'price_change_24h': coin.get('price_change_percentage_24h'),
                'price_change_7d': coin.get('price_change_percentage_7d_in_currency'),
                'circulating_supply': coin.get('circulating_supply'),
                'total_supply': coin.get('total_supply'),
                'max_supply': coin.get('max_supply'),
                'last_updated': last_updated
            })
        
        df = pd.DataFrame(processed)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df

    @staticmethod
    def process_historical_data(raw_data: Dict, coin_id: str) -> pd.DataFrame:
        """Convert historical data to DataFrame"""
        prices = raw_data.get('prices', [])
        volumes = raw_data.get('total_volumes', [])
        market_caps = raw_data.get('market_caps', [])

        processed = []
        min_length = min(len(prices), len(volumes), len(market_caps))
        
        for i in range(min_length):
            try:
                ts = datetime.fromtimestamp(prices[i][0] / 1000)
            except (IndexError, ValueError, TypeError):
                ts = None

            processed.append({
                'coin_id': coin_id,
                'timestamp': ts,
                'price_usd': prices[i][1] if len(prices[i]) > 1 else None,
                'volume_24h': volumes[i][1] if len(volumes[i]) > 1 else None,
                'market_cap': market_caps[i][1] if len(market_caps[i]) > 1 else None,
            })

        df = pd.DataFrame(processed)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    @staticmethod
    def validate_historical_data(df: pd.DataFrame) -> bool:
        """Validate historical data with Pandera"""
        try:
            if df is None or df.empty:
                logger.error("Empty DataFrame received for validation")
                return False

            if pa is None:
                logger.warning("Pandera not available, skipping validation")
                return True

            # Validar con Pandera
            DataProcessor.HISTORICAL_SCHEMA.validate(df, lazy=True)
            logger.info("Historical data validation successful")
            return True

        except pa.errors.SchemaErrors as e:
            logger.error(f"Pandera validation failed: {e}")
            # Log detallado de los errores
            for error in e.schema_errors:
                logger.error(f"Validation error: {error}")
            return False
        except Exception as e:
            logger.error(f"Unexpected validation error: {str(e)}")
            return False

    @staticmethod
    def validate_current_prices(df: pd.DataFrame) -> bool:
        """Validate current prices data with Pandera"""
        if pa is None or df is None or df.empty:
            return True
            
        try:
            DataProcessor.CURRENT_PRICES_SCHEMA.validate(df, lazy=True)
            return True
        except pa.errors.SchemaErrors:
            return False

    @staticmethod
    def validate_market_data(df: pd.DataFrame) -> bool:
        """Validate market data with Pandera - versión más robusta"""
        if pa is None or df is None or df.empty:
            return True
            
        try:
            # Primero asegurarnos de que las columnas numéricas sean del tipo correcto
            numeric_columns = ['market_cap_rank', 'price_change_1h', 'price_change_24h', 
                              'price_change_7d', 'circulating_supply', 'total_supply', 'max_supply']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Validar con Pandera
            DataProcessor.MARKET_DATA_SCHEMA.validate(df, lazy=True)
            logger.info("Market data validation successful")
            return True

        except pa.errors.SchemaErrors as e:
            logger.warning(f"Market data validation warnings: {len(e.schema_errors)} issues")
            # Solo loguear errores críticos, no warnings de columnas opcionales
            critical_errors = []
            for error in e.schema_errors:
                if "column" in str(error).lower() and "required" not in str(error).lower():
                    critical_errors.append(error)
            
            if critical_errors:
                logger.error(f"Critical validation errors: {critical_errors}")
                return False
            else:
                logger.info("Market data validation passed with optional field warnings")
                return True
                
        except Exception as e:
            logger.error(f"Unexpected validation error: {str(e)}")
            return False

    def process_ingestion_results(self, ingestion_results: Dict) -> Dict[str, pd.DataFrame]:
        """Process complete ingestion results into validated DataFrames"""
        processed_data = {}
        
        try:
            # Process current data
            current_data = ingestion_results.get('current_data', {})
            if 'prices' in current_data:
                current_df = self.process_current_prices(current_data['prices'])
                if self.validate_current_prices(current_df):
                    processed_data['current_prices'] = current_df
                else:
                    logger.warning("Current prices validation failed")

            if 'markets' in current_data:
                market_df = self.process_market_data(current_data['markets'])
                if self.validate_market_data(market_df):
                    processed_data['market_data'] = market_df
                else:
                    logger.warning("Market data validation failed")

            # Process historical data
            historical_data = ingestion_results.get('historical_data', {})
            if isinstance(historical_data, dict):
                historical_dict = historical_data.get('data', {})
                historical_dfs = []
                
                for coin_id, coin_data in historical_dict.items():
                    try:
                        coin_df = self.process_historical_data(coin_data, coin_id)
                        if not coin_df.empty:
                            historical_dfs.append(coin_df)
                    except Exception as e:
                        logger.error(f"Error processing historical data for {coin_id}: {str(e)}")
                
                if historical_dfs:
                    combined_df = pd.concat(historical_dfs, ignore_index=True)
                    if self.validate_historical_data(combined_df):
                        processed_data['historical_data'] = combined_df
                    else:
                        logger.error("Historical data validation failed")
                        # Opcional: puedes decidir si quieres raise exception o continuar
                        # raise ValueError("Historical data validation failed")
            
            logger.info(f"Successfully processed {len(processed_data)} datasets")
            return processed_data

        except Exception as e:
            logger.error(f"Failed to process ingestion results: {str(e)}")
            raise