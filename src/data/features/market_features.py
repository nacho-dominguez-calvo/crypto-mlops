# src/data/features/market_features.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketFeatureConfig:
    """Configuration for market features calculation."""
    dom_lookback: int = 14  # Days for dominance moving average
    volatility_window: int = 30 # Days for market volatility calculation

class MarketFeaturePipeline:
    """
    Generates market-level features from processed historical data.
    """
    def __init__(self, config: Optional[MarketFeatureConfig] = None):
        self.config = config or MarketFeatureConfig()

    def generate_features(self, historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates market-level features.

        Args:
            historical_df: A DataFrame containing historical data for multiple coins,
                           including 'coin_id' and 'market_cap'.

        Returns:
            A DataFrame with timestamp and calculated market features.
        """
        if historical_df.empty or 'market_cap' not in historical_df.columns:
            logger.warning("Input DataFrame is empty or missing 'market_cap'. Cannot generate market features.")
            return pd.DataFrame()

        logger.info("Generating market features...")

        # Ensure correct data types
        df = historical_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['coin_id', 'timestamp'])

        # 1. Calculate Market Dominance
        # Calculate total market cap for each day
        market_cap_total = df.groupby('timestamp')['market_cap'].sum().reset_index()
        market_cap_total.rename(columns={'market_cap': 'total_market_cap'}, inplace=True)
        
        # Calculate individual coin dominance
        df = df.merge(market_cap_total, on='timestamp', how='left')
        df['dominance_ratio'] = df['market_cap'] / df['total_market_cap']

        # 2. Calculate Market Volatility (using daily returns of total market)
        market_cap_total['market_returns'] = market_cap_total['total_market_cap'].pct_change()
        
        # Calculate standard deviation of returns (volatility) over a rolling window
        market_cap_total['market_volatility'] = market_cap_total['market_returns'].rolling(
            window=self.config.volatility_window
        ).std()

        # 3. Pivot the dominance data to have coins as columns
        dom_pivot_df = df.pivot(
            index='timestamp',
            columns='coin_id',
            values='dominance_ratio'
        ).add_prefix('market_dom_')
        
        # Reset index to merge
        dom_pivot_df.reset_index(inplace=True)
        
        # 4. Combine all features into a single DataFrame
        final_df = market_cap_total[['timestamp', 'market_volatility']].merge(
            dom_pivot_df,
            on='timestamp',
            how='left'
        )

        logger.info(f"Market features generated. Final DataFrame shape: {final_df.shape}")
        
        return final_df

if __name__ == '__main__':
    # Este es un ejemplo de cómo se usaría el pipeline
    # En un pipeline real, historical_df vendría del paso de procesamiento.
    
    # Crea un DataFrame de ejemplo con múltiples monedas
    coins_data = []
    
    # Datos para Bitcoin
    dates = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=100))
    btc_mc = np.random.rand(100) * 1e11
    eth_mc = np.random.rand(100) * 5e10
    
    btc_df = pd.DataFrame({'coin_id': 'bitcoin', 'timestamp': dates, 'market_cap': btc_mc})
    eth_df = pd.DataFrame({'coin_id': 'ethereum', 'timestamp': dates, 'market_cap': eth_mc})
    
    historical_df_example = pd.concat([btc_df, eth_df], ignore_index=True)

    # Inicializa y ejecuta el pipeline de características de mercado
    market_pipeline = MarketFeaturePipeline()
    market_features_df = market_pipeline.generate_features(historical_df_example)
    
    if not market_features_df.empty:
        print("\nDataFrame de características de mercado:")
        print(market_features_df.head())
        print(f"Columnas: {list(market_features_df.columns)}")
        print(f"Filas: {len(market_features_df)}")
    else:
        print("No se pudieron generar las características de mercado.")