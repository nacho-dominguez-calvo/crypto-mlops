# src/data/features/technical.py
import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Generate technical analysis indicators for cryptocurrency price data.
    Uses the 'ta' library for robust indicator calculations.
    """
    
    def __init__(self):
        self.indicators = {}
    
    def calculate_sma(self, data: pd.Series, windows: List[int] = [7, 14, 30, 50]) -> Dict[str, pd.Series]:
        """Calculate Simple Moving Averages"""
        sma_indicators = {}
        for window in windows:
            sma_indicators[f'sma_{window}'] = data.rolling(window=window).mean()
        return sma_indicators
    
    def calculate_ema(self, data: pd.Series, windows: List[int] = [12, 26, 50]) -> Dict[str, pd.Series]:
        """Calculate Exponential Moving Averages"""
        ema_indicators = {}
        for window in windows:
            ema_indicators[f'ema_{window}'] = data.ewm(span=window).mean()
        return ema_indicators
    
    def calculate_rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        return ta.momentum.RSIIndicator(close=data, window=window).rsi()
    
    def calculate_macd(self, data: pd.Series, window_fast: int = 12, 
                      window_slow: int = 26, window_sign: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicators"""
        macd_indicator = ta.trend.MACD(
            close=data, 
            window_fast=window_fast, 
            window_slow=window_slow, 
            window_sign=window_sign
        )
        
        return {
            'macd': macd_indicator.macd(),
            'macd_signal': macd_indicator.macd_signal(),
            'macd_histogram': macd_indicator.macd_diff()
        }
    
    def calculate_bollinger_bands(self, data: pd.Series, window: int = 20, 
                                 window_dev: int = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        bollinger = ta.volatility.BollingerBands(
            close=data, 
            window=window, 
            window_dev=window_dev
        )
        
        return {
            'bb_upper': bollinger.bollinger_hband(),
            'bb_middle': bollinger.bollinger_mavg(),
            'bb_lower': bollinger.bollinger_lband(),
            'bb_width': bollinger.bollinger_wband(),
            'bb_position': bollinger.bollinger_pband()
        }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, 
                      close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average True Range (volatility measure)"""
        return ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=window
        ).average_true_range()
    
    def calculate_volume_indicators(self, close: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """Calculate volume-based indicators"""
        volume_indicators = {}
        
        # Volume SMA for comparison
        volume_indicators['volume_sma_20'] = volume.rolling(window=20).mean()
        volume_indicators['volume_ratio'] = volume / volume_indicators['volume_sma_20']
        
        # On Balance Volume
        volume_indicators['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=close, volume=volume
        ).on_balance_volume()
        
        # Volume Price Trend
        volume_indicators['vpt'] = ta.volume.VolumePriceTrendIndicator(
            close=close, volume=volume
        ).volume_price_trend()
        
        return volume_indicators
    
    def calculate_momentum_indicators(self, close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate momentum indicators"""
        momentum_indicators = {}
        
        # Rate of Change
        momentum_indicators['roc_10'] = ta.momentum.ROCIndicator(
            close=close, window=10
        ).roc()
        
        # Williams %R
        momentum_indicators['williams_r'] = ta.momentum.WilliamsRIndicator(
            high=close.rolling(14).max(), 
            low=close.rolling(14).min(), 
            close=close
        ).williams_r()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=close.rolling(14).max(),
            low=close.rolling(14).min(),
            close=close
        )
        momentum_indicators['stoch_k'] = stoch.stoch()
        momentum_indicators['stoch_d'] = stoch.stoch_signal()
        
        return momentum_indicators
    
    def calculate_volatility_features(self, close: pd.Series, 
                                     windows: List[int] = [7, 14, 30]) -> Dict[str, pd.Series]:
        """Calculate various volatility measures"""
        volatility_features = {}
        
        # Calculate returns first
        returns = close.pct_change()
        
        for window in windows:
            # Rolling standard deviation of returns
            volatility_features[f'volatility_{window}d'] = returns.rolling(window=window).std()
            
            # Rolling variance
            volatility_features[f'variance_{window}d'] = returns.rolling(window=window).var()
            
            # Price range volatility (high-low relative to close)
            price_range = (close.rolling(window).max() - close.rolling(window).min()) / close
            volatility_features[f'price_range_{window}d'] = price_range
        
        return volatility_features
    
    def calculate_price_features(self, close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate price-based features"""
        price_features = {}
        
        # Returns
        price_features['returns_1d'] = close.pct_change(1)
        price_features['returns_7d'] = close.pct_change(7)
        price_features['returns_30d'] = close.pct_change(30)
        
        # Log returns
        price_features['log_returns_1d'] = np.log(close / close.shift(1))
        
        # Cumulative returns
        price_features['cumulative_returns_7d'] = (1 + price_features['returns_1d']).rolling(7).apply(lambda x: x.prod() - 1)
        price_features['cumulative_returns_30d'] = (1 + price_features['returns_1d']).rolling(30).apply(lambda x: x.prod() - 1)
        
        # Price momentum
        price_features['momentum_3d'] = close - close.shift(3)
        price_features['momentum_7d'] = close - close.shift(7)
        price_features['momentum_14d'] = close - close.shift(14)
        
        # Price position relative to recent highs/lows
        price_features['high_20d'] = close.rolling(20).max()
        price_features['low_20d'] = close.rolling(20).min()
        price_features['price_position_20d'] = (close - price_features['low_20d']) / (price_features['high_20d'] - price_features['low_20d'])
        
        return price_features


class TechnicalFeaturePipeline:
    """
    Main pipeline for generating all technical features for cryptocurrency data.
    """
    
    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
    
    def generate_features(self, df: pd.DataFrame, 
                         price_col: str = 'price_usd',
                         volume_col: str = 'volume_24h',
                         high_col: Optional[str] = None,
                         low_col: Optional[str] = None) -> pd.DataFrame:
        """
        Generate all technical features for a cryptocurrency dataset.
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for close price
            volume_col: Column name for volume
            high_col: Column name for high price (optional)
            low_col: Column name for low price (optional)
        
        Returns:
            DataFrame with original data + technical features
        """
        
        if df.empty:
            logger.warning("Empty DataFrame provided to technical feature pipeline")
            return df
        
        logger.info(f"Generating technical features for {len(df)} rows")
        
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Ensure data is sorted by timestamp
        if 'timestamp' in result_df.columns:
            result_df = result_df.sort_values('timestamp')
        
        close_price = result_df[price_col]
        volume = result_df[volume_col] if volume_col in result_df.columns else None
        
        # Use price as high/low if OHLC data not available
        high_price = result_df[high_col] if high_col and high_col in result_df.columns else close_price
        low_price = result_df[low_col] if low_col and low_col in result_df.columns else close_price
        
        all_features = {}
        
        try:
            # 1. Moving Averages
            logger.debug("Calculating moving averages...")
            all_features.update(self.technical_indicators.calculate_sma(close_price))
            all_features.update(self.technical_indicators.calculate_ema(close_price))
            
            # 2. Oscillators
            logger.debug("Calculating oscillators...")
            all_features['rsi_14'] = self.technical_indicators.calculate_rsi(close_price)
            
            # 3. MACD
            logger.debug("Calculating MACD...")
            all_features.update(self.technical_indicators.calculate_macd(close_price))
            
            # 4. Bollinger Bands
            logger.debug("Calculating Bollinger Bands...")
            all_features.update(self.technical_indicators.calculate_bollinger_bands(close_price))
            
            # 5. Volatility measures
            logger.debug("Calculating volatility features...")
            if len(close_price) > 1:  # Need at least 2 points for volatility
                all_features.update(self.technical_indicators.calculate_volatility_features(close_price))
            
            # 6. Price features
            logger.debug("Calculating price features...")
            all_features.update(self.technical_indicators.calculate_price_features(close_price))
            
            # 7. ATR (if high/low available)
            if high_col and low_col:
                logger.debug("Calculating ATR...")
                all_features['atr_14'] = self.technical_indicators.calculate_atr(
                    high_price, low_price, close_price
                )
            
            # 8. Volume indicators (if volume available)
            if volume is not None and not volume.empty:
                logger.debug("Calculating volume indicators...")
                all_features.update(
                    self.technical_indicators.calculate_volume_indicators(close_price, volume)
                )
            
            # 9. Momentum indicators
            logger.debug("Calculating momentum indicators...")
            all_features.update(
                self.technical_indicators.calculate_momentum_indicators(close_price)
            )
            
            # Add all features to result DataFrame
            for feature_name, feature_series in all_features.items():
                if isinstance(feature_series, pd.Series) and len(feature_series) == len(result_df):
                    result_df[f'tech_{feature_name}'] = feature_series
            
            # Log feature summary
            new_features = [col for col in result_df.columns if col.startswith('tech_')]
            logger.info(f"Generated {len(new_features)} technical features")
            logger.debug(f"Features: {new_features[:10]}...")  # Log first 10 feature names
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating technical features: {e}")
            return result_df  # Return original data if feature generation fails
    
    def generate_features_by_coin(self, df: pd.DataFrame, 
                                 coin_id_col: str = 'coin_id') -> pd.DataFrame:
        """
        Generate technical features for multiple coins in a single DataFrame.
        
        Args:
            df: DataFrame with multiple coins
            coin_id_col: Column name that identifies different coins
        
        Returns:
            DataFrame with technical features for each coin
        """
        
        if coin_id_col not in df.columns:
            logger.warning(f"Coin ID column '{coin_id_col}' not found. Processing as single coin.")
            return self.generate_features(df)
        
        all_coins_data = []
        unique_coins = df[coin_id_col].unique()
        
        logger.info(f"Processing {len(unique_coins)} coins: {unique_coins}")
        
        for coin_id in unique_coins:
            logger.debug(f"Processing technical features for {coin_id}")
            coin_data = df[df[coin_id_col] == coin_id].copy()
            
            if not coin_data.empty:
                coin_features = self.generate_features(coin_data)
                all_coins_data.append(coin_features)
            else:
                logger.warning(f"No data found for coin: {coin_id}")
        
        if all_coins_data:
            result = pd.concat(all_coins_data, ignore_index=True)
            logger.info(f"Combined technical features for all coins. Final shape: {result.shape}")
            return result
        else:
            logger.error("No technical features could be generated for any coin")
            return df


# Utility functions for easy import and usage
def add_technical_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function to add technical features to a DataFrame.
    """
    pipeline = TechnicalFeaturePipeline()
    return pipeline.generate_features(df, **kwargs)

def add_technical_features_by_coin(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function to add technical features for multiple coins.
    """
    pipeline = TechnicalFeaturePipeline()
    return pipeline.generate_features_by_coin(df, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    import datetime
    
    # Create sample price data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate Bitcoin-like price movement
    base_price = 30000
    returns = np.random.normal(0.001, 0.03, len(dates))  # Daily returns
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    sample_data = pd.DataFrame({
        'coin_id': 'bitcoin',
        'timestamp': dates,
        'price_usd': prices,
        'volume_24h': np.random.uniform(1e9, 5e9, len(dates))  # Random volume
    })
    
    print("=== TESTING TECHNICAL FEATURES ===")
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Price range: ${sample_data['price_usd'].min():.0f} - ${sample_data['price_usd'].max():.0f}")
    
    # Generate technical features
    pipeline = TechnicalFeaturePipeline()
    features_df = pipeline.generate_features(sample_data)
    
    print(f"\nAfter adding technical features: {features_df.shape}")
    
    # Show technical feature columns
    tech_columns = [col for col in features_df.columns if col.startswith('tech_')]
    print(f"\nGenerated {len(tech_columns)} technical features:")
    for i, col in enumerate(tech_columns[:15]):  # Show first 15
        print(f"{i+1:2d}. {col}")
    
    if len(tech_columns) > 15:
        print(f"    ... and {len(tech_columns) - 15} more")
    
    # Show sample of the data
    print(f"\nSample data (last 5 rows, selected columns):")
    display_cols = ['timestamp', 'price_usd', 'tech_sma_7', 'tech_rsi_14', 'tech_macd']
    available_cols = [col for col in display_cols if col in features_df.columns]
    print(features_df[available_cols].tail())