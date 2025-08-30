# src/data/features/feature_pipeline.py

import logging
import pandas as pd
from typing import Dict
from src.data.features.technical import TechnicalFeaturePipeline
from src.data.features.sentiment import SentimentFeaturePipeline
from src.data.features.market_features import MarketFeaturePipeline
import numpy as np
# Importa aquí las clases para otras características, como On-chain o Macro

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """
    Orquestador principal para la generación de todas las características.
    Combina características técnicas, de sentimiento, on-chain, etc.
    """
    def __init__(self):
        self.technical_pipeline = TechnicalFeaturePipeline()
        self.sentiment_pipeline = SentimentFeaturePipeline()
        self.market_pipeline = MarketFeaturePipeline()
        # Inicializa aquí otras pipelines de características

    def run(self, historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta el pipeline de características.
        
        Args:
            historical_df: DataFrame con los datos históricos de precios y volumen.
        
        Returns:
            DataFrame enriquecido con todas las características calculadas.
        """
        if historical_df.empty:
            logger.warning("DataFrame de entrada está vacío. No se pueden generar características.")
            return pd.DataFrame()

        # 1. Generar características técnicas
        logger.info("Generando características técnicas...")
        technical_df = self.technical_pipeline.generate_features(historical_df)
        logger.info("Generando características de mercado...")
        market_df = self.market_pipeline.generate_features(historical_df)

        # 2. Generar características de sentimiento
        logger.info("Generando características de sentimiento...")
        # Asume que generate_sentiment_features puede aceptar el df histórico para alinear las fechas
        # o que usa un rango de fechas similar al del dataframe
        sentiment_df = self.sentiment_pipeline.generate_sentiment_features(
            trends_timeframe="today 3-m"
        )
        
        # 3. Unir los dataframes
        # La unión debe hacerse con el DataFrame técnico como base
        combined_df = technical_df.merge(
            market_df,
            on='timestamp',
            how='left'
        )

        # Luego unir con sentimiento
        sentiment_df = self.sentiment_pipeline.generate_sentiment_features()
        final_df = combined_df.merge(
            sentiment_df,
            on='timestamp',
            how='left'
        )
        
        logger.info(f"Pipeline de características completado. Forma del DataFrame final: {final_df.shape}")
        
        return final_df

if __name__ == "__main__":
    # Este es solo un ejemplo. En un pipeline real, el DataFrame de entrada
    # vendría del paso de procesamiento (DataProcessor).
    
    # Crea un DataFrame de ejemplo para probar
    data = {
        'coin_id': ['bitcoin'] * 100,
        'timestamp': pd.to_datetime(pd.date_range(end=datetime.now(), periods=100)),
        'price_usd': np.random.rand(100) * 10000,
        'volume_24h': np.random.rand(100) * 1e9,
        'market_cap': np.random.rand(100) * 1e11
    }
    sample_df = pd.DataFrame(data)
    
    # Ejecuta el pipeline de características
    feature_pipeline = FeaturePipeline()
    final_features_df = feature_pipeline.run(sample_df)
    
    if not final_features_df.empty:
        print("\n=== ESTRUCTURA DEL DATASET CON CARACTERÍSTICAS ===")
        print(final_features_df.info())
        print("\n=== PRIMERAS FILAS ===")
        print(final_features_df.head())