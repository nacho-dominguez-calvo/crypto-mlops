# tests/unit/data/test_feature_pipeline.py

import pandas as pd
import numpy as np
import pytest
from datetime import datetime
from unittest.mock import MagicMock

from src.data.features.feature_pipeline import FeaturePipeline


@pytest.fixture
def sample_df():
    data = {
        'coin_id': ['bitcoin'] * 10,
        'timestamp': pd.to_datetime(pd.date_range(end=datetime.now(), periods=10)),
        'price_usd': np.random.rand(10) * 10000,
        'volume_24h': np.random.rand(10) * 1e9,
        'market_cap': np.random.rand(10) * 1e11
    }
    return pd.DataFrame(data)


@pytest.fixture
def feature_pipeline(monkeypatch):
    pipeline = FeaturePipeline()

    # Mock technical pipeline
    mock_tech = MagicMock()
    mock_tech.generate_features.return_value = pd.DataFrame({
        "timestamp": pd.date_range(end=datetime.now(), periods=10),
        "tech_feature": np.arange(10)
    })
    pipeline.technical_pipeline = mock_tech

    # Mock market pipeline
    mock_market = MagicMock()
    mock_market.generate_features.return_value = pd.DataFrame({
        "timestamp": pd.date_range(end=datetime.now(), periods=10),
        "market_feature": np.arange(10, 20)
    })
    pipeline.market_pipeline = mock_market

    # Mock sentiment pipeline
    mock_sentiment = MagicMock()
    mock_sentiment.generate_sentiment_features.return_value = pd.DataFrame({
        "timestamp": pd.date_range(end=datetime.now(), periods=10),
        "sentiment_feature": np.arange(20, 30)
    })
    pipeline.sentiment_pipeline = mock_sentiment

    return pipeline


def test_run_returns_dataframe(feature_pipeline, sample_df):
    result = feature_pipeline.run(sample_df)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_run_includes_all_features(feature_pipeline, sample_df):
    result = feature_pipeline.run(sample_df)
    assert "tech_feature" in result.columns
    assert "market_feature" in result.columns
    assert "sentiment_feature" in result.columns


def test_run_empty_input(feature_pipeline):
    empty_df = pd.DataFrame()
    result = feature_pipeline.run(empty_df)
    assert result.empty
