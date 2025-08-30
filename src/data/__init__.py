from .ingestion import DataIngestionPipeline, CoinGeckoClient, CoinGeckoConfig
from .processor import DataProcessor

__all__ = [
    'DataIngestionPipeline',
    'CoinGeckoClient', 
    'CoinGeckoConfig',
    'DataProcessor'
]


# Example usage script: src/scripts/run_ingestion.py
if __name__ == "__main__":
    from src.data import DataIngestionPipeline, DataProcessor
    
    # Initialize pipeline
    pipeline = DataIngestionPipeline(
        coins=['bitcoin', 'ethereum']  # Start with 2 coins for testing
    )
    
    # Initialize processor
    processor = DataProcessor()
    
    try:
        # Run ingestion
        raw_results = pipeline.run_ingestion(include_historical=True, historical_days=7)
        
        # Process results into DataFrames
        processed_data = processor.process_ingestion_results(raw_results)
        
        # Print results
        print("\n=== INGESTION RESULTS ===")
        print(f"Pipeline ID: {raw_results['pipeline_run_id']}")
        print(f"Status: {raw_results['status']}")
        print(f"Coins: {raw_results['coins']}")
        
        print("\n=== PROCESSED DATA ===")
        for key, df in processed_data.items():
            print(f"\n{key.upper()}:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Sample:\n{df.head(2)}")
            
    except Exception as e:
        print(f"Pipeline failed: {e}")