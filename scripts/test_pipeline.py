#!/usr/bin/env python3
"""
Test complete pipeline with 10K sample
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.data.loader import load_data
from src.data.preprocessor import FeatureEngineer


def main():
    print("="*80)
    print("🧪 TESTING COMPLETE PIPELINE")
    print("="*80)
    
    # 1. Load config
    print("\n1️⃣  Loading config...")
    config = load_config('base')
    
    # 2. Set seed
    set_seed(config['project']['random_seed'])
    
    # 3. Logger
    logger = get_logger('test_pipeline', config['logging'])
    logger.info("Pipeline test started")
    
    # 4. Load data (10K sample)
    logger.info("Loading data...")
    df = load_data(config, nrows=10000)
    
    # 5. Feature engineering
    logger.info("Engineering features...")
    engineer = FeatureEngineer(config)
    df = engineer.engineer_features(df)
    
    # 6. Check clustering features
    logger.info("Checking clustering features...")
    clustering_features = config['features']['clustering_features']
    
    print(f"\n📊 Feature Check:")
    for feat in clustering_features:
        exists = feat in df.columns
        symbol = "✅" if exists else "❌"
        print(f"  {symbol} {feat}")
    
    # 7. Summary
    print(f"\n{'='*80}")
    print(f"✅ PIPELINE TEST COMPLETE")
    print(f"{'='*80}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"  NaN count: {df.isnull().sum().sum()}")
    
    logger.info("Pipeline test completed successfully")


if __name__ == "__main__":
    main()