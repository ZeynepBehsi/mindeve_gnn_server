"""
Data loading with sample support
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import time


class DataLoader:
    """CSV data loader with sampling"""
    
    def __init__(self, config: dict):
        self.config = config
        self.data_path = config['data']['raw_data_path']
        self.sample_size = config['data'].get('sample_size', None)
        
        # Column names (Phase 1'den)
        self.column_names = [
            'TRANS_ID', 'TRANS_DATE', 'STORE_CODE', 'CUST_ID',
            'PRODUCT_CODE', 'BARCODE', 'AMOUNT', 'UNIT_PRICE'
        ]
        
        self.dtypes = {
            'TRANS_ID': 'int64',
            'STORE_CODE': 'int32',
            'CUST_ID': 'int64',
            'PRODUCT_CODE': 'int64',
            'BARCODE': 'int64',
            'AMOUNT': 'int32'
        }
    
    def load(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from CSV
        
        Args:
            nrows: Number of rows to load (None = all)
        
        Returns:
            DataFrame
        """
        print(f"\n{'='*60}")
        print(f"📊 LOADING DATA")
        print(f"{'='*60}")
        
        # Determine rows to load
        if nrows is not None:
            rows_to_load = nrows
        elif self.sample_size is not None:
            rows_to_load = self.sample_size
        else:
            rows_to_load = None
        
        print(f"  Data path: {self.data_path}")
        if rows_to_load:
            print(f"  Loading: {rows_to_load:,} rows (sample mode)")
        else:
            print(f"  Loading: ALL rows (full data)")
        
        # Check file exists
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Load
        start_time = time.time()
        
        df = pd.read_csv(
            self.data_path,
            header=None,
            names=self.column_names,
            dtype=self.dtypes,
            parse_dates=['TRANS_DATE'],
            nrows=rows_to_load,
            low_memory=False
        )
        
        elapsed = time.time() - start_time
        
        # Lowercase columns
        df.columns = df.columns.str.lower()
        
        print(f"\n✅ Data loaded in {elapsed:.2f}s")
        print(f"  Shape: {df.shape}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        print(f"  Date range: {df['trans_date'].min().date()} to {df['trans_date'].max().date()}")
        
        return df
    
    def create_sample(self, output_path: str, n_samples: int = 10000):
        """
        Create sample CSV for testing
        
        Args:
            output_path: Output CSV path
            n_samples: Number of samples
        """
        print(f"\n📦 Creating sample data: {n_samples:,} rows")
        
        df = self.load(nrows=n_samples)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        print(f"✅ Sample saved to: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")


def load_data(config: dict, nrows: Optional[int] = None) -> pd.DataFrame:
    """Shortcut function"""
    loader = DataLoader(config)
    return loader.load(nrows=nrows)