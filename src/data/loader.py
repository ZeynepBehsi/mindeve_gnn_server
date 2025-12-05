"""
Data Loading Module
Handles CSV loading with optimized dtypes and column mapping
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import time

def load_data(config: dict, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load transaction data with automatic column mapping
    
    Args:
        config: Configuration dictionary
        sample_size: Number of rows to load (None for all)
    
    Returns:
        DataFrame with standardized column names
    """
    
    print(f"\n{'='*60}")
    print(f"📊 LOADING DATA")
    print(f"{'='*60}")
    
    # Get data path
    data_path = config['data'].get('raw_data_path', 'data/sample/2024_05_sample.csv')
    
    # Determine if using new or old dataset
    is_new_dataset = 'combined_sales' in data_path or 'Copy of' in data_path
    
    print(f"  Data path: {data_path}")
    
    # Check test mode
    test_mode = config.get('test_mode', {}).get('enabled', False)
    if test_mode and sample_size is None:
        sample_size = config.get('test_mode', {}).get('sample_size', 10000)
    
    if sample_size:
        print(f"  Loading: {sample_size:,} rows (sample mode)")
    else:
        print(f"  Loading: ALL rows (full data)")
    
    start_time = time.time()
    
    # Load based on dataset type
    if is_new_dataset:
        df = _load_new_dataset(data_path, sample_size)
    else:
        df = _load_old_dataset(data_path, sample_size)
    
    load_time = time.time() - start_time
    
    print(f"\n✅ Data loaded in {load_time:.2f}s")
    print(f"  Shape: {df.shape}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Print date range if date column exists
    if 'trans_date' in df.columns:
        print(f"  Date range: {df['trans_date'].min()} to {df['trans_date'].max()}")
    
    return df


def _load_old_dataset(data_path: str, sample_size: Optional[int]) -> pd.DataFrame:
    """
    Load old dataset (2024_05_sample.csv format)
    
    Format:
    - No header row
    - 8 columns: TRANS_ID, TRANS_DATE, STORE_CODE, CUST_ID, PRODUCT_CODE, BARCODE, AMOUNT, UNIT_PRICE
    """
    
    # Column names (uppercase in old format)
    column_names = [
        'TRANS_ID', 'TRANS_DATE', 'STORE_CODE', 'CUST_ID',
        'PRODUCT_CODE', 'BARCODE', 'AMOUNT', 'UNIT_PRICE'
    ]
    
    # Load
    df = pd.read_csv(
        data_path,
        names=column_names,
        header=None,
        nrows=sample_size
    )
    
    # Normalize column names to lowercase (match new dataset)
    df.columns = df.columns.str.lower()
    
    # Convert trans_date to datetime
    df['trans_date'] = pd.to_datetime(df['trans_date'], errors='coerce')
    
    # Add placeholder columns for new dataset compatibility
    df['no_discount'] = df['unit_price']  # Assume no discount initially
    df['discounted_unit_price'] = df['unit_price']
    df['discounted_total_price'] = df['unit_price'] * df['amount']
    df['total_discount_amount'] = 0.0
    df['unit_campaign_discount'] = 0.0
    
    return df


def _load_new_dataset(data_path: str, sample_size: Optional[int]) -> pd.DataFrame:
    """
    Load new dataset (combined_sales_2022-2025 format)
    
    Format:
    - Has header row
    - 13 columns including discount/campaign information
    - Price columns use comma as decimal separator
    """
    
    # Optimized dtypes
    dtypes = {
        'trans_id': 'int64',
        'store_code': 'int32',
        'cust_id': 'int64',
        'product_code': 'int64',
        'barcode': 'int64',
        'amount': 'int32'
        # Price columns will be converted after loading
    }
    
    # Load with dtypes
    df = pd.read_csv(
        data_path,
        dtype=dtypes,
        parse_dates=['trans_date'],
        low_memory=False,
        nrows=sample_size
    )
    
    # Convert price columns (comma → dot decimal separator)
    price_columns = [
        'unit_price',
        'no_discount',
        'discounted_unit_price',
        'discounted_total_price',
        'total_discount_amount',
        'unit_campaign_discount'
    ]
    
    for col in price_columns:
        if col in df.columns:
            # Handle comma decimal separator
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
            else:
                df[col] = df[col].astype(float)
    
    return df


def get_column_mapping() -> Dict[str, str]:
    """
    Get column name mapping (old → new)
    
    Returns:
        Dictionary mapping old column names to new names
    """
    return {
        'TRANS_ID': 'trans_id',
        'TRANS_DATE': 'trans_date',
        'STORE_CODE': 'store_code',
        'CUST_ID': 'cust_id',
        'PRODUCT_CODE': 'product_code',
        'BARCODE': 'barcode',
        'AMOUNT': 'amount',
        'UNIT_PRICE': 'unit_price'
    }


def validate_required_columns(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has all required columns
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if all required columns exist
    
    Raises:
        ValueError: If required columns are missing
    """
    required_columns = [
        'trans_id', 'trans_date', 'store_code', 'cust_id',
        'product_code', 'barcode', 'amount', 'unit_price'
    ]
    
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return True


def get_data_info(df: pd.DataFrame) -> Dict:
    """
    Get summary information about loaded data
    
    Args:
        df: Loaded DataFrame
    
    Returns:
        Dictionary with data statistics
    """
    return {
        'shape': df.shape,
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'date_range': {
            'min': str(df['trans_date'].min()) if 'trans_date' in df.columns else None,
            'max': str(df['trans_date'].max()) if 'trans_date' in df.columns else None
        },
        'unique_counts': {
            'customers': df['cust_id'].nunique() if 'cust_id' in df.columns else 0,
            'products': df['product_code'].nunique() if 'product_code' in df.columns else 0,
            'stores': df['store_code'].nunique() if 'store_code' in df.columns else 0
    
    }
}