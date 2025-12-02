"""
Feature engineering (Phase 1'den)
"""

import pandas as pd
import numpy as np
from typing import List
import gc


class FeatureEngineer:
    """Feature engineering pipeline"""
    
    def __init__(self, config: dict):
        self.config = config
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            df: Raw dataframe
        
        Returns:
            DataFrame with features
        """
        print(f"\n{'='*60}")
        print(f"⚙️  FEATURE ENGINEERING")
        print(f"{'='*60}")
        
        df = df.copy()
        
        # 1. Price features
        df = self._price_features(df)
        
        # 2. Time features
        df = self._time_features(df)
        
        # 3. Customer aggregates
        df = self._customer_aggregates(df)
        
        # 4. Product aggregates
        df = self._product_aggregates(df)
        
        # 5. Anomaly indicators
        df = self._anomaly_indicators(df)
        
        # 6. Cleanup
        df = self._cleanup(df)
        
        print(f"\n✅ Feature engineering complete")
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df
    
    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-related features"""
        print("  → Price features...")
        
        df['total_price'] = (df['amount'] * df['unit_price']).astype('float32')
        df['is_return'] = (df['unit_price'] < 0).astype('int8')
        
        return df
    
    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Temporal features"""
        print("  → Time features...")
        
        df['hour'] = df['trans_date'].dt.hour.astype('int8')
        df['day_of_week'] = df['trans_date'].dt.dayofweek.astype('int8')
        df['day_of_month'] = df['trans_date'].dt.day.astype('int8')
        df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
        df['is_night_transaction'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype('int8')
        
        return df
    
    def _customer_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Customer-level aggregates"""
        print("  → Customer aggregates...")
        
        # Exclude guest users
        df_filtered = df[df['cust_id'] != 0].copy()
        
        customer_agg = df_filtered.groupby('cust_id').agg({
            'trans_id': 'count',
            'total_price': ['sum', 'mean'],
            'store_code': 'nunique',
            'trans_date': ['min', 'max']
        }).reset_index()
        
        customer_agg.columns = [
            'cust_id', 'transaction_count', 'total_spent', 'avg_transaction_value',
            'unique_stores', 'first_trans_date', 'last_trans_date'
        ]
        
        # Transaction velocity
        customer_agg['days_active'] = (
            (customer_agg['last_trans_date'] - customer_agg['first_trans_date']).dt.days + 1
        )
        customer_agg['transaction_velocity'] = (
            customer_agg['transaction_count'] / customer_agg['days_active']
        ).astype('float32')
        
        # Return rate
        return_counts = df_filtered[df_filtered['is_return'] == 1].groupby('cust_id').size()
        customer_agg['return_count'] = customer_agg['cust_id'].map(return_counts).fillna(0).astype('int32')
        customer_agg['return_rate'] = (
            customer_agg['return_count'] / customer_agg['transaction_count']
        ).astype('float32')
        
        # Merge
        df = df.merge(
            customer_agg[[
                'cust_id', 'transaction_count', 'total_spent',
                'avg_transaction_value', 'transaction_velocity',
                'unique_stores', 'return_rate'
            ]],
            on='cust_id',
            how='left'
        )
        
        # Fill NaN for guests
        fill_cols = ['transaction_count', 'total_spent', 'avg_transaction_value',
                     'transaction_velocity', 'unique_stores', 'return_rate']
        for col in fill_cols:
            df[col] = df[col].fillna(0).astype('float32')
        
        del df_filtered, customer_agg
        gc.collect()
        
        return df
    
    def _product_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Product-level aggregates"""
        print("  → Product aggregates...")
        
        df_filtered = df[df['cust_id'] != 0].copy()
        
        product_agg = df_filtered.groupby('product_code').agg({
            'unit_price': ['mean', 'std'],
            'amount': ['mean', 'median'],
            'trans_id': 'count'
        }).reset_index()
        
        product_agg.columns = [
            'product_code', 'product_avg_price', 'product_price_std',
            'product_avg_amount', 'product_median_amount', 'product_popularity'
        ]
        
        df = df.merge(product_agg, on='product_code', how='left')
        
        # Price deviation
        df['price_deviation'] = np.abs(df['unit_price'] - df['product_avg_price'])
        df['price_deviation'] = df['price_deviation'].fillna(0).astype('float32')
        
        # Amount anomaly
        df['is_unusual_amount'] = (
            df['amount'] > df['product_median_amount'] * 3
        ).astype('int8')
        
        del df_filtered, product_agg
        gc.collect()
        
        return df
    
    def _anomaly_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Anomaly indicators"""
        print("  → Anomaly indicators...")
        
        df['is_negative_price'] = (df['unit_price'] < 0).astype('int8')
        df['is_high_value'] = (df['total_price'] > 5000).astype('int8')
        df['is_bulk_purchase'] = (df['amount'] > 10).astype('int8')
        
        return df
    
    def _cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove temp columns, fill NaN"""
        print("  → Cleanup...")
        
        # Drop temp columns
        temp_cols = ['product_avg_price', 'product_price_std', 'product_median_amount']
        df = df.drop(columns=temp_cols, errors='ignore')
        
        # Fill NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        gc.collect()
        
        return df