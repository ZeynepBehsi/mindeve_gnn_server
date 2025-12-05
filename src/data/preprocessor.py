"""
Feature Engineering Module
Processes raw transaction data into ML-ready features
"""

import pandas as pd
import numpy as np
from typing import Dict
import time


class FeatureEngineer:
    """
    Feature Engineering for Transaction Data
    
    Creates features for:
    - Price analysis (with discount handling)
    - Temporal patterns
    - Customer behavior
    - Product popularity
    - Anomaly detection
    - Campaign/discount patterns
    """
    
    def __init__(self, config: dict):
        self.config = config
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            df: Raw dataframe with transaction data
        
        Returns:
            DataFrame with engineered features
        """
        print(f"\n{'='*60}")
        print(f"⚙️  FEATURE ENGINEERING")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Make a copy
        df = df.copy()
        
        # Feature engineering steps
        print(f"  → Price features...")
        df = self._create_price_features(df)
        
        print(f"  → Discount features...")
        df = self._create_discount_features(df)
        
        print(f"  → Time features...")
        df = self._create_time_features(df)
        
        print(f"  → Customer aggregates...")
        df = self._create_customer_features(df)
        
        print(f"  → Product aggregates...")
        df = self._create_product_features(df)
        
        print(f"  → Anomaly indicators...")
        df = self._create_anomaly_features(df)
        
        print(f"  → Cleanup...")
        df = self._cleanup_features(df)
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ Feature engineering complete")
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features using discounted prices"""
        
        # Use discounted prices as the "effective" price
        df['effective_price'] = df['discounted_unit_price']
        df['total_price'] = df['discounted_total_price']
        
        # Price statistics (using effective prices)
        price_median = df['effective_price'].median()
        price_std = df['effective_price'].std()
        
        # Price deviation from median
        df['price_deviation'] = (df['effective_price'] - price_median).abs() / (price_std + 1e-6)
        
        # High/low price indicators
        price_75 = df['effective_price'].quantile(0.75)
        price_25 = df['effective_price'].quantile(0.25)
        
        df['is_high_price'] = (df['effective_price'] > price_75).astype(int)
        df['is_low_price'] = (df['effective_price'] < price_25).astype(int)
        
        # Unusual amount indicators
        df['is_unusual_amount'] = (df['amount'] > 5).astype(int)
        df['is_bulk_purchase'] = (df['amount'] >= 3).astype(int)
        
        return df
    
    def _create_discount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create discount and campaign-related features"""
        
        # Discount rate (how much % discount applied)
        df['discount_rate'] = (
            (df['no_discount'] - df['discounted_unit_price']) / 
            df['no_discount'].clip(lower=0.01)
        ).fillna(0)
        
        # Discount percentage (total discount as % of original price)
        df['discount_percentage'] = (
            df['total_discount_amount'] / 
            (df['no_discount'] * df['amount']).clip(lower=0.01) * 100
        ).fillna(0)
        
        # Campaign indicators
        df['has_campaign'] = (df['unit_campaign_discount'] > 0).astype(int)
        df['has_discount'] = (df['total_discount_amount'] > 0).astype(int)
        
        # Discount amount per unit
        df['discount_per_unit'] = df['total_discount_amount'] / df['amount'].clip(lower=1)
        
        # High discount indicator (>20% discount)
        df['is_high_discount'] = (df['discount_percentage'] > 20).astype(int)
        
        # Anomaly: campaign but no discount
        df['campaign_no_discount'] = (
            (df['has_campaign'] == 1) & (df['has_discount'] == 0)
        ).astype(int)
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        
        # Ensure trans_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['trans_date']):
            df['trans_date'] = pd.to_datetime(df['trans_date'])
        
        # Extract time components
        df['hour'] = df['trans_date'].dt.hour
        df['day_of_week'] = df['trans_date'].dt.dayofweek
        df['day_of_month'] = df['trans_date'].dt.day
        df['month'] = df['trans_date'].dt.month
        df['year'] = df['trans_date'].dt.year
        
        # Time-based indicators
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night_transaction'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(int)
        
        # Month indicators (seasonal patterns)
        df['is_holiday_season'] = df['month'].isin([11, 12, 1]).astype(int)
        
        return df
    
    def _create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer-level aggregated features"""
        
        # Customer transaction count
        cust_trans_count = df.groupby('cust_id').size()
        df['customer_transaction_count'] = df['cust_id'].map(cust_trans_count)
        
        # Customer total spending (using discounted prices)
        cust_total_spend = df.groupby('cust_id')['total_price'].sum()
        df['customer_total_spending'] = df['cust_id'].map(cust_total_spend)
        
        # Customer average transaction value
        df['avg_transaction_value'] = (
            df['customer_total_spending'] / df['customer_transaction_count']
        )
        
        # Customer unique products
        cust_unique_products = df.groupby('cust_id')['product_code'].nunique()
        df['customer_unique_products'] = df['cust_id'].map(cust_unique_products)
        
        # Customer unique stores (if multiple stores exist)
        if df['store_code'].nunique() > 1:
            cust_unique_stores = df.groupby('cust_id')['store_code'].nunique()
            df['unique_stores'] = df['cust_id'].map(cust_unique_stores)
        else:
            df['unique_stores'] = 1
        
        # Customer return rate (transactions with amount=1 vs bulk)
        cust_return_mask = df['amount'] == 1
        
        cust_return_rate = df.groupby('cust_id', group_keys=False).apply(
            lambda x: (x['amount'] == 1).sum() / len(x)
        )

        df['return_rate'] = df['cust_id'].map(cust_return_rate)
        
        # Transaction velocity (transactions per day)
        if df['trans_date'].notna().any():
            cust_first_trans = df.groupby('cust_id')['trans_date'].min()
            cust_last_trans = df.groupby('cust_id')['trans_date'].max()
            cust_days = (cust_last_trans - cust_first_trans).dt.days + 1
            
            cust_velocity = cust_trans_count / cust_days.clip(lower=1)
            df['transaction_velocity'] = df['cust_id'].map(cust_velocity)
        else:
            df['transaction_velocity'] = 0
        
        # Customer discount usage rate
        cust_discount_rate = df.groupby('cust_id')['has_discount'].mean()
        df['customer_discount_rate'] = df['cust_id'].map(cust_discount_rate)
        
        # Customer campaign usage rate
        cust_campaign_rate = df.groupby('cust_id')['has_campaign'].mean()
        df['customer_campaign_rate'] = df['cust_id'].map(cust_campaign_rate)
        
        return df
    
    def _create_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create product-level aggregated features"""
        
        # Product popularity (transaction count)
        prod_trans_count = df.groupby('product_code').size()
        df['product_popularity'] = df['product_code'].map(prod_trans_count)
        
        # Product average price
        prod_avg_price = df.groupby('product_code')['effective_price'].mean()
        df['product_avg_price'] = df['product_code'].map(prod_avg_price)
        
        # Product price deviation (current vs average)
        df['product_price_deviation'] = (
            (df['effective_price'] - df['product_avg_price']).abs() / 
            df['product_avg_price'].clip(lower=0.01)
        )
        
        # Product unique customers
        prod_unique_customers = df.groupby('product_code')['cust_id'].nunique()
        df['product_unique_customers'] = df['product_code'].map(prod_unique_customers)
        
        # Product discount frequency
        prod_discount_freq = df.groupby('product_code')['has_discount'].mean()
        df['product_discount_frequency'] = df['product_code'].map(prod_discount_freq)
        
        return df
    
    def _create_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create anomaly detection features"""
        
        # Extreme transaction value
        total_price_99 = df['total_price'].quantile(0.99)
        df['is_extreme_value'] = (df['total_price'] > total_price_99).astype(int)
        
        # Rapid successive transactions (same customer, short time)
        if df['trans_date'].notna().any():
            df = df.sort_values(['cust_id', 'trans_date'])
            df['time_since_last_trans'] = (
                df.groupby('cust_id')['trans_date'].diff().dt.total_seconds() / 3600
            )  # Hours
            df['is_rapid_transaction'] = (df['time_since_last_trans'] < 1).astype(int)
            df['time_since_last_trans'] = df['time_since_last_trans'].fillna(999)
        else:
            df['time_since_last_trans'] = 999
            df['is_rapid_transaction'] = 0
        
        # Same product, same customer, short time
        df['product_cust_key'] = df['product_code'].astype(str) + '_' + df['cust_id'].astype(str)
        df = df.sort_values(['product_cust_key', 'trans_date'])
        df['same_product_time_gap'] = (
            df.groupby('product_cust_key')['trans_date'].diff().dt.total_seconds() / 3600
        ).fillna(999)
        df['is_repeated_product_purchase'] = (df['same_product_time_gap'] < 24).astype(int)
        
        return df
    
    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up and finalize features"""
        
        # Drop temporary columns
        cols_to_drop = ['product_cust_key']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        # Fill any remaining NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Ensure no infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
    
    def get_feature_names(self) -> list:
        """Get list of engineered feature names"""
        return [
            # Price features
            'effective_price', 'total_price', 'price_deviation',
            'is_high_price', 'is_low_price', 'is_unusual_amount', 'is_bulk_purchase',
            
            # Discount features
            'discount_rate', 'discount_percentage', 'has_campaign', 'has_discount',
            'discount_per_unit', 'is_high_discount', 'campaign_no_discount',
            
            # Time features
            'hour', 'day_of_week', 'day_of_month', 'month', 'year',
            'is_weekend', 'is_night_transaction', 'is_business_hours', 'is_holiday_season',
            
            # Customer features
            'customer_transaction_count', 'customer_total_spending', 'avg_transaction_value',
            'customer_unique_products', 'unique_stores', 'return_rate', 'transaction_velocity',
            'customer_discount_rate', 'customer_campaign_rate',
            
            # Product features
            'product_popularity', 'product_avg_price', 'product_price_deviation',
            'product_unique_customers', 'product_discount_frequency',
            
            # Anomaly features
            'is_extreme_value', 'time_since_last_trans', 'is_rapid_transaction',
            'same_product_time_gap', 'is_repeated_product_purchase'
        ]