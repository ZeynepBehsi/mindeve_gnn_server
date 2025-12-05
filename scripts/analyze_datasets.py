#!/usr/bin/env python3
"""
Dataset Analysis & Comparison Script
Analyzes old (2024_05_sample.csv) vs new (combined_sales_2022-2025) datasets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Setup
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Paths
OLD_DATA = "data/sample/2024_05_sample.csv"
NEW_DATA = "data/raw/Copy of combined_sales_2022-2025_filtered_more_than_6_transactions.csv"
EDA_DIR = Path("EDA")

def load_old_dataset():
    """Load old dataset (simple structure)"""
    print("\n" + "="*80)
    print("📊 LOADING OLD DATASET")
    print("="*80)
    
    # ESKİ DATASET HEADER YOK - MANUEL EKLE
    column_names = [
        'TRANS_ID', 'TRANS_DATE', 'STORE_CODE', 'CUST_ID', 
        'PRODUCT_CODE', 'BARCODE', 'AMOUNT', 'UNIT_PRICE'
    ]
    
    df = pd.read_csv(OLD_DATA, names=column_names, header=None)
    
    print(f"✅ Shape: {df.shape}")
    print(f"✅ Columns: {list(df.columns)}")
    print(f"✅ Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def load_new_dataset():
    """Load new dataset with optimized dtypes"""
    print("\n" + "="*80)
    print("📊 LOADING NEW DATASET")
    print("="*80)
    
    # Dtype optimization
    dtypes = {
        'trans_id': 'int64',
        'store_code': 'int32',
        'cust_id': 'int64',
        'product_code': 'int64',
        'barcode': 'int64',
        'amount': 'int32'
    }
    
    print("Loading (this may take a while for large files)...")
    df = pd.read_csv(
        NEW_DATA,
        dtype=dtypes,
        parse_dates=['trans_date'],
        low_memory=False,
        nrows=100000  # First 100K rows for analysis
    )
    
    # Convert price columns
    price_cols = [
        'unit_price', 'no_discount', 'discounted_unit_price',
        'discounted_total_price', 'total_discount_amount',
        'unit_campaign_discount'
    ]
    
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
    
    print(f"✅ Shape: {df.shape} (first 100K rows)")
    print(f"✅ Columns: {list(df.columns)}")
    print(f"✅ Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def compare_columns(df_old, df_new):
    """Compare column structures"""
    print("\n" + "="*80)
    print("🔍 COLUMN COMPARISON")
    print("="*80)
    
    old_cols = set(df_old.columns)
    new_cols = set(df_new.columns)
    
    # Find differences
    only_old = old_cols - new_cols
    only_new = new_cols - old_cols
    common = old_cols & new_cols
    
    print(f"\n📊 Column Stats:")
    print(f"  Old dataset: {len(old_cols)} columns")
    print(f"  New dataset: {len(new_cols)} columns")
    print(f"  Common: {len(common)} columns")
    print(f"  Only in old: {len(only_old)} columns")
    print(f"  Only in new: {len(only_new)} columns")
    
    if only_old:
        print(f"\n❌ Columns ONLY in OLD dataset:")
        for col in sorted(only_old):
            print(f"  - {col}")
    
    if only_new:
        print(f"\n✨ NEW columns in new dataset:")
        for col in sorted(only_new):
            print(f"  - {col}")
    
    # Create mapping report
    mapping = {
        "common_columns": sorted(list(common)),
        "removed_columns": sorted(list(only_old)),
        "new_columns": sorted(list(only_new)),
        "total_old": len(old_cols),
        "total_new": len(new_cols)
    }
    
    # Save to JSON
    with open(EDA_DIR / "reports" / "column_mapping.json", 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n💾 Saved: EDA/reports/column_mapping.json")
    
    return mapping

def analyze_dtypes(df_old, df_new, common_cols):
    """Compare data types"""
    print("\n" + "="*80)
    print("🔍 DATA TYPE COMPARISON")
    print("="*80)
    
    dtype_changes = []
    
    for col in common_cols:
        old_dtype = str(df_old[col].dtype)
        new_dtype = str(df_new[col].dtype)
        
        if old_dtype != new_dtype:
            dtype_changes.append({
                'column': col,
                'old_dtype': old_dtype,
                'new_dtype': new_dtype
            })
            print(f"⚠️  {col}: {old_dtype} → {new_dtype}")
    
    if not dtype_changes:
        print("✅ No dtype changes in common columns")
    
    return dtype_changes

def analyze_statistics(df_old, df_new):
    """Compare basic statistics"""
    print("\n" + "="*80)
    print("📈 STATISTICAL COMPARISON")
    print("="*80)
    
    # Numeric columns only
    old_numeric = df_old.select_dtypes(include=[np.number])
    new_numeric = df_new.select_dtypes(include=[np.number])
    
    print("\n📊 OLD Dataset Statistics:")
    print(old_numeric.describe().T)
    
    print("\n📊 NEW Dataset Statistics:")
    print(new_numeric.describe().T)
    
    # Save to CSV
    old_numeric.describe().T.to_csv(EDA_DIR / "reports" / "old_dataset_stats.csv")
    new_numeric.describe().T.to_csv(EDA_DIR / "reports" / "new_dataset_stats.csv")
    
    print(f"\n💾 Saved statistics to EDA/reports/")
    
def analyze_missing_values(df_old, df_new):
    """Compare missing values"""
    print("\n" + "="*80)
    print("🔍 MISSING VALUES ANALYSIS")
    print("="*80)
    
    old_missing = df_old.isnull().sum()
    new_missing = df_new.isnull().sum()
    
    old_missing_pct = (old_missing / len(df_old) * 100).round(2)
    new_missing_pct = (new_missing / len(df_new) * 100).round(2)
    
    print("\n📊 OLD Dataset Missing Values:")
    if old_missing.sum() > 0:
        print(old_missing[old_missing > 0])
        print(f"\nMissing %:")
        print(old_missing_pct[old_missing_pct > 0])
    else:
        print("✅ No missing values")
    
    print("\n📊 NEW Dataset Missing Values:")
    if new_missing.sum() > 0:
        print(new_missing[new_missing > 0])
        print(f"\nMissing %:")
        print(new_missing_pct[new_missing_pct > 0])
    else:
        print("✅ No missing values")
    
    # Visualization - only if there are missing values
    has_old_missing = (old_missing_pct > 0).any()
    has_new_missing = (new_missing_pct > 0).any()
    
    if has_old_missing or has_new_missing:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        if has_old_missing:
            old_missing_pct[old_missing_pct > 0].plot(kind='barh', ax=ax1, color='coral')
            ax1.set_title('Old Dataset - Missing Values %')
            ax1.set_xlabel('Missing %')
        else:
            ax1.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            ax1.set_title('Old Dataset - Missing Values %')
        
        if has_new_missing:
            new_missing_pct[new_missing_pct > 0].plot(kind='barh', ax=ax2, color='skyblue')
            ax2.set_title('New Dataset - Missing Values %')
            ax2.set_xlabel('Missing %')
        else:
            ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            ax2.set_title('New Dataset - Missing Values %')
        
        plt.tight_layout()
        plt.savefig(EDA_DIR / "visualizations" / "missing_values_comparison.png", dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved: EDA/visualizations/missing_values_comparison.png")
        plt.close()
    else:
        print("\n✅ No missing values in either dataset - skipping visualization")

def create_comparison_report(df_old, df_new, mapping, dtype_changes):
    """Create comprehensive markdown report"""
    print("\n" + "="*80)
    print("📝 CREATING COMPARISON REPORT")
    print("="*80)
    
    report = f"""# Dataset Comparison Report
## MindEve GNN Fraud Detection

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Overview

| Metric | Old Dataset | New Dataset |
|--------|-------------|-------------|
| Rows | {len(df_old):,} | {len(df_new):,} (sample) |
| Columns | {len(df_old.columns)} | {len(df_new.columns)} |
| Memory | {df_old.memory_usage(deep=True).sum() / 1024**2:.2f} MB | {df_new.memory_usage(deep=True).sum() / 1024**2:.2f} MB |
| Date Range | {df_old['TRANS_DATE'].min() if 'TRANS_DATE' in df_old.columns else 'N/A'} to {df_old['TRANS_DATE'].max() if 'TRANS_DATE' in df_old.columns else 'N/A'} | {df_new['trans_date'].min()} to {df_new['trans_date'].max()} |

---

## 🔍 Column Changes

### Common Columns ({len(mapping['common_columns'])})
```
{', '.join(mapping['common_columns'])}
```

### Removed Columns ({len(mapping['removed_columns'])})
```
{', '.join(mapping['removed_columns']) if mapping['removed_columns'] else 'None'}
```

### New Columns ({len(mapping['new_columns'])})
```
{', '.join(mapping['new_columns']) if mapping['new_columns'] else 'None'}
```

---

## ⚙️ Data Type Changes

{'No data type changes detected.' if not dtype_changes else ''}

"""
    
    if dtype_changes:
        report += "| Column | Old Type | New Type |\n"
        report += "|--------|----------|----------|\n"
        for change in dtype_changes:
            report += f"| {change['column']} | {change['old_dtype']} | {change['new_dtype']} |\n"
    
    report += """

---

## 🔧 Required Code Changes

### 1. Data Loader (`src/data/loader.py`)
- [ ] Update column name mapping
- [ ] Adjust dtype specifications
- [ ] Update date parsing logic

### 2. Preprocessor (`src/data/preprocessor.py`)
- [ ] Review feature engineering for new columns
- [ ] Adjust aggregation logic if needed
- [ ] Update price column handling (comma → dot)

### 3. Config Files
- [ ] Create `config/data_mapping.yaml`
- [ ] Update `config/base_config.yaml` with new path
- [ ] Adjust date ranges

### 4. Tests
- [ ] Unit tests for new column mapping
- [ ] Integration test with sample data
- [ ] Validation of feature engineering

---

## 📊 Next Steps

1. Review this report
2. Create column mapping config
3. Update data loader
4. Test with 100K sample
5. Full pipeline test

---

*Report generated by `scripts/analyze_datasets.py`*
"""
    
    # Save report
    with open(EDA_DIR / "reports" / "dataset_comparison.md", 'w') as f:
        f.write(report)
    
    print(f"💾 Saved: EDA/reports/dataset_comparison.md")

def main():
    """Main analysis pipeline"""
    print("\n" + "="*80)
    print("🚀 DATASET ANALYSIS & COMPARISON")
    print("="*80)
    
    # Load datasets
    df_old = load_old_dataset()
    df_new = load_new_dataset()
    
    # Compare columns
    mapping = compare_columns(df_old, df_new)
    
    # Compare dtypes
    common_cols = list(set(df_old.columns) & set(df_new.columns))
    dtype_changes = analyze_dtypes(df_old, df_new, common_cols)
    
    # Statistics
    analyze_statistics(df_old, df_new)
    
    # Missing values
    analyze_missing_values(df_old, df_new)
    
    # Final report
    create_comparison_report(df_old, df_new, mapping, dtype_changes)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 Check EDA/ directory for:")
    print(f"  - EDA/reports/dataset_comparison.md")
    print(f"  - EDA/reports/column_mapping.json")
    print(f"  - EDA/visualizations/missing_values_comparison.png")
    print("\n")

if __name__ == "__main__":
    main()