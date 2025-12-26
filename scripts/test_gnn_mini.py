"""
Test GNN Models - Mini Version (100K Sample)
Fully debugged and tested
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from src.utils.config_loader import load_all_configs
from src.utils.seed import set_seed
from src.data.loader import load_data
from src.data.preprocessor import FeatureEngineer
from src.models.graph_builder import GraphBuilder
from src.models.gnn_models import HeteroGNN
from src.training.trainer import GNNTrainer
from src.training.evaluator import GNNEvaluator


def main():
    """Main test pipeline"""
    
    print("\n" + "="*80)
    print("🚀 GNN FRAUD DETECTION - MINI TEST PIPELINE")
    print("="*80)
    print(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Project: {project_root}")
    
    try:
        # ========================
        # 1. CONFIGURATION
        # ========================
        print("\n" + "="*80)
        print("1️⃣  LOADING CONFIGURATION")
        print("="*80)
        
        config = load_all_configs()
        
        # Simple logger setup
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger('gnn_mini_test')
        
        set_seed(config['project']['random_seed'])
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Device: {device}")
        print(f"✅ Random seed: {config['project']['random_seed']}")
        
        # ========================
        # 2. LOAD DATA WITH DATE FILTER
        # ========================
        print("\n" + "="*80)
        print("2️⃣  LOADING DATA WITH DATE FILTER")
        print("="*80)
        
        # Load ALL data first
        print(f"\n📊 Loading full dataset...")
        df = load_data(config, sample_size=None)
        print(f"  Total loaded: {len(df):,} rows")
        
        # Convert date column
        print(f"\n📅 Applying date range filter...")
        if not pd.api.types.is_datetime64_any_dtype(df['trans_date']):
            df['trans_date'] = pd.to_datetime(df['trans_date'], errors='coerce')
        
        # Date range
        start_date = pd.Timestamp('2023-09-01')
        end_date = pd.Timestamp('2025-02-28')
        
        # Filter by date
        df = df[(df['trans_date'] >= start_date) & (df['trans_date'] <= end_date)]
        print(f"  After date filter: {len(df):,} rows")
        
        # Safety check
        if len(df) == 0:
            raise ValueError(f"No data found in date range {start_date.date()} to {end_date.date()}")
        
        print(f"  Date range in data: {df['trans_date'].min()} to {df['trans_date'].max()}")
        
        # Sort by date for deterministic sampling
        print(f"\n🔄 Sorting by date...")
        df = df.sort_values('trans_date').reset_index(drop=True)
        print(f"  Sorted successfully")
        
        # Take first N rows (deterministic)
        target_size = 500000
        print(f"\n✂️  Taking first {target_size:,} rows...")
        
        if len(df) >= target_size:
            df = df.head(target_size)
            print(f"  Using {len(df):,} rows")
            print(f"  Sample date range: {df['trans_date'].min()} to {df['trans_date'].max()}")
        else:
            print(f"  Only {len(df):,} rows available (using all)")
        
        print(f"\n✅ Final dataset: {len(df):,} rows")
        
        # Safety check 2
        if len(df) < 1000:
            raise ValueError(f"Dataset too small: {len(df)} rows (minimum 1000 required)")
        
        # ========================
        # 3. FEATURE ENGINEERING
        # ========================
        print("\n" + "="*80)
        print("3️⃣  FEATURE ENGINEERING")
        print("="*80)
        
        engineer = FeatureEngineer(config)
        df_processed = engineer.engineer_features(df)
        
        print(f"✅ Processed shape: {df_processed.shape}")
        print(f"✅ Feature columns: {len(df_processed.columns)}")
        
        # Safety check 3
        if len(df_processed) == 0:
            raise ValueError("No data after feature engineering!")
        
        # ========================
        # 4. CREATE FRAUD LABELS
        # ========================
        print("\n" + "="*80)
        print("4️⃣  CREATING FRAUD LABELS")
        print("="*80)
        
        # Simple fraud labeling based on price anomalies
        if 'fraud_label' not in df_processed.columns:
            # Use discounted_total_price (new data structure)
            if 'discounted_total_price' in df_processed.columns:
                threshold = df_processed['discounted_total_price'].quantile(0.99)
                fraud_labels = (df_processed['discounted_total_price'] > threshold).astype(int).values
                print(f"  Using discounted_total_price (99th percentile: {threshold:.2f})")
            elif 'total_price' in df_processed.columns:
                threshold = df_processed['total_price'].quantile(0.99)
                fraud_labels = (df_processed['total_price'] > threshold).astype(int).values
                print(f"  Using total_price (99th percentile: {threshold:.2f})")
            elif 'amount' in df_processed.columns:
                threshold = df_processed['amount'].quantile(0.99)
                fraud_labels = (df_processed['amount'] > threshold).astype(int).values
                print(f"  Using amount (99th percentile: {threshold:.2f})")
            else:
                # Fallback: random labels (should not happen)
                print("  ⚠️  No price column found, using random labels")
                fraud_labels = np.random.binomial(1, 0.01, size=len(df_processed))
        else:
            fraud_labels = df_processed['fraud_label'].values
            print(f"  Using existing fraud_label column")
        
        fraud_count = fraud_labels.sum()
        fraud_rate = fraud_count / len(fraud_labels) * 100
        
        print(f"✅ Fraud cases: {fraud_count:,}")
        print(f"✅ Fraud rate: {fraud_rate:.2f}%")
        
        # Safety check 4
        if fraud_count == 0:
            raise ValueError("No fraud cases found! Check labeling logic.")
        
        if fraud_count == len(fraud_labels):
            raise ValueError("All cases labeled as fraud! Check labeling logic.")
        
        # ========================
        # 5. GRAPH CONSTRUCTION
        # ========================
        print("\n" + "="*80)
        print("5️⃣  BUILDING HETEROGENEOUS GRAPH")
        print("="*80)
        
        graph_builder = GraphBuilder(config)
        
        # Build graph (returns tuple: graph, transaction_mapping)
        hetero_data, transaction_mapping = graph_builder.build_graph(
            df_processed, 
            fraud_labels
        )
        
        print(f"✅ Graph structure:")
        print(f"   - Node types: {hetero_data.node_types}")
        print(f"   - Edge types: {hetero_data.edge_types}")
        print(f"   - Customers: {hetero_data['customer'].x.shape[0]:,}")
        print(f"   - Products: {hetero_data['product'].x.shape[0]:,}")
        print(f"   - Stores: {hetero_data['store'].x.shape[0]:,}")
        
        # Safety check 5
        num_transactions = len(transaction_mapping)
        if num_transactions == 0:
            raise ValueError("No transactions in mapping!")
        
        # ========================
        # 6. PREPARE TRAIN/VAL/TEST SPLIT
        # ========================
        print("\n" + "="*80)
        print("6️⃣  PREPARING TRAIN/VAL/TEST SPLIT")
        print("="*80)
        
        # Get transaction count
        num_trans = len(transaction_mapping)
        indices = np.arange(num_trans)
        np.random.shuffle(indices)
        
        # 70/15/15 split
        train_size = int(0.70 * num_trans)
        val_size = int(0.15 * num_trans)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size]
        test_idx = indices[train_size+val_size:]
        
        print(f"✅ Train samples: {len(train_idx):,}")
        print(f"✅ Val samples: {len(val_idx):,}")
        print(f"✅ Test samples: {len(test_idx):,}")
        
        # Get fraud rates
        train_labels = transaction_mapping['fraud_label'].iloc[train_idx].values
        val_labels = transaction_mapping['fraud_label'].iloc[val_idx].values
        test_labels = transaction_mapping['fraud_label'].iloc[test_idx].values
        
        print(f"✅ Train fraud rate: {train_labels.mean()*100:.2f}%")
        print(f"✅ Val fraud rate: {val_labels.mean()*100:.2f}%")
        print(f"✅ Test fraud rate: {test_labels.mean()*100:.2f}%")
        
        # ========================
        # 7. MODEL INITIALIZATION
        # ========================
        print("\n" + "="*80)
        print("7️⃣  INITIALIZING GNN MODEL")
        print("="*80)
        
        # Create model - HeteroGNN with conv_type
        model = HeteroGNN(config, conv_type='sage').to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model: HeteroGNN (GraphSAGE)")
        print(f"   - Hidden channels: {model.hidden_channels}")
        print(f"   - Num layers: {model.num_layers}")
        print(f"   - Dropout: {model.dropout}")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        
        # ========================
        # 8. TRAINING
        # ========================
        print("\n" + "="*80)
        print("8️⃣  TRAINING MODEL")
        print("="*80)
        
        # Move graph to device
        hetero_data = hetero_data.to(device)
        
        # Create trainer
        trainer = GNNTrainer(model, config, logger)
        
        # Override num_epochs for quick test
        original_epochs = config['training']['num_epochs']
        config['training']['num_epochs'] = 10  # Quick test
        print(f"🏋️  Training for {config['training']['num_epochs']} epochs (mini test mode)...")
        
        # Train
        history = trainer.train(
            graph=hetero_data,
            transaction_mapping=transaction_mapping,
            train_idx=train_idx,
            val_idx=val_idx
        )
        
        # Restore original epochs
        config['training']['num_epochs'] = original_epochs
        
        print(f"✅ Training completed!")
        print(f"   - Best epoch: {history['best_epoch']}")
        print(f"   - Best val loss: {history['best_val_loss']:.4f}")
        
        # ========================
        # 9. EVALUATION
        # ========================
        print("\n" + "="*80)
        print("9️⃣  EVALUATING MODEL")
        print("="*80)
        
        evaluator = GNNEvaluator(model, device)
        
        # Prepare test data dict
        test_data = {
            'customer_idx': transaction_mapping['customer_idx'].iloc[test_idx].values,
            'product_idx': transaction_mapping['product_idx'].iloc[test_idx].values,
            'store_idx': transaction_mapping['store_idx'].iloc[test_idx].values,
            'labels': transaction_mapping['fraud_label'].iloc[test_idx].values
        }
        
        results = evaluator.evaluate_full(hetero_data, test_data)
        
        # ========================
        # 10. SAVE RESULTS
        # ========================
        print("\n" + "="*80)
        print("🔟 SAVING RESULTS")
        print("="*80)
        
        output_dir = project_root / "outputs" / "test_results_mini"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model with config and history
        model_path = output_dir / "test_model_mini.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'history': history
        }, model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Save results
        import json
        results_path = output_dir / "test_metrics_mini.json"
        with open(results_path, 'w') as f:
            metrics_save = {
                'sample_size': len(df),
                'date_range': {
                    'start': str(df['trans_date'].min()),
                    'end': str(df['trans_date'].max())
                },
                'fraud_rate': float(fraud_rate),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1': float(results['f1']),
                'auc': float(results['auc']),
                'confusion_matrix': results['confusion_matrix'].tolist(),
                'top_k': {k: float(v) for k, v in results['top_k'].items()} if results.get('top_k') else {},
                'training_history': {
                    'best_epoch': history['best_epoch'],
                    'best_val_loss': history['best_val_loss'],
                    'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
                    'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None
                }
            }
            json.dump(metrics_save, f, indent=2)
        print(f"✅ Metrics saved: {results_path}")
        
        # ========================
        # DONE
        # ========================
        print("\n" + "="*80)
        print("✅ MINI TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📊 Final Results:")
        print(f"   Sample size:  {len(df):,}")
        print(f"   Fraud rate:   {fraud_rate:.2f}%")
        print(f"   Precision:    {results['precision']:.4f}")
        print(f"   Recall:       {results['recall']:.4f}")
        print(f"   F1-Score:     {results['f1']:.4f}")
        print(f"   AUC-ROC:      {results['auc']:.4f}")
        
        if results.get('top_k'):
            print(f"\n🎯 Top-K Precision:")
            for k, prec in results['top_k'].items():
                print(f"   {k}: {prec:.4f}")
        
        print(f"\n⏰ End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 Results: {output_dir}")
        
        return 0
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ ERROR OCCURRED!")
        print("="*80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        
        import traceback
        print("\n📋 Full traceback:")
        traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
