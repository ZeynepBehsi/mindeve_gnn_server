#!/usr/bin/env python3
"""
Test Phase 4: GNN models
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.data.loader import load_data
from src.data.preprocessor import FeatureEngineer
from src.labeling.clustering import ClusteringExperiment
from src.models.graph_builder import GraphBuilder
from src.models.gnn_models import GraphSAGE, GAT, GCN
from src.training.trainer import GNNTrainer
from src.training.evaluator import GNNEvaluator
import torch
import numpy as np


def create_splits(transaction_mapping, test_size=0.15, val_size=0.15):
    """Create temporal train/val/test splits"""
    n = len(transaction_mapping)
    
    # Sort by transaction ID (temporal order)
    transaction_mapping = transaction_mapping.sort_values('trans_id').reset_index(drop=True)
    
    # Split indices
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    
    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, n))
    
    return train_idx, val_idx, test_idx


def prepare_data_dict(transaction_mapping, indices):
    """Prepare data dictionary for training/evaluation"""
    subset = transaction_mapping.iloc[indices]
    
    return {
        'customer_idx': subset['customer_idx'].values,
        'product_idx': subset['product_idx'].values,
        'store_idx': subset['store_idx'].values,
        'labels': subset['fraud_label'].values
    }


def main():
    print("="*80)
    print("🚀 PHASE 4: GNN MODELS (TEST MODE)")
    print("="*80)
    
    # 1. Load configs with deep merge
    print("\n1️⃣  Loading configs...")
    loader = ConfigLoader()
    config = loader.load_all()
    config['test_mode']['enabled'] = True
    
    print("  ⚡ Test mode: ENABLED")
    
    # 2. Set seed
    set_seed(config['project']['random_seed'])
    
    # 3. Logger
    logger = get_logger('phase4_gnn', config['logging'])
    logger.info("Phase 4 started")
    
    # 4. Load data
    logger.info("Loading data...")
    df = load_data(config, nrows=10000)
    
    # 5. Feature engineering
    logger.info("Engineering features...")
    engineer = FeatureEngineer(config)
    df = engineer.engineer_features(df)
    
    # 6. Clustering for labels
    logger.info("Running clustering for labels...")
    experiment = ClusteringExperiment(config, test_mode=True)
    results = experiment.run_all(df)
    fraud_label, fraud_score = experiment.create_ensemble()
    
    print(f"\n  Fraud labels: {fraud_label.sum():,} / {len(fraud_label):,} ({fraud_label.mean()*100:.2f}%)")
    
    # 7. Build graph
    logger.info("Building graph...")
    builder = GraphBuilder(config)
    graph, transaction_mapping = builder.build_graph(df, fraud_label)
    
    # Save graph
    builder.save_graph(graph, transaction_mapping, 'data/processed')
    
    # 8. Create splits
    logger.info("Creating train/val/test splits...")
    train_idx, val_idx, test_idx = create_splits(transaction_mapping)
    
    train_data = prepare_data_dict(transaction_mapping, train_idx)
    val_data = prepare_data_dict(transaction_mapping, val_idx)
    test_data = prepare_data_dict(transaction_mapping, test_idx)
    
    print(f"\n  Splits:")
    print(f"    Train: {len(train_idx):,} ({train_data['labels'].sum():,} fraud)")
    print(f"    Val: {len(val_idx):,} ({val_data['labels'].sum():,} fraud)")
    print(f"    Test: {len(test_idx):,} ({test_data['labels'].sum():,} fraud)")
    
    # 9. Train GNN (GraphSAGE only in test mode)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")
    
    logger.info("Training GraphSAGE...")
    model = GraphSAGE(config)
    
    trainer = GNNTrainer(model, config, device=device)
    history = trainer.train(graph, train_data, val_data, verbose=True)
    
    # 10. Evaluate
    logger.info("Evaluating model...")
    evaluator = GNNEvaluator(model, device=device)
    test_results = evaluator.evaluate_full(graph, test_data)
    
    # 11. Save model
    import os
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/graphsage_test.pt')
    print(f"\n  ✅ Model saved: outputs/models/graphsage_test.pt")
    
    # 12. Save training history
    import pickle
    with open('outputs/models/training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print(f"  ✅ History saved: outputs/models/training_history.pkl")
    
    # 13. Summary
    print(f"\n{'='*80}")
    print(f"✅ PHASE 4 TEST COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Final Results (GraphSAGE):")
    print(f"  Test F1: {test_results['f1']:.4f}")
    print(f"  Test AUC: {test_results['auc']:.4f}")
    print(f"  Test Precision: {test_results['precision']:.4f}")
    print(f"  Test Recall: {test_results['recall']:.4f}")
    
    if test_results['top_k']:
        print(f"\n  🎯 Top-K Precision:")
        for k, prec in test_results['top_k'].items():
            print(f"    {k.replace('_', '-').upper()}: {prec:.4f}")
    
    print(f"\n📁 Outputs:")
    print(f"  Model: outputs/models/graphsage_test.pt")
    print(f"  History: outputs/models/training_history.pkl")
    print(f"  Graph: data/processed/hetero_graph.pt")
    print(f"  Logs: outputs/logs/")
    
    logger.info("Phase 4 completed successfully")


if __name__ == "__main__":
    main()