#!/usr/bin/env python3
"""
Test script for GNN models (Phase 4)
Runs on 10K sample data for quick testing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger
from src.data.loader import load_data
from src.data.preprocessor import FeatureEngineer
from src.labeling.clustering import ClusteringExperiment
from src.models.graph_builder import GraphBuilder
from src.models.gnn_models import GraphSAGE
from src.training.trainer import GNNTrainer
from src.training.evaluator import GNNEvaluator

def main():
    """Main test function"""
    
    print("\n" + "="*80)
    print("🚀 PHASE 4: GNN MODELS (TEST MODE)")
    print("="*80)
    
    # ============================================================
    # 1. SETUP
    # ============================================================
    print("\n1️⃣  Loading configs...")
    config_loader = ConfigLoader()
    config = config_loader.load_all()  # Load and merge all configs
    
    # Override with test mode
    config['test_mode']['enabled'] = True
    config['test_mode']['sample_size'] = 10000
    
    # Setup logger with full config
    logger = get_logger('phase4_gnn', config['logging'])
    
    # Set device
    device = torch.device(
        config['compute']['device'] if config['compute']['device'] != 'auto'
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Set seed
    seed = config['project']['random_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"✅ Random seed set to {seed}")
    
    logger.info("Phase 4 started")
    
    # ============================================================
    # 2. LOAD & PREPARE DATA
    # ============================================================
    logger.info("Loading data...")
    df = load_data(config)
    
    logger.info("Engineering features...")
    feature_engineer = FeatureEngineer(config)
    df = feature_engineer.engineer_features(df)
    
    logger.info("Running clustering for labels...")
    clustering_exp = ClusteringExperiment(config, test_mode=True)
    clustering_results = clustering_exp.run_all(df)
    
    # Get ensemble labels
    fraud_labels, fraud_scores = clustering_exp.create_ensemble()
    
    # Add to dataframe
    df['fraud_label'] = fraud_labels
    df['fraud_score'] = fraud_scores
    
    print(f"\n  Fraud labels: {fraud_labels.sum():,} / {len(fraud_labels):,} ({fraud_labels.mean()*100:.2f}%)")
    
    logger.info("Building graph...")
    graph_builder = GraphBuilder(config)
    graph = graph_builder.build_graph(df, fraud_labels)
    
    # Create transaction mapping
    transaction_mapping = df[['cust_id', 'product_code', 'store_code', 'fraud_label', 'fraud_score']].copy()
    transaction_mapping = transaction_mapping.reset_index(drop=True)
    
    # Save processed data
    output_dir = Path(config['data']['processed_data_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(graph, output_dir / 'hetero_graph.pt')
    transaction_mapping.to_pickle(output_dir / 'transaction_mapping.pkl')
    
    print(f"\n✅ Saved:")
    print(f"  Graph: {output_dir / 'hetero_graph.pt'}")
    print(f"  Transaction mapping: {output_dir / 'transaction_mapping.pkl'}")
    
    # ============================================================
    # 3. CREATE SPLITS
    # ============================================================
    logger.info("Creating train/val/test splits...")
    
    # Temporal split (70/15/15)
    n_samples = len(transaction_mapping)
    train_size = int(config['data']['splits']['train'] * n_samples)
    val_size = int(config['data']['splits']['val'] * n_samples)
    
    train_mask = np.zeros(n_samples, dtype=bool)
    val_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)
    
    train_mask[:train_size] = True
    val_mask[train_size:train_size+val_size] = True
    test_mask[train_size+val_size:] = True
    
    # Get indices
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    # Get labels
    train_labels = torch.LongTensor(
        transaction_mapping.loc[train_idx, 'fraud_label'].values
    )
    val_labels = torch.LongTensor(
        transaction_mapping.loc[val_idx, 'fraud_label'].values
    )
    test_labels = torch.LongTensor(
        transaction_mapping.loc[test_idx, 'fraud_label'].values
    )
    
    logger.info(f"\n  Splits:")
    logger.info(f"    Train: {len(train_idx):,} ({train_labels.sum():,} fraud)")
    logger.info(f"    Val: {len(val_idx):,} ({val_labels.sum():,} fraud)")
    logger.info(f"    Test: {len(test_idx):,} ({test_labels.sum():,} fraud)")
    logger.info(f"\n  Device: {device}")
    
    # ============================================================
    # 4. TRAIN GNN
    # ============================================================
    model_name = 'GraphSAGE'
    logger.info(f"Training {model_name}...")
    
    # Get model config
    model_config = config['architectures']['sage']
    
    # Initialize model
    model = GraphSAGE(
        in_channels=graph['customer'].x.shape[1],
        hidden_channels=model_config['hidden_channels'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    ).to(device)
    
    # Initialize trainer
    trainer = GNNTrainer(model, config)
    
    # Train
    history = trainer.train(
        graph=graph.to(device),
        transaction_mapping=transaction_mapping,
        train_idx=train_idx,
        val_idx=val_idx
    )
    
    # ============================================================
    # 5. EVALUATE
    # ============================================================
    logger.info("Evaluating model...")
    
    # Prepare test data
    test_data = {
        'customer_idx': transaction_mapping.loc[test_idx, 'cust_id'].map(
            graph_builder.customer_to_idx
        ).values,
        'product_idx': transaction_mapping.loc[test_idx, 'product_code'].map(
            graph_builder.product_to_idx
        ).values,
        'store_idx': transaction_mapping.loc[test_idx, 'store_code'].map(
            graph_builder.store_to_idx
        ).values,
        'labels': test_labels.numpy()
    }
    
    evaluator = GNNEvaluator(config)
    evaluator.model = model
    evaluator.device = device
    
    results = evaluator.evaluate_full(graph.to(device), test_data)
    
    # Print results
    print("\n" + "="*80)
    print("✅ PHASE 4 TEST COMPLETE")
    print("="*80)
    print(f"\n📊 Final Results ({model_name}):")
    print(f"  Test F1: {results['f1']:.4f}")
    print(f"  Test Precision: {results['precision']:.4f}")
    print(f"  Test Recall: {results['recall']:.4f}")
    print(f"  Test AUC: {results['auc']:.4f}")
    print("\n" + "="*80)
    
    logger.info("Phase 4 test completed successfully")

if __name__ == "__main__":
    main()