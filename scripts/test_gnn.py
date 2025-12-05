"""
Test GNN Models - Simplified Version
Tests graph construction and GNN model with small sample
"""

import sys
import os

# PYTHONPATH ayarı
project_root = "/home/zeynep/work_spase/mindeve_gnn_server/mindeve_gnn_server-main"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from datetime import datetime

# Project imports - Düzeltilmiş
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.data.loader import load_data
from src.data.preprocessor import FeatureEngineer
from src.models.graph_builder import GraphBuilder  # src/models'da!
from src.models.gnn_models import GraphSAGE
from src.training.trainer import GNNTrainer
from src.training.evaluator import GNNEvaluator  # Doğru class adı!


def main():
    """Main test pipeline"""
    
    print("\n" + "="*80)
    print("🚀 GNN MODEL TEST - SIMPLIFIED VERSION")
    print("="*80)
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Project Root: {project_root}")
    print(f"🐍 Python Path: {sys.path[0]}")
    
    try:
        # ========================
        # 1. SETUP
        # ========================
        print("\n" + "="*80)
        print("1️⃣  CONFIGURATION & SETUP")
        print("="*80)
        
        config_loader = ConfigLoader()
        config = config_loader.load_all()
        
        # Test mode settings
        config['test_mode'] = {
            'enabled': True,
            'sample_size': 10000,
            'quick_test': True
        }
        
        logger = get_logger('gnn_test')
        set_seed(config.get('random_seed', 42))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Device: {device}")
        print(f"✅ Random seed: {config.get('random_seed', 42)}")
        
        # ========================
        # 2. LOAD DATA
        # ========================
        print("\n" + "="*80)
        print("2️⃣  DATA LOADING")
        print("="*80)
        
        data_path = Path(config['data']['raw_path'])
        print(f"📂 Loading from: {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = load_data(str(data_path))
        print(f"✅ Original data shape: {df.shape}")
        
        # Sample for testing
        if config['test_mode']['enabled']:
            sample_size = config['test_mode']['sample_size']
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            print(f"✅ Sampled data shape: {df.shape}")
        
        # ========================
        # 3. FEATURE ENGINEERING
        # ========================
        print("\n" + "="*80)
        print("3️⃣  FEATURE ENGINEERING")
        print("="*80)
        
        engineer = FeatureEngineer(config)
        df_processed = engineer.fit_transform(df)
        
        print(f"✅ Processed shape: {df_processed.shape}")
        print(f"✅ Features: {df_processed.columns.tolist()[:10]}...")
        
        # ========================
        # 4. CREATE LABELS
        # ========================
        print("\n" + "="*80)
        print("4️⃣  LABEL CREATION")
        print("="*80)
        
        # Simple fraud label (using existing or creating dummy)
        if 'is_fraud' in df_processed.columns:
            fraud_labels = df_processed['is_fraud'].values
        else:
            # Create dummy labels based on amount anomalies
            if 'amount' in df_processed.columns:
                threshold = df_processed['amount'].quantile(0.99)
                fraud_labels = (df_processed['amount'] > threshold).astype(int).values
            else:
                fraud_labels = np.zeros(len(df_processed), dtype=int)
        
        df_processed['fraud_label'] = fraud_labels
        print(f"✅ Fraud labels created: {fraud_labels.sum()} frauds / {len(fraud_labels)} total")
        print(f"✅ Fraud rate: {fraud_labels.mean()*100:.2f}%")
        
        # ========================
        # 5. GRAPH CONSTRUCTION
        # ========================
        print("\n" + "="*80)
        print("5️⃣  GRAPH CONSTRUCTION")
        print("="*80)
        
        graph_config = config.get('graph', {
            'similarity_threshold': 0.5,
            'max_neighbors': 10,
            'edge_types': ['customer-product', 'customer-store', 'product-store']
        })
        
        graph_builder = GraphBuilder(graph_config)
        hetero_data = graph_builder.build_heterogeneous_graph(
            df_processed,
            node_types=['customer', 'product', 'store'],
            edge_types=graph_config['edge_types']
        )
        
        print(f"✅ Graph created:")
        print(f"   - Node types: {hetero_data.node_types}")
        print(f"   - Edge types: {hetero_data.edge_types}")
        for node_type in hetero_data.node_types:
            print(f"   - {node_type}: {hetero_data[node_type].num_nodes} nodes")
        for edge_type in hetero_data.edge_types:
            print(f"   - {edge_type}: {hetero_data[edge_type].num_edges} edges")
        
        # ========================
        # 6. MODEL INITIALIZATION
        # ========================
        print("\n" + "="*80)
        print("6️⃣  MODEL INITIALIZATION")
        print("="*80)
        
        model_config = config.get('gnn', {
            'hidden_channels': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'heads': 4
        })
        
        # Get feature dimensions from graph
        metadata = (hetero_data.node_types, hetero_data.edge_types)
        
        model = GraphSAGE(
            hidden_channels=model_config['hidden_channels'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            metadata=metadata
        ).to(device)
        
        print(f"✅ Model: GraphSAGE")
        print(f"   - Hidden channels: {model_config['hidden_channels']}")
        print(f"   - Num layers: {model_config['num_layers']}")
        print(f"   - Dropout: {model_config['dropout']}")
        print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # ========================
        # 7. TRAINING
        # ========================
        print("\n" + "="*80)
        print("7️⃣  MODEL TRAINING (QUICK TEST)")
        print("="*80)
        
        train_config = {
            'num_epochs': 2,  # Quick test
            'learning_rate': 0.001,
            'weight_decay': 5e-4,
            'patience': 10
        }
        
        trainer = GNNTrainer(model, train_config, device, logger)
        
        # Move graph to device
        hetero_data = hetero_data.to(device)
        
        # Simple train/val split
        num_nodes = hetero_data['customer'].num_nodes
        train_size = int(0.8 * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[:train_size] = True
        val_mask = ~train_mask
        
        print(f"✅ Train nodes: {train_mask.sum()}")
        print(f"✅ Val nodes: {val_mask.sum()}")
        
        # Train
        print("\n🏋️ Training...")
        history = trainer.train(
            hetero_data,
            train_mask,
            val_mask,
            num_epochs=train_config['num_epochs']
        )
        
        print(f"✅ Training completed!")
        print(f"   - Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"   - Final val loss: {history['val_loss'][-1]:.4f}")
        
        # ========================
        # 8. EVALUATION
        # ========================
        print("\n" + "="*80)
        print("8️⃣  MODEL EVALUATION")
        print("="*80)
        
        evaluator = GNNEvaluator(device)
        
        model.eval()
        with torch.no_grad():
            predictions = model(hetero_data.x_dict, hetero_data.edge_index_dict)
            
            # Evaluate on validation set
            metrics = evaluator.evaluate(
                predictions[val_mask],
                hetero_data['customer'].y[val_mask]
            )
        
        print(f"✅ Validation Metrics:")
        for metric, value in metrics.items():
            print(f"   - {metric}: {value:.4f}")
        
        # ========================
        # 9. SAVE RESULTS
        # ========================
        print("\n" + "="*80)
        print("9️⃣  SAVING RESULTS")
        print("="*80)
        
        output_dir = Path(config['paths']['models'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / "test_model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Save results
        results = {
            'config': config,
            'history': history,
            'metrics': metrics,
            'graph_info': {
                'num_nodes': {nt: hetero_data[nt].num_nodes for nt in hetero_data.node_types},
                'num_edges': {et: hetero_data[et].num_edges for et in hetero_data.edge_types}
            }
        }
        
        results_path = output_dir / "test_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"✅ Results saved: {results_path}")
        
        # ========================
        # DONE
        # ========================
        print("\n" + "="*80)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ ERROR OCCURRED!")
        print("="*80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
