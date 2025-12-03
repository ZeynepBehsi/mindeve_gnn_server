# scripts/run_full_pipeline.py
"""
Full pipeline for 89M dataset on server
Phases: Data → Features → Clustering → Graph → GNN Training
Estimated time: 3-5 hours on 3x RTX A4000
"""

import sys
sys.path.append('.')

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.utils.helpers import set_seed, get_device
from src.data.data_loader import DataLoader
from src.features.feature_engineer import FeatureEngineer
from src.labeling.clustering import ClusteringExperiment
from src.models.graph_builder import GraphBuilder
from src.models.gnn_models import GraphSAGE, GAT, GCN
from src.training.trainer import GNNTrainer
from src.training.evaluator import GNNEvaluator
import torch
import time
import json
from pathlib import Path


def main():
    print("\n" + "=" * 80)
    print("🚀 FULL PIPELINE: MindEve GNN Fraud Detection")
    print("=" * 80)
    
    # ========================================================================
    # SETUP
    # ========================================================================
    print("\n1️⃣  Setup...")
    loader = ConfigLoader()
    config = loader.load_all()
    logger = setup_logger('full_pipeline', config)
    
    set_seed(config['project']['random_seed'])
    device = get_device(config)
    
    logger.info("=" * 80)
    logger.info("FULL PIPELINE STARTED")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Test mode: {config['test_mode']['enabled']}")
    
    start_time = time.time()
    
    # ========================================================================
    # PHASE 1: DATA LOADING
    # ========================================================================
    print("\n2️⃣  Phase 1: Loading data...")
    logger.info("Phase 1: Data loading")
    
    phase_start = time.time()
    data_loader = DataLoader(config)
    df = data_loader.load_data()
    
    logger.info(f"✅ Data loaded: {len(df):,} transactions")
    logger.info(f"   Time: {time.time() - phase_start:.1f}s")
    
    # ========================================================================
    # PHASE 2: FEATURE ENGINEERING
    # ========================================================================
    print("\n3️⃣  Phase 2: Feature engineering...")
    logger.info("Phase 2: Feature engineering")
    
    phase_start = time.time()
    feature_eng = FeatureEngineer(config)
    df = feature_eng.engineer_features(df)
    
    logger.info(f"✅ Features engineered: {len(df.columns)} columns")
    logger.info(f"   Time: {time.time() - phase_start:.1f}s")
    
    # ========================================================================
    # PHASE 3: CLUSTERING FOR LABELS
    # ========================================================================
    print("\n4️⃣  Phase 3: Clustering for fraud labels...")
    logger.info("Phase 3: Clustering")
    
    phase_start = time.time()
    clustering = ClusteringExperiment(
        config, 
        test_mode=config['test_mode']['enabled']
    )
    
    clustering.run_all(df)
    fraud_label, fraud_score = clustering.create_ensemble()
    
    df['fraud_label'] = fraud_label
    df['fraud_score'] = fraud_score
    
    logger.info(f"✅ Clustering complete: {fraud_label.sum():,} fraud labels")
    logger.info(f"   Fraud rate: {fraud_label.mean()*100:.2f}%")
    logger.info(f"   Time: {time.time() - phase_start:.1f}s")
    
    # ========================================================================
    # PHASE 4: GRAPH CONSTRUCTION
    # ========================================================================
    print("\n5️⃣  Phase 4: Building heterogeneous graph...")
    logger.info("Phase 4: Graph construction")
    
    phase_start = time.time()
    graph_builder = GraphBuilder(config)
    graph, transaction_mapping, node_mappings = graph_builder.build_graph(df)
    
    logger.info(f"✅ Graph built:")
    logger.info(f"   Customers: {graph['customer'].num_nodes:,}")
    logger.info(f"   Products: {graph['product'].num_nodes:,}")
    logger.info(f"   Stores: {graph['store'].num_nodes:,}")
    logger.info(f"   Edges: {sum([e.shape[1] for e in graph.edge_index_dict.values()]):,}")
    logger.info(f"   Time: {time.time() - phase_start:.1f}s")
    
    # ========================================================================
    # PHASE 5: TRAIN/VAL/TEST SPLITS
    # ========================================================================
    print("\n6️⃣  Phase 5: Creating splits...")
    logger.info("Phase 5: Splits")
    
    train_idx, val_idx, test_idx = data_loader.create_splits(df)
    
    logger.info(f"✅ Splits created:")
    logger.info(f"   Train: {len(train_idx):,}")
    logger.info(f"   Val: {len(val_idx):,}")
    logger.info(f"   Test: {len(test_idx):,}")
    
    # ========================================================================
    # PHASE 6: GNN TRAINING (ALL ARCHITECTURES)
    # ========================================================================
    print("\n7️⃣  Phase 6: Training GNN models...")
    logger.info("Phase 6: GNN training")
    
    results = {}
    models_to_train = config['test_mode'].get('models_to_test', ['sage', 'gat', 'gcn'])
    
    for model_name in models_to_train:
        print(f"\n  🔹 Training {model_name.upper()}...")
        logger.info(f"Training {model_name.upper()}")
        
        phase_start = time.time()
        
        # Initialize model
        if model_name == 'sage':
            model = GraphSAGE(config)
        elif model_name == 'gat':
            model = GAT(config)
        elif model_name == 'gcn':
            model = GCN(config)
        else:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue
        
        model = model.to(device)
        
        # Train
        trainer = GNNTrainer(model, config, logger)
        history = trainer.train(
            graph, 
            transaction_mapping, 
            train_idx, 
            val_idx
        )
        
        # Evaluate
        evaluator = GNNEvaluator(config)
        metrics = evaluator.evaluate(
            model, 
            graph, 
            transaction_mapping, 
            test_idx
        )
        
        # Save results
        results[model_name] = {
            'metrics': metrics,
            'history': history,
            'training_time': time.time() - phase_start
        }
        
        logger.info(f"✅ {model_name.upper()} complete:")
        logger.info(f"   Test F1: {metrics['f1']:.4f}")
        logger.info(f"   Test AUC: {metrics['auc']:.4f}")
        logger.info(f"   Training time: {results[model_name]['training_time']/60:.1f} min")
        
        # Save model
        model_path = Path(config['output']['models_dir']) / f'{model_name}_final.pt'
        torch.save(model.state_dict(), model_path)
        logger.info(f"   Model saved: {model_path}")
    
    # ========================================================================
    # SAVE FINAL RESULTS
    # ========================================================================
    print("\n8️⃣  Saving results...")
    
    # Create summary
    summary = {
        'dataset': {
            'total_transactions': len(df),
            'fraud_count': fraud_label.sum(),
            'fraud_rate': fraud_label.mean(),
            'date_range': {
                'start': str(df['trans_date'].min()),
                'end': str(df['trans_date'].max())
            }
        },
        'graph': {
            'num_customers': graph['customer'].num_nodes,
            'num_products': graph['product'].num_nodes,
            'num_stores': graph['store'].num_nodes,
            'num_edges': sum([e.shape[1] for e in graph.edge_index_dict.values()])
        },
        'splits': {
            'train': len(train_idx),
            'val': len(val_idx),
            'test': len(test_idx)
        },
        'models': {}
    }
    
    # Add model results
    for model_name, result in results.items():
        summary['models'][model_name] = {
            'test_f1': float(result['metrics']['f1']),
            'test_auc': float(result['metrics']['auc']),
            'test_precision': float(result['metrics']['precision']),
            'test_recall': float(result['metrics']['recall']),
            'training_time_minutes': result['training_time'] / 60,
            'best_epoch': result['history']['best_epoch']
        }
    
    # Save summary
    summary_path = Path(config['output']['reports_dir']) / 'full_pipeline_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Summary saved: {summary_path}")
    
    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("✅ FULL PIPELINE COMPLETE")
    print("=" * 80)
    
    print(f"\n⏱️  Total Time: {total_time/3600:.2f} hours")
    
    print("\n📊 Results:")
    for model_name, result in results.items():
        print(f"\n  {model_name.upper()}:")
        print(f"    F1:        {result['metrics']['f1']:.4f}")
        print(f"    AUC:       {result['metrics']['auc']:.4f}")
        print(f"    Precision: {result['metrics']['precision']:.4f}")
        print(f"    Recall:    {result['metrics']['recall']:.4f}")
        print(f"    Time:      {result['training_time']/60:.1f} min")
    
    # Best model
    best_model = max(results.items(), key=lambda x: x[1]['metrics']['f1'])
    print(f"\n🏆 Best Model: {best_model[0].upper()}")
    print(f"   F1: {best_model[1]['metrics']['f1']:.4f}")
    print(f"   AUC: {best_model[1]['metrics']['auc']:.4f}")
    
    print(f"\n📁 Outputs:")
    print(f"   Summary: {summary_path}")
    print(f"   Models: {config['output']['models_dir']}")
    print(f"   Logs: {config['output']['log_dir']}")
    
    logger.info("=" * 80)
    logger.info("FULL PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)