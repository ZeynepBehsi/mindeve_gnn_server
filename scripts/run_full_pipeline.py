#!/usr/bin/env python3
"""
Full Pipeline: GNN Fraud Detection with Clustering Comparison
=============================================================

Tests 12 model combinations:
- 3 GNN architectures (GraphSAGE, GAT, GCN)  
- 4 Clustering methods (Simple quantile-based labels)

With 5-Fold Cross-Validation and MLflow Tracking

Based on test_gnn.py working structure
Author: Zeynep
Date: 2025-12-08
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
import json
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# MLflow
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow not available. Install with: pip install mlflow")

# Project modules (from test_gnn.py)
from src.utils.config_loader import load_all_configs
from src.utils.seed import set_seed
from src.data.loader import load_data
from src.data.preprocessor import FeatureEngineer
from src.models.graph_builder import GraphBuilder
from src.models.gnn_models import HeteroGNN, GraphSAGE, GAT, GCN
from src.training.trainer import GNNTrainer
from src.training.evaluator import GNNEvaluator


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

SAMPLE_SIZE = 5_000_000
CV_FOLDS = 5
GNN_TYPES = ['sage', 'gat', 'gcn']
CLUSTERING_TYPES = ['quantile_99', 'quantile_95', 'quantile_90', 'z_score']

OUTPUT_BASE = Path("outputs/full_pipeline_results")
MODELS_DIR = OUTPUT_BASE / "models"
METRICS_DIR = OUTPUT_BASE / "metrics"
COMPARISON_DIR = OUTPUT_BASE / "comparison"
MLRUNS_DIR = OUTPUT_BASE / "mlruns"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_directories():
    """Create output directories"""
    for dir_path in [MODELS_DIR, METRICS_DIR, COMPARISON_DIR, MLRUNS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✅ Output directories created")


def setup_mlflow():
    """Setup MLflow experiment"""
    if not MLFLOW_AVAILABLE:
        return None
    
    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    experiment_name = f"MindEve_Full_Pipeline_{SAMPLE_SIZE//1_000_000}M"
    mlflow.set_experiment(experiment_name)
    
    print(f"✅ MLflow experiment: {experiment_name}")
    print(f"   Tracking URI: {MLRUNS_DIR}")
    
    return experiment_name


# ============================================================================
# PHASE 1: DATA LOADING & PREPROCESSING
# ============================================================================

def phase_1_load_data(config):
    """Load and preprocess data (from test_gnn.py)"""
    
    print("\n" + "="*80)
    print("📊 PHASE 1: DATA LOADING & PREPROCESSING")
    print("="*80)
    
    print(f"\n1️⃣  Loading {SAMPLE_SIZE:,} samples...")
    df = load_data(config, sample_size=SAMPLE_SIZE)
    print(f"   ✅ Loaded: {len(df):,} rows")
    
    print(f"\n2️⃣  Feature engineering...")
    engineer = FeatureEngineer(config)
    df_processed = engineer.engineer_features(df)
    print(f"   ✅ Features: {len(df_processed.columns)} columns")
    
    return df_processed


# ============================================================================
# PHASE 2: CREATE MULTIPLE FRAUD LABELINGS
# ============================================================================

def phase_2_create_labels(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Create 4 different fraud labeling strategies (simple, no clustering dependencies)"""
    
    print("\n" + "="*80)
    print("🏷️  PHASE 2: CREATING FRAUD LABELS (4 Methods)")
    print("="*80)
    
    labels_dict = {}
    
    # Use discounted_total_price or amount
    if 'discounted_total_price' in df.columns:
        price_col = 'discounted_total_price'
    elif 'amount' in df.columns:
        price_col = 'amount'
    else:
        raise ValueError("No price column found!")
    
    prices = df[price_col].values
    
    # Method 1: 99th percentile
    threshold_99 = np.percentile(prices, 99)
    labels_dict['quantile_99'] = (prices > threshold_99).astype(int)
    print(f"   ✅ Quantile 99%: {labels_dict['quantile_99'].sum():,} fraud ({labels_dict['quantile_99'].mean()*100:.2f}%)")
    
    # Method 2: 95th percentile  
    threshold_95 = np.percentile(prices, 95)
    labels_dict['quantile_95'] = (prices > threshold_95).astype(int)
    print(f"   ✅ Quantile 95%: {labels_dict['quantile_95'].sum():,} fraud ({labels_dict['quantile_95'].mean()*100:.2f}%)")
    
    # Method 3: 90th percentile
    threshold_90 = np.percentile(prices, 90)
    labels_dict['quantile_90'] = (prices > threshold_90).astype(int)
    print(f"   ✅ Quantile 90%: {labels_dict['quantile_90'].sum():,} fraud ({labels_dict['quantile_90'].mean()*100:.2f}%)")
    
    # Method 4: Z-score > 3
    mean_price = prices.mean()
    std_price = prices.std()
    z_scores = (prices - mean_price) / std_price
    labels_dict['z_score'] = (z_scores > 3).astype(int)
    print(f"   ✅ Z-Score > 3: {labels_dict['z_score'].sum():,} fraud ({labels_dict['z_score'].mean()*100:.2f}%)")
    
    return labels_dict


# ============================================================================
# PHASE 3: GRAPH CONSTRUCTION
# ============================================================================

def phase_3_build_graphs(df: pd.DataFrame, labels_dict: Dict[str, np.ndarray], config):
    """Build graphs for each labeling method"""
    
    print("\n" + "="*80)
    print("🗃️  PHASE 3: GRAPH CONSTRUCTION")
    print("="*80)
    
    graphs_dict = {}
    mappings_dict = {}
    
    graph_builder = GraphBuilder(config)
    
    for label_type, fraud_labels in labels_dict.items():
        print(f"\n   Building graph for {label_type.upper()}...")
        
        hetero_data, transaction_mapping = graph_builder.build_graph(df, fraud_labels)
        
        graphs_dict[label_type] = hetero_data
        mappings_dict[label_type] = transaction_mapping
        
        print(f"   ✅ {label_type}: {hetero_data['customer'].num_nodes:,} customers, "
              f"{hetero_data['product'].num_nodes:,} products, "
              f"{hetero_data['store'].num_nodes:,} stores")
    
    return graphs_dict, mappings_dict


# ============================================================================
# PHASE 4: STRATIFIED K-FOLD SPLIT
# ============================================================================

def phase_4_create_folds(labels_dict: Dict[str, np.ndarray]) -> Dict:
    """Create stratified K-fold splits"""
    
    print("\n" + "="*80)
    print("🔀 PHASE 4: STRATIFIED K-FOLD SPLIT")
    print("="*80)
    
    folds_dict = {}
    
    for label_type, fraud_labels in labels_dict.items():
        print(f"\n   Creating folds for {label_type.upper()}...")
        
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        
        indices = np.arange(len(fraud_labels))
        folds = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, fraud_labels)):
            folds.append({
                'fold_id': fold_idx,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'train_fraud_rate': fraud_labels[train_idx].mean(),
                'val_fraud_rate': fraud_labels[val_idx].mean()
            })
            
            print(f"      Fold {fold_idx}: Train={len(train_idx):,} ({fraud_labels[train_idx].mean()*100:.2f}% fraud), "
                  f"Val={len(val_idx):,} ({fraud_labels[val_idx].mean()*100:.2f}% fraud)")
        
        folds_dict[label_type] = folds
    
    return folds_dict


# ============================================================================
# PHASE 5: TRAIN ALL COMBINATIONS
# ============================================================================

def train_single_combination(
    gnn_type: str,
    label_type: str,
    fold_idx: int,
    graph,
    transaction_mapping,
    train_idx,
    val_idx,
    config,
    device
) -> Dict:
    """Train a single GNN model on one fold (from test_gnn.py)"""
    
    # Initialize model
    if gnn_type == 'sage':
        model = GraphSAGE(config).to(device)
    elif gnn_type == 'gat':
        model = GAT(config).to(device)
    elif gnn_type == 'gcn':
        model = GCN(config).to(device)
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")
    
    # Create logger
    import logging
    logger = logging.getLogger(f'{gnn_type}_{label_type}_fold{fold_idx}')
    logger.setLevel(logging.WARNING)  # Reduce verbosity
    
    # Train
    trainer = GNNTrainer(model, config, logger)
    trainer.use_mlflow = False  # We handle MLflow at higher level
    
    history = trainer.train(
        graph=graph,
        transaction_mapping=transaction_mapping,
        train_idx=train_idx,
        val_idx=val_idx
    )
    
    # Evaluate
    evaluator = GNNEvaluator(model, device)
    
    val_data = {
    'customer_idx': transaction_mapping.iloc[val_idx]['customer_idx'].values,
    'product_idx': transaction_mapping.iloc[val_idx]['product_idx'].values,
    'store_idx': transaction_mapping.iloc[val_idx]['store_idx'].values,
    'labels': transaction_mapping.iloc[val_idx]['fraud_label'].values
    }
    
    results = evaluator.evaluate_full(graph, val_data)
    
    return {
        'model': model,
        'history': history,
        'metrics': results
    }


def phase_5_train_all_combinations(
    graphs_dict: Dict,
    mappings_dict: Dict,
    folds_dict: Dict,
    config,
    device
) -> Dict:
    """Train all 12 combinations with cross-validation"""
    
    print("\n" + "="*80)
    print("🏋️  PHASE 5: TRAINING ALL COMBINATIONS")
    print("="*80)
    
    total_combinations = len(GNN_TYPES) * len(CLUSTERING_TYPES)
    print(f"\n   Total combinations: {total_combinations}")
    print(f"   Folds per combination: {CV_FOLDS}")
    print(f"   Total training runs: {total_combinations * CV_FOLDS}")
    
    all_results = {}
    combination_counter = 0
    
    for gnn_type in GNN_TYPES:
        for label_type in CLUSTERING_TYPES:
            combination_counter += 1
            combination_id = f"{gnn_type}_{label_type}"
            
            print(f"\n{'='*80}")
            print(f"🔹 Combination {combination_counter}/{total_combinations}: {combination_id.upper()}")
            print(f"{'='*80}")
            
            # Start MLflow run
            if MLFLOW_AVAILABLE:
                mlflow_run = mlflow.start_run(run_name=combination_id)
                
                mlflow.log_params({
                    "gnn_type": gnn_type,
                    "label_type": label_type,
                    "sample_size": SAMPLE_SIZE,
                    "cv_folds": CV_FOLDS,
                    "hidden_channels": config['architectures'][gnn_type]['hidden_channels'],
                    "num_layers": config['architectures'][gnn_type]['num_layers'],
                    "dropout": config['architectures'][gnn_type]['dropout']
                })
            
            # Get data for this combination
            graph = graphs_dict[label_type]
            mapping = mappings_dict[label_type]
            folds = folds_dict[label_type]
            
            # Cross-validation
            cv_results = []
            fold_models = []
            
            for fold in folds:
                fold_idx = fold['fold_id']
                print(f"\n   📂 Fold {fold_idx + 1}/{CV_FOLDS}")
                
                fold_start = time.time()
                
                # Train
                result = train_single_combination(
                    gnn_type=gnn_type,
                    label_type=label_type,
                    fold_idx=fold_idx,
                    graph=graph.to(device),
                    transaction_mapping=mapping,
                    train_idx=fold['train_idx'],
                    val_idx=fold['val_idx'],
                    config=config,
                    device=device
                )
                
                fold_time = time.time() - fold_start
                
                # Extract metrics
                fold_metrics = {
                    'fold_id': fold_idx,
                    'precision': float(result['metrics']['precision']),
                    'recall': float(result['metrics']['recall']),
                    'f1': float(result['metrics']['f1']),
                    'auc': float(result['metrics']['auc']),
                    'training_time': fold_time,
                    'best_epoch': result['history']['best_epoch']
                }
                
                cv_results.append(fold_metrics)
                fold_models.append(result['model'])
                
                # Log to MLflow
                if MLFLOW_AVAILABLE:
                    mlflow.log_metrics({
                        f"fold_{fold_idx}_f1": fold_metrics['f1'],
                        f"fold_{fold_idx}_recall": fold_metrics['recall'],
                        f"fold_{fold_idx}_precision": fold_metrics['precision'],
                        f"fold_{fold_idx}_auc": fold_metrics['auc']
                    }, step=fold_idx)
                
                # Save fold model
                model_dir = MODELS_DIR / combination_id
                model_dir.mkdir(exist_ok=True, parents=True)
                model_path = model_dir / f"fold_{fold_idx}_model.pt"
                torch.save(result['model'].state_dict(), model_path)
                
                print(f"      ✅ F1: {fold_metrics['f1']:.4f}, Recall: {fold_metrics['recall']:.4f}, "
                      f"Time: {fold_time/60:.1f}min")
            
            # Calculate aggregated metrics
            metrics_array = {
                'f1': [r['f1'] for r in cv_results],
                'recall': [r['recall'] for r in cv_results],
                'precision': [r['precision'] for r in cv_results],
                'auc': [r['auc'] for r in cv_results]
            }
            
            aggregated = {
                'mean_f1': float(np.mean(metrics_array['f1'])),
                'std_f1': float(np.std(metrics_array['f1'])),
                'mean_recall': float(np.mean(metrics_array['recall'])),
                'std_recall': float(np.std(metrics_array['recall'])),
                'mean_precision': float(np.mean(metrics_array['precision'])),
                'std_precision': float(np.std(metrics_array['precision'])),
                'mean_auc': float(np.mean(metrics_array['auc'])),
                'std_auc': float(np.std(metrics_array['auc']))
            }
            
            # Find best fold
            best_fold_idx = np.argmax(metrics_array['f1'])
            best_fold_f1 = metrics_array['f1'][best_fold_idx]
            
            # Save best model
            best_model_path = model_dir / "best_model.pt"
            torch.save(fold_models[best_fold_idx].state_dict(), best_model_path)
            
            print(f"\n   📊 Cross-Validation Results:")
            print(f"      Mean F1:        {aggregated['mean_f1']:.4f} ± {aggregated['std_f1']:.4f}")
            print(f"      Mean Recall:    {aggregated['mean_recall']:.4f} ± {aggregated['std_recall']:.4f}")
            print(f"      Mean Precision: {aggregated['mean_precision']:.4f} ± {aggregated['std_precision']:.4f}")
            print(f"      Mean AUC:       {aggregated['mean_auc']:.4f} ± {aggregated['std_auc']:.4f}")
            print(f"      Best Fold:      Fold {best_fold_idx} (F1: {best_fold_f1:.4f})")
            
            # Log aggregated metrics to MLflow
            if MLFLOW_AVAILABLE:
                mlflow.log_metrics(aggregated)
                mlflow.log_metric("best_fold_id", best_fold_idx)
                mlflow.log_metric("best_fold_f1", best_fold_f1)
                mlflow.pytorch.log_model(fold_models[best_fold_idx], "best_model")
                mlflow.end_run()
            
            # Store results
            all_results[combination_id] = {
                'gnn_type': gnn_type,
                'label_type': label_type,
                'cv_results': cv_results,
                'aggregated_metrics': aggregated,
                'best_fold': {
                    'fold_id': int(best_fold_idx),
                    'f1': float(best_fold_f1),
                    'model_path': str(best_model_path)
                }
            }
            
            # Save combination metrics
            metrics_path = METRICS_DIR / f"{combination_id}_cv_results.json"
            with open(metrics_path, 'w') as f:
                json.dump(all_results[combination_id], f, indent=2)
    
    return all_results


# ============================================================================
# PHASE 6: COMPARISON & VISUALIZATION
# ============================================================================

def phase_6_create_comparisons(all_results: Dict):
    """Create comparison tables and visualizations"""
    
    print("\n" + "="*80)
    print("📊 PHASE 6: COMPARISON & VISUALIZATION")
    print("="*80)
    
    # Create comparison table
    print("\n   1️⃣  Creating comparison table...")
    
    comparison_data = []
    for combination_id, results in all_results.items():
        agg = results['aggregated_metrics']
        comparison_data.append({
            'GNN': results['gnn_type'].upper(),
            'Labeling': results['label_type'].replace('_', ' ').title(),
            'Mean F1': agg['mean_f1'],
            'Std F1': agg['std_f1'],
            'Mean Recall': agg['mean_recall'],
            'Std Recall': agg['std_recall'],
            'Mean Precision': agg['mean_precision'],
            'Std Precision': agg['std_precision'],
            'Mean AUC': agg['mean_auc'],
            'Std AUC': agg['std_auc'],
            'Best Fold F1': results['best_fold']['f1']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Save CSV
    csv_path = COMPARISON_DIR / "cv_comparison_table.csv"
    df_comparison.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"      ✅ Saved: {csv_path}")
    
    # Print table
    print("\n" + "="*80)
    print("📋 CROSS-VALIDATION COMPARISON TABLE")
    print("="*80)
    print(df_comparison.to_string(index=False))
    
    # Create visualizations
    print("\n   2️⃣  Creating visualizations...")
    create_comparison_plots(all_results, df_comparison)
    
    print(f"   ✅ All visualizations saved to: {COMPARISON_DIR}")


def create_comparison_plots(all_results: Dict, df: pd.DataFrame):
    """Create comparison visualizations"""
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # 1. Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    pivot = df.pivot(index='GNN', columns='Labeling', values='Mean F1')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'F1-Score'})
    ax.set_title('Performance Heatmap: GNN × Labeling Method', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✅ performance_heatmap.png")
    
    # 2. Best combinations
    fig, ax = plt.subplots(figsize=(12, 6))
    top5 = df.nlargest(5, 'Mean F1')
    labels = [f"{row['GNN']} + {row['Labeling']}" for _, row in top5.iterrows()]
    values = top5['Mean F1'].values
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, alpha=0.7)
    colors = plt.cm.RdYlGn(np.linspace(0.4, 0.9, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Mean F1-Score')
    ax.set_title('Top-5 Model Combinations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val, bar.get_y() + bar.get_height()/2., f' {val:.4f}',
               ha='left', va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "best_combinations.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✅ best_combinations.png")


def save_summary_report(all_results: Dict, elapsed_time: float):
    """Save comprehensive summary report"""
    
    summary = {
        "experiment_info": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sample_size": SAMPLE_SIZE,
            "cv_folds": CV_FOLDS,
            "total_combinations": len(GNN_TYPES) * len(CLUSTERING_TYPES),
            "total_time_hours": elapsed_time / 3600
        },
        "gnn_types": GNN_TYPES,
        "labeling_types": CLUSTERING_TYPES,
        "results": all_results
    }
    
    summary_path = OUTPUT_BASE / "summary_report.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Summary saved: {summary_path}")
    return summary


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline execution"""
    
    print("\n" + "="*80)
    print("🚀 FULL PIPELINE: COMPREHENSIVE GNN FRAUD DETECTION")
    print("="*80)
    print(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Sample size: {SAMPLE_SIZE:,}")
    print(f"🔀 CV folds: {CV_FOLDS}")
    print(f"🤖 GNN types: {', '.join([g.upper() for g in GNN_TYPES])}")
    print(f"🏷️  Labeling types: {', '.join(CLUSTERING_TYPES)}")
    print(f"🎯 Total combinations: {len(GNN_TYPES) * len(CLUSTERING_TYPES)}")
    
    pipeline_start = time.time()
    
    try:
        # Setup
        print("\n" + "="*80)
        print("⚙️  SETUP")
        print("="*80)
        
        setup_directories()
        
        config = load_all_configs()
        set_seed(config['project']['random_seed'])
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Device: {device}")
        
        # Test mode override
        config['test_mode'] = {
            'enabled': False,
            'sample_size': SAMPLE_SIZE
        }
        
        # Setup MLflow
        experiment_name = setup_mlflow()
        
        # Phase 1: Data
        df_processed = phase_1_load_data(config)
        
        # Phase 2: Labels (4 simple methods, no clustering dependencies)
        labels_dict = phase_2_create_labels(df_processed)
        
        # Phase 3: Graphs
        graphs_dict, mappings_dict = phase_3_build_graphs(df_processed, labels_dict, config)
        
        # Phase 4: Folds
        folds_dict = phase_4_create_folds(labels_dict)
        
        # Phase 5: Train all combinations
        all_results = phase_5_train_all_combinations(
            graphs_dict, mappings_dict, folds_dict, config, device
        )
        
        # Phase 6: Comparisons
        phase_6_create_comparisons(all_results)
        
        # Save summary
        elapsed_time = time.time() - pipeline_start
        summary = save_summary_report(all_results, elapsed_time)
        
        # Final summary
        print("\n" + "="*80)
        print("✅ FULL PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print(f"\n⏱️  Total time: {elapsed_time/3600:.2f} hours")
        print(f"\n📊 Results Summary:")
        
        # Find best combination
        best_combination = max(
            all_results.items(),
            key=lambda x: x[1]['aggregated_metrics']['mean_f1']
        )
        
        best_id = best_combination[0]
        best_data = best_combination[1]
        
        print(f"\n🏆 Best Combination: {best_id.upper()}")
        print(f"   Mean F1:     {best_data['aggregated_metrics']['mean_f1']:.4f} ± {best_data['aggregated_metrics']['std_f1']:.4f}")
        print(f"   Mean Recall: {best_data['aggregated_metrics']['mean_recall']:.4f} ± {best_data['aggregated_metrics']['std_recall']:.4f}")
        print(f"   Mean AUC:    {best_data['aggregated_metrics']['mean_auc']:.4f} ± {best_data['aggregated_metrics']['std_auc']:.4f}")
        
        print(f"\n📂 Outputs:")
        print(f"   Models: {MODELS_DIR}")
        print(f"   Metrics: {METRICS_DIR}")
        print(f"   Comparisons: {COMPARISON_DIR}")
        print(f"   Summary: {OUTPUT_BASE / 'summary_report.json'}")
        
        if MLFLOW_AVAILABLE:
            print(f"   MLflow: {MLRUNS_DIR}")
            print(f"\n💡 View results: mlflow ui --backend-store-uri {MLRUNS_DIR}")
        
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