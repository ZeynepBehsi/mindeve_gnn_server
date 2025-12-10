"""
Full Research Test: GraphSAGE + 4 Clustering Methods
Comparison and Visualization
FIXED VERSION - Compatible with project structure
"""

import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import json
warnings.filterwarnings('ignore')

# Import project modules
from src.utils.config_loader import load_all_configs
from src.utils.seed import set_seed
from src.data.loader import load_data
from src.data.preprocessor import FeatureEngineer
from src.labeling.clustering import ClusteringExperiment
# ❌ ClusteringVisualizer import'u kaldırdık - config problemi var
from src.models.graph_builder import GraphBuilder
from src.models.gnn_models import HeteroGNN
from src.training.trainer import GNNTrainer
from src.training.evaluator import GNNEvaluator


def main():
    print("\n" + "="*80)
    print("🔬 RESEARCH PIPELINE: GraphSAGE + Clustering Methods")
    print("="*80)
    print(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # ========================
        # 1. SETUP
        # ========================
        print("\n" + "="*80)
        print("1️⃣  CONFIGURATION")
        print("="*80)
        
        config = load_all_configs()
        set_seed(config['project']['random_seed'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Simple logger setup
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger('research_test')
        
        print(f"✅ Device: {device}")
        print(f"✅ Random seed: {config['project']['random_seed']}")
        
        # Test mode override
        config['test_mode'] = {
            'enabled': True,
            'sample_size':  5000000
        }
        
        # ========================
        # 2. LOAD & PROCESS DATA
        # ========================
        print("\n" + "="*80)
        print("2️⃣  LOADING DATA")
        print("="*80)
        
        df = load_data(config, sample_size= 5000000)
        print(f"✅ Loaded: {len(df):,} rows")
        
        print("\n" + "="*80)
        print("3️⃣  FEATURE ENGINEERING")
        print("="*80)
        
        engineer = FeatureEngineer(config)
        df_processed = engineer.engineer_features(df)
        print(f"✅ Features: {len(df_processed.columns)}")
        
        # ========================
        # 3. CLUSTERING EXPERIMENTS
        # ========================
        print("\n" + "="*80)
        print("4️⃣  CLUSTERING EXPERIMENTS")
        print("="*80)

        clustering_exp = ClusteringExperiment(config, test_mode=False)
        clustering_results = clustering_exp.run_all(df_processed)

        fraud_labels_ensemble, fraud_scores = clustering_exp.create_ensemble()

        print(f"\n✅ Clustering experiments: {len(clustering_results)}")
        print(f"✅ Ensemble fraud rate: {fraud_labels_ensemble.mean()*100:.2f}%")

        # ========================
        # ✅ YENİ: CLUSTERING VISUALIZATIONS
        # ========================
        print("\n" + "="*80)
        print("🎨 CREATING CLUSTERING VISUALIZATIONS")
        print("="*80)

        # Import visualizer
        from src.labeling.visualizer import ClusteringVisualizer

        # Visualization config (add to config if not exists)
        if 'visualization' not in config:
            config['visualization'] = {
                'enabled': True,
                'save_format': 'png',
                'dpi': 300,
                'pca': {
                    'n_components': 2
                },
                'tsne': {
                    'n_components': 2,
                    'perplexity': 30,
                    'n_iter': 1000,
                    'random_state': 42,
                    'max_samples_test': 5000
                },
                'umap': {
                    'n_components': 2,
                    'n_neighbors': 15,
                    'min_dist': 0.1,
                    'random_state': 42
                },
                'silhouette': {
                    'sample_size': 10000
                }
            }

        # Create visualizer
        visualizer = ClusteringVisualizer(config)

        # Get best result per algorithm
        best_by_algo = {}
        for algo_name in ['KMeans', 'DBSCAN', 'IsolationForest', 'GMM']:
            algo_results = [r for r in clustering_results 
                        if r['algorithm'].startswith(algo_name)]
            if algo_results:
                best = max(algo_results, key=lambda x: x['metrics']['silhouette'])
                best_by_algo[algo_name] = best
                print(f"  Found best {algo_name}: silhouette={best['metrics']['silhouette']:.3f}")

        # Prepare features for visualization
        X_scaled, _ = clustering_exp.prepare_features(df_processed)

        # Create individual plots for each algorithm
        viz_output_dir = 'outputs/figures/clustering'
        Path(viz_output_dir).mkdir(parents=True, exist_ok=True)

        for algo_name, result in best_by_algo.items():
            print(f"\n  Creating visualizations for {algo_name}...")
            try:
                visualizer.plot_all(
                    X=X_scaled,
                    labels=result['labels'],
                    fraud_mask=result['fraud_mask'],
                    algorithm_name=result['algorithm'],
                    save_dir=viz_output_dir
                )
                print(f"    ✅ {algo_name} visualizations saved")
            except Exception as e:
                print(f"    ❌ Error visualizing {algo_name}: {e}")

        # Create comparison summary plot
        print(f"\n  Creating comparison summary...")
        try:
            visualizer.plot_comparison_summary(
                clustering_results,
                viz_output_dir
            )
            print(f"    ✅ Comparison summary saved")
        except Exception as e:
            print(f"    ❌ Error creating comparison: {e}")

        print(f"\n✅ All clustering visualizations saved to: {viz_output_dir}/")
        
        # ========================
        # CLUSTERING VISUALIZATION (SIMPLIFIED)
        # ========================
        print("\n" + "="*80)
        print("📊 SAVING CLUSTERING RESULTS")
        print("="*80)
        
        # Get best result per algorithm
        best_by_algo = {}
        for algo_name in ['KMeans', 'DBSCAN', 'IsolationForest', 'GMM']:
            algo_results = [r for r in clustering_results 
                          if r['algorithm'].startswith(algo_name)]
            if algo_results:
                best = max(algo_results, key=lambda x: x['metrics']['silhouette'])
                best_by_algo[algo_name] = best
                print(f"  {algo_name:20s}: silhouette={best['metrics']['silhouette']:.3f}, "
                      f"fraud_rate={best['metrics']['fraud_rate']*100:.2f}%")
        
        # Save clustering comparison to CSV (no visualization to avoid config issues)
        clustering_output_dir = Path('outputs/clustering_results')
        clustering_output_dir.mkdir(parents=True, exist_ok=True)
        
        clustering_comparison = []
        for algo_name, result in best_by_algo.items():
            clustering_comparison.append({
                'Algorithm': algo_name,
                'Silhouette': result['metrics']['silhouette'],
                'Davies-Bouldin': result['metrics'].get('davies_bouldin', 0),
                'Fraud Rate (%)': result['metrics']['fraud_rate'] * 100,
                'Config': str(result['params'])
            })
        
        df_clustering = pd.DataFrame(clustering_comparison)
        df_clustering.to_csv(clustering_output_dir / 'clustering_comparison.csv', index=False)
        print(f"✅ Clustering comparison saved: {clustering_output_dir / 'clustering_comparison.csv'}")
        
        # ========================
        # 4. TRAIN GRAPHSAGE WITH EACH CLUSTERING METHOD
        # ========================
        print("\n" + "="*80)
        print("5️⃣  TRAINING GraphSAGE WITH EACH CLUSTERING METHOD")
        print("="*80)
        
        all_model_results = {}
        
        # Prepare labeling methods
        labeling_methods = {
            'Ensemble': fraud_labels_ensemble
        }
        
        # Add best method from each algorithm
        for algo_name, result in best_by_algo.items():
            labeling_methods[f"{algo_name}_best"] = result['fraud_mask']
        
        # Train a model for each labeling method
        for method_name, fraud_labels in labeling_methods.items():
            print(f"\n{'='*60}")
            print(f"🔹 Method: {method_name}")
            print(f"{'='*60}")
            
            fraud_rate = fraud_labels.mean()
            fraud_count = fraud_labels.sum()
            print(f"  Fraud cases: {fraud_count:,}")
            print(f"  Fraud rate: {fraud_rate*100:.2f}%")
            
            # Build graph
            print("  Building graph...")
            graph_builder = GraphBuilder(config)
            hetero_data, transaction_mapping = graph_builder.build_graph(
                df_processed, 
                fraud_labels
            )
            
            print(f"  Graph built: {hetero_data['customer'].num_nodes:,} customers, "
                  f"{hetero_data['product'].num_nodes:,} products, "
                  f"{hetero_data['store'].num_nodes:,} stores")
            
            # Train/val/test split
            num_trans = len(transaction_mapping)
            indices = np.arange(num_trans)
            np.random.shuffle(indices)
            
            train_size = int(0.70 * num_trans)
            val_size = int(0.15 * num_trans)
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size+val_size]
            test_idx = indices[train_size+val_size:]
            
            print(f"  Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}")
            
            # ✅ FIX: Use .iloc for DataFrame indexing
            train_labels = transaction_mapping['fraud_label'].iloc[train_idx].values
            val_labels = transaction_mapping['fraud_label'].iloc[val_idx].values
            test_labels = transaction_mapping['fraud_label'].iloc[test_idx].values
            
            print(f"  Train fraud: {train_labels.mean()*100:.2f}%")
            print(f"  Val fraud: {val_labels.mean()*100:.2f}%")
            print(f"  Test fraud: {test_labels.mean()*100:.2f}%")
            
            # Create model (same as test_gnn.py)
            model = HeteroGNN(config, conv_type='sage').to(device)
            hetero_data = hetero_data.to(device)
            
            print(f"  Model: HeteroGNN (GraphSAGE)")
            print(f"    Hidden channels: {model.hidden_channels}")
            print(f"    Num layers: {model.num_layers}")
            
            # Train
            print(f"  Training...")
            trainer = GNNTrainer(model, config, logger)
            
            history = trainer.train(
                graph=hetero_data,
                transaction_mapping=transaction_mapping,
                train_idx=train_idx,
                val_idx=val_idx
            )
            
            print(f"  ✅ Training completed!")
            print(f"    Best epoch: {history['best_epoch']}")
            print(f"    Best val loss: {history['best_val_loss']:.4f}")
            
            # Evaluate (same as test_gnn.py)
            print(f"  Evaluating...")
            evaluator = GNNEvaluator(model, device)
            
            # ✅ FIX: Use .iloc and .values
            test_data = {
                'customer_idx': transaction_mapping['customer_idx'].iloc[test_idx].values,
                'product_idx': transaction_mapping['product_idx'].iloc[test_idx].values,
                'store_idx': transaction_mapping['store_idx'].iloc[test_idx].values,
                'labels': transaction_mapping['fraud_label'].iloc[test_idx].values
            }
            
            results = evaluator.evaluate_full(hetero_data, test_data)
            
            print(f"  ✅ Evaluation completed!")
            print(f"    Precision: {results['precision']:.4f}")
            print(f"    Recall: {results['recall']:.4f}")
            print(f"    F1: {results['f1']:.4f}")
            print(f"    AUC: {results['auc']:.4f}")
            
            # Store results
            all_model_results[method_name] = {
                'clustering_method': method_name,
                'fraud_rate': float(fraud_rate),
                'metrics': {
                    'precision': float(results['precision']),
                    'recall': float(results['recall']),
                    'f1': float(results['f1']),
                    'auc': float(results['auc'])
                },
                'top_k': {k: float(v) for k, v in results['top_k'].items()} if results.get('top_k') else {},
                'confusion_matrix': results['confusion_matrix'].tolist(),
                'training_history': {
                    'best_epoch': history['best_epoch'],
                    'best_val_loss': history['best_val_loss']
                }
            }
            
            # Save individual model
            models_dir = Path('outputs/models')
            models_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = models_dir / f"graphsage_{method_name.lower().replace(' ', '_')}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'method': method_name,
                'fraud_rate': fraud_rate,
                'metrics': all_model_results[method_name]['metrics'],
                'config': config
            }, model_path)
            print(f"  💾 Model saved: {model_path}")
        
        # ========================
        # 5. COMPARISON & RESULTS
        # ========================
        print("\n" + "="*80)
        print("6️⃣  COMPARISON OF ALL METHODS")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_data = []
        for method_name, result in all_model_results.items():
            comparison_data.append({
                'Method': method_name,
                'Fraud Rate (%)': result['fraud_rate'] * 100,
                'Precision': result['metrics']['precision'],
                'Recall': result['metrics']['recall'],
                'F1-Score': result['metrics']['f1'],
                'AUC-ROC': result['metrics']['auc'],
                'Top-100': result['top_k'].get('top_100', 0),
                'Top-500': result['top_k'].get('top_500', 0),
                'Best Epoch': result['training_history']['best_epoch']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Save results
        output_dir = Path('outputs/research_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df_comparison.to_csv(output_dir / 'methods_comparison.csv', index=False)
        
        with open(output_dir / 'full_results.json', 'w') as f:
            json.dump(all_model_results, f, indent=2)
        
        # Print comparison table
        print("\n📊 COMPARISON TABLE:")
        print(df_comparison.to_string(index=False))
        
        # Find best methods
        best_f1_idx = df_comparison['F1-Score'].idxmax()
        best_auc_idx = df_comparison['AUC-ROC'].idxmax()
        best_precision_idx = df_comparison['Precision'].idxmax()
        best_recall_idx = df_comparison['Recall'].idxmax()
        
        print(f"\n🏆 BEST RESULTS:")
        print(f"  Best F1-Score:  {df_comparison.iloc[best_f1_idx]['Method']:20s} ({df_comparison.iloc[best_f1_idx]['F1-Score']:.4f})")
        print(f"  Best AUC-ROC:   {df_comparison.iloc[best_auc_idx]['Method']:20s} ({df_comparison.iloc[best_auc_idx]['AUC-ROC']:.4f})")
        print(f"  Best Precision: {df_comparison.iloc[best_precision_idx]['Method']:20s} ({df_comparison.iloc[best_precision_idx]['Precision']:.4f})")
        print(f"  Best Recall:    {df_comparison.iloc[best_recall_idx]['Method']:20s} ({df_comparison.iloc[best_recall_idx]['Recall']:.4f})")
        
        # Create simple comparison plot
        create_comparison_plots(df_comparison, output_dir)
        
        print(f"\n✅ ALL RESULTS SAVED:")
        print(f"  📁 {output_dir}")
        print(f"    - methods_comparison.csv")
        print(f"    - full_results.json")
        print(f"    - comparison_plots.png")
        print(f"  📁 {clustering_output_dir}")
        print(f"    - clustering_comparison.csv")
        print(f"  📁 {models_dir}")
        print(f"    - graphsage_*.pt (5 models)")
        
        print(f"\n⏰ End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
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


def create_comparison_plots(df_comparison, output_dir):
    """Create comparison visualization plots"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    try:
        sns.set_style('whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('GraphSAGE + Clustering Methods Comparison', fontsize=16, fontweight='bold')
        
        methods = df_comparison['Method']
        
        # 1. F1-Score comparison
        axes[0, 0].barh(methods, df_comparison['F1-Score'], color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('F1-Score')
        axes[0, 0].set_title('F1-Score by Clustering Method')
        axes[0, 0].grid(True, alpha=0.3)
        for i, v in enumerate(df_comparison['F1-Score']):
            axes[0, 0].text(v, i, f' {v:.3f}', va='center')
        
        # 2. Precision vs Recall
        axes[0, 1].scatter(df_comparison['Recall'], df_comparison['Precision'], 
                          s=200, alpha=0.6, c=range(len(methods)), cmap='viridis')
        for i, method in enumerate(methods):
            axes[0, 1].annotate(method, 
                              (df_comparison.iloc[i]['Recall'], df_comparison.iloc[i]['Precision']),
                              fontsize=8, ha='right')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision vs Recall')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. AUC-ROC comparison
        axes[1, 0].barh(methods, df_comparison['AUC-ROC'], color='coral', alpha=0.7)
        axes[1, 0].set_xlabel('AUC-ROC')
        axes[1, 0].set_title('AUC-ROC by Clustering Method')
        axes[1, 0].grid(True, alpha=0.3)
        for i, v in enumerate(df_comparison['AUC-ROC']):
            axes[1, 0].text(v, i, f' {v:.3f}', va='center')
        
        # 4. All Metrics Comparison (Heatmap)
        metrics_for_heatmap = df_comparison[['Precision', 'Recall', 'F1-Score', 'AUC-ROC']].T
        metrics_for_heatmap.columns = methods
        
        im = axes[1, 1].imshow(metrics_for_heatmap.values, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
        axes[1, 1].set_xticks(np.arange(len(methods)))
        axes[1, 1].set_yticks(np.arange(len(metrics_for_heatmap)))
        axes[1, 1].set_xticklabels(methods, rotation=45, ha='right')
        axes[1, 1].set_yticklabels(metrics_for_heatmap.index)
        axes[1, 1].set_title('All Metrics Heatmap')
        
        # Add values to heatmap
        for i in range(len(metrics_for_heatmap)):
            for j in range(len(methods)):
                text = axes[1, 1].text(j, i, f'{metrics_for_heatmap.iloc[i, j]:.3f}',
                                     ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Comparison plots created")
        
    except Exception as e:
        print(f"⚠️  Plot creation failed: {e}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)