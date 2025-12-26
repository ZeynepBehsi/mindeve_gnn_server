"""
Complete Clustering Evaluation Script
- Internal metrics only (no ground truth needed)
- t-SNE + PCA visualizations
- Comprehensive CSV reports
- Compatible with src/ project structure

Author: Claude + Zeynep
Date: 2025-12-11
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Project imports - src/ structure
from src.utils.config_loader import load_all_configs
from src.data.loader import load_data
from src.data.preprocessor import FeatureEngineer
from src.labeling.clustering import ClusteringExperiment


# ============================================================================
# SEED FUNCTION (torch-free for clustering)
# ============================================================================

def set_seed_simple(seed: int = 42):
    """Set random seeds without torch (clustering doesn't need it)"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    print(f"✅ Random seed set to {seed}")


# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLE_SIZE = 1000000 # 1M veri olarak ayarladım
OUTPUT_DIR = Path("outputs/clustering_evaluation")
TSNE_SAMPLE_SIZE = 5000


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_inputs(df: pd.DataFrame, X: np.ndarray, clustering_results: list):
    """Validate all inputs before processing"""
    print("\n" + "="*80)
    print("🔍 VALIDATING INPUTS")
    print("="*80)
    
    if df is None or len(df) == 0:
        raise ValueError("❌ DataFrame is empty!")
    print(f"✅ DataFrame: {len(df):,} rows, {len(df.columns)} columns")
    
    if X is None or len(X) == 0:
        raise ValueError("❌ Feature matrix is empty!")
    if X.shape[0] != len(df):
        raise ValueError(f"❌ Shape mismatch: X has {X.shape[0]} rows, df has {len(df)} rows")
    print(f"✅ Feature matrix: {X.shape}")
    
    if not clustering_results or len(clustering_results) == 0:
        raise ValueError("❌ No clustering results found!")
    print(f"✅ Clustering results: {len(clustering_results)} experiments")
    
    required_keys = ['algorithm', 'params', 'labels', 'fraud_mask', 'metrics']
    for i, result in enumerate(clustering_results):
        missing = [key for key in required_keys if key not in result]
        if missing:
            raise ValueError(f"❌ Result {i} missing keys: {missing}")
    
    print("✅ All validations passed!")


def setup_output_directory(output_dir: Path):
    """Create output directory structure"""
    (output_dir / "tsne_plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "pca_plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    
    print(f"\n✅ Output directories created:")
    print(f"   {output_dir}/")
    print(f"   ├── tsne_plots/")
    print(f"   ├── pca_plots/")
    print(f"   └── reports/")


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_clustering_internal(clustering_results: list, X: np.ndarray) -> pd.DataFrame:
    """Evaluate clustering with internal metrics only"""
    print("\n" + "="*80)
    print("📊 CLUSTERING EVALUATION - INTERNAL METRICS")
    print("="*80)
    
    evaluation_results = []
    
    for idx, result in enumerate(clustering_results):
        method = result['algorithm']
        params = result['params']
        labels = result['labels']
        fraud_mask = result['fraud_mask']
        
        print(f"\n{'─'*70}")
        print(f"📌 [{idx+1}/{len(clustering_results)}] Method: {method}")
        print(f"   Params: {params}")
        print(f"{'─'*70}")
        
        try:
            # Silhouette Score
            try:
                if -1 in labels:
                    mask = labels != -1
                    if mask.sum() > 100:
                        silhouette = silhouette_score(X[mask], labels[mask])
                    else:
                        silhouette = 0.0
                else:
                    silhouette = silhouette_score(X, labels)
            except:
                silhouette = 0.0
            
            # Davies-Bouldin Index
            try:
                if -1 in labels:
                    mask = labels != -1
                    if mask.sum() > 100:
                        davies_bouldin = davies_bouldin_score(X[mask], labels[mask])
                    else:
                        davies_bouldin = 999.0
                else:
                    davies_bouldin = davies_bouldin_score(X, labels)
            except:
                davies_bouldin = 999.0
            
            # Calinski-Harabasz Score
            try:
                if -1 in labels:
                    mask = labels != -1
                    if mask.sum() > 100:
                        calinski = calinski_harabasz_score(X[mask], labels[mask])
                    else:
                        calinski = 0.0
                else:
                    calinski = calinski_harabasz_score(X, labels)
            except:
                calinski = 0.0
            
            # Fraud statistics
            fraud_rate = fraud_mask.mean()
            fraud_count = fraud_mask.sum()
            normal_count = len(fraud_mask) - fraud_count
            
            # Cluster statistics
            unique_labels = np.unique(labels[labels >= 0])
            n_clusters = len(unique_labels)
            
            if n_clusters > 0:
                cluster_sizes = [np.sum(labels == label) for label in unique_labels]
                min_cluster_size = min(cluster_sizes)
                max_cluster_size = max(cluster_sizes)
                avg_cluster_size = np.mean(cluster_sizes)
                std_cluster_size = np.std(cluster_sizes)
            else:
                min_cluster_size = max_cluster_size = avg_cluster_size = std_cluster_size = 0
            
            n_outliers = np.sum(labels == -1)
            outlier_rate = n_outliers / len(labels) if len(labels) > 0 else 0
            
            # Validity checks
            fraud_rate_valid = 0.07 <= fraud_rate <= 0.15
            silhouette_valid = silhouette > 0.1
            davies_bouldin_valid = davies_bouldin < 2.0
            
            # Composite score
            silhouette_norm = max(0, min(1, (silhouette + 1) / 2))
            davies_bouldin_norm = 1 / (1 + davies_bouldin)
            calinski_norm = min(calinski / 1000, 1.0)
            
            if fraud_rate < 0.07:
                fraud_rate_score = fraud_rate / 0.07
            elif fraud_rate > 0.15:
                fraud_rate_score = 0.15 / fraud_rate
            else:
                fraud_rate_score = 1.0
            
            composite_score = (
                silhouette_norm * 0.35 +
                davies_bouldin_norm * 0.25 +
                calinski_norm * 0.20 +
                fraud_rate_score * 0.20
            )
            
            # Print results
            print(f"\n  📐 Internal Quality Metrics:")
            print(f"    Silhouette Score:      {silhouette:>8.4f}  {'✅' if silhouette > 0.3 else '⚠️' if silhouette > 0.1 else '❌'}")
            print(f"    Davies-Bouldin Index:  {davies_bouldin:>8.4f}  {'✅' if davies_bouldin < 1.0 else '⚠️' if davies_bouldin < 2.0 else '❌'}")
            print(f"    Calinski-Harabasz:     {calinski:>8.0f}  {'✅' if calinski > 100 else '⚠️' if calinski > 50 else '❌'}")
            
            print(f"\n  🎯 Fraud Labeling Statistics:")
            print(f"    Fraud Count:           {fraud_count:>8,}")
            print(f"    Fraud Rate:            {fraud_rate*100:>7.2f}%  {'✅' if fraud_rate_valid else '❌'}")
            
            print(f"\n  📊 Cluster Distribution:")
            print(f"    Num Clusters:          {n_clusters:>8}")
            if n_clusters > 0:
                print(f"    Min Cluster Size:      {min_cluster_size:>8,}")
                print(f"    Max Cluster Size:      {max_cluster_size:>8,}")
                print(f"    Avg Cluster Size:      {avg_cluster_size:>8,.0f}")
            
            if n_outliers > 0:
                print(f"    Outliers:              {n_outliers:>8,} ({outlier_rate*100:.2f}%)")
            
            print(f"\n  🏆 Composite Score:      {composite_score:>8.4f}")
            
            # Store results
            evaluation_results.append({
                'method': method,
                'params': str(params),
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'calinski_harabasz': calinski,
                'fraud_rate': fraud_rate * 100,
                'fraud_count': fraud_count,
                'normal_count': normal_count,
                'n_clusters': n_clusters,
                'min_cluster_size': min_cluster_size,
                'max_cluster_size': max_cluster_size,
                'avg_cluster_size': avg_cluster_size,
                'n_outliers': n_outliers,
                'outlier_rate': outlier_rate * 100,
                'composite_score': composite_score,
                'silhouette_valid': silhouette_valid,
                'davies_bouldin_valid': davies_bouldin_valid,
                'fraud_rate_valid': fraud_rate_valid
            })
            
        except Exception as e:
            print(f"\n  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return pd.DataFrame(evaluation_results)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_tsne_visualization(X, labels, fraud_mask, method_name, output_dir, sample_size=5000):
    """Create t-SNE visualization"""
    print(f"\n  🎨 Creating t-SNE for {method_name}...")
    
    try:
        n_samples = len(X)
        if n_samples > sample_size:
            sample_idx = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[sample_idx]
            labels_sample = labels[sample_idx]
            fraud_sample = fraud_mask[sample_idx]
            print(f"     Sampling: {n_samples:,} → {sample_size:,}")
        else:
            X_sample, labels_sample, fraud_sample = X, labels, fraud_mask
        
        print(f"     Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, n_jobs=-1)
        X_tsne = tsne.fit_transform(X_sample)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f't-SNE: {method_name}', fontsize=14, fontweight='bold')
        
        # Plot 1: Cluster labels
        mask_valid = labels_sample >= 0
        if mask_valid.sum() > 0:
            unique_labels = np.unique(labels_sample[mask_valid])
            n_clusters = len(unique_labels)
            
            if n_clusters > 1:
                colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
                for i, label in enumerate(unique_labels):
                    mask = labels_sample == label
                    axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                                  c=[colors[i]], label=f'Cluster {label}',
                                  s=20, alpha=0.6, edgecolors='none')
            else:
                axes[0].scatter(X_tsne[mask_valid, 0], X_tsne[mask_valid, 1],
                              c='steelblue', s=20, alpha=0.6)
        
        if np.any(labels_sample == -1):
            mask_outliers = labels_sample == -1
            axes[0].scatter(X_tsne[mask_outliers, 0], X_tsne[mask_outliers, 1],
                          c='red', marker='x', label='Outliers', s=30, alpha=0.8)
        
        axes[0].set_title('By Cluster Labels')
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        axes[0].legend(loc='upper right', fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Fraud labels
        axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=fraud_sample,
                       cmap='RdYlGn_r', s=20, alpha=0.6, edgecolors='none')
        axes[1].set_title('By Fraud Labels')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label='Normal'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                      markersize=10, label='Fraud')
        ]
        axes[1].legend(handles=handles, loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        safe_name = method_name.replace(' ', '_').replace('/', '_').lower()
        filepath = output_dir / "tsne_plots" / f"{safe_name}_tsne.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     ✅ Saved: {filepath.name}")
        
    except Exception as e:
        print(f"     ❌ t-SNE failed: {e}")


def create_comparison_plots(df_results, output_dir):
    """Create comparison plots"""
    print("\n" + "="*80)
    print("📊 CREATING COMPARISON PLOTS")
    print("="*80)
    
    try:
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Clustering Methods Comparison', fontsize=16, fontweight='bold')
        
        methods = df_results['method'].values
        n_methods = len(methods)
        colors = plt.cm.viridis(np.linspace(0, 1, n_methods))
        
        # 1. Composite Score
        axes[0, 0].barh(range(n_methods), df_results['composite_score'], color=colors, alpha=0.7)
        axes[0, 0].set_yticks(range(n_methods))
        axes[0, 0].set_yticklabels(methods, fontsize=8)
        axes[0, 0].set_xlabel('Composite Score')
        axes[0, 0].set_title('🏆 Overall Ranking')
        axes[0, 0].invert_yaxis()
        for i, v in enumerate(df_results['composite_score']):
            axes[0, 0].text(v, i, f' {v:.3f}', va='center', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # 2. Silhouette Score
        axes[0, 1].barh(range(n_methods), df_results['silhouette'], color=colors, alpha=0.7)
        axes[0, 1].set_yticks(range(n_methods))
        axes[0, 1].set_yticklabels(methods, fontsize=8)
        axes[0, 1].set_xlabel('Silhouette Score')
        axes[0, 1].set_title('📐 Cluster Separation')
        axes[0, 1].axvline(x=0.3, color='green', linestyle='--', alpha=0.5, label='Good (>0.3)')
        axes[0, 1].legend(fontsize=7)
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # 3. Fraud Rate
        axes[1, 0].barh(range(n_methods), df_results['fraud_rate'], color=colors, alpha=0.7)
        axes[1, 0].set_yticks(range(n_methods))
        axes[1, 0].set_yticklabels(methods, fontsize=8)
        axes[1, 0].set_xlabel('Fraud Rate (%)')
        axes[1, 0].set_title('🎯 Fraud Labeling Rate')
        axes[1, 0].axvspan(7, 15, alpha=0.2, color='green', label='Ideal (7-15%)')
        axes[1, 0].legend(fontsize=7)
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # 4. All metrics normalized
        x = np.arange(len(methods))
        width = 0.25
        
        metrics_norm = df_results[['silhouette', 'davies_bouldin', 'fraud_rate']].copy()
        metrics_norm['silhouette'] = (metrics_norm['silhouette'] + 1) / 2
        metrics_norm['davies_bouldin'] = 1 / (1 + metrics_norm['davies_bouldin'])
        
        fraud_rates = metrics_norm['fraud_rate'].values
        fraud_score = []
        for fr in fraud_rates:
            if fr < 7:
                fraud_score.append(fr / 7)
            elif fr > 15:
                fraud_score.append(15 / fr)
            else:
                fraud_score.append(1.0)
        
        axes[1, 1].barh(x - width, metrics_norm['silhouette'], width, 
                       label='Silhouette', alpha=0.8)
        axes[1, 1].barh(x, metrics_norm['davies_bouldin'], width,
                       label='Davies-Bouldin', alpha=0.8)
        axes[1, 1].barh(x + width, fraud_score, width,
                       label='Fraud Rate', alpha=0.8)
        
        axes[1, 1].set_yticks(x)
        axes[1, 1].set_yticklabels(methods, fontsize=8)
        axes[1, 1].set_xlabel('Normalized Score (0-1)')
        axes[1, 1].set_title('📈 All Metrics')
        axes[1, 1].legend(fontsize=7)
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        filepath = output_dir / "reports" / "comparison_plots.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {filepath.name}")
        
    except Exception as e:
        print(f"❌ Comparison plots failed: {e}")


def create_ranking_report(df_results, output_dir):
    """Create ranking report"""
    print("\n" + "="*80)
    print("📋 CLUSTERING METHODS RANKING")
    print("="*80)
    
    df_sorted = df_results.sort_values('composite_score', ascending=False)
    
    print("\n" + df_sorted.to_string(index=False))
    
    # Save CSVs
    csv_detailed = output_dir / "reports" / "clustering_ranking_detailed.csv"
    df_sorted.to_csv(csv_detailed, index=False)
    print(f"\n✅ Saved: {csv_detailed.name}")
    
    summary_cols = ['method', 'composite_score', 'silhouette', 'davies_bouldin',
                    'fraud_rate', 'n_clusters', 'fraud_count']
    df_summary = df_sorted[summary_cols]
    
    csv_summary = output_dir / "reports" / "clustering_ranking_summary.csv"
    df_summary.to_csv(csv_summary, index=False)
    print(f"✅ Saved: {csv_summary.name}")
    
    # Winner
    print("\n" + "="*80)
    print("🏆 BEST CLUSTERING METHOD")
    print("="*80)
    
    best = df_sorted.iloc[0]
    print(f"\n  Method: {best['method']}")
    print(f"  Composite Score: {best['composite_score']:.4f}")
    print(f"\n  📐 Quality Metrics:")
    print(f"    Silhouette:       {best['silhouette']:>8.4f}")
    print(f"    Davies-Bouldin:   {best['davies_bouldin']:>8.4f}")
    print(f"    Calinski-Harabasz:{best['calinski_harabasz']:>8.0f}")
    print(f"\n  🎯 Fraud Labeling:")
    print(f"    Fraud Rate:  {best['fraud_rate']:>7.2f}%")
    print(f"    Fraud Count: {best['fraud_count']:>8,}")
    
    # Top 3
    print("\n" + "="*80)
    print("🥇🥈🥉 TOP 3 METHODS")
    print("="*80)
    
    for i, (idx, row) in enumerate(df_sorted.head(3).iterrows(), 1):
        medal = ['🥇', '🥈', '🥉'][i-1]
        print(f"\n{medal} #{i}: {row['method']}")
        print(f"    Score: {row['composite_score']:.4f} | Fraud: {row['fraud_rate']:.2f}%")
    
    # Save winner
    winner_file = output_dir / "reports" / "best_method.txt"
    with open(winner_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BEST CLUSTERING METHOD\n")
        f.write("="*80 + "\n\n")
        f.write(f"Method: {best['method']}\n")
        f.write(f"Composite Score: {best['composite_score']:.4f}\n\n")
        f.write(f"Quality Metrics:\n")
        f.write(f"  Silhouette:  {best['silhouette']:.4f}\n")
        f.write(f"  Davies-B:    {best['davies_bouldin']:.4f}\n")
        f.write(f"  Calinski-H:  {best['calinski_harabasz']:.0f}\n\n")
        f.write(f"Fraud Labeling:\n")
        f.write(f"  Rate:  {best['fraud_rate']:.2f}%\n")
        f.write(f"  Count: {best['fraud_count']:,}\n\n")
        f.write(f"Next Steps:\n")
        f.write(f"  1. Update config/clustering_config.yaml\n")
        f.write(f"  2. Enable only '{best['method']}'\n")
        f.write(f"  3. Run: python scripts/test_gnn.py\n")
    
    print(f"\n✅ Saved: {winner_file.name}")
    
    return best


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("🔬 CLUSTERING EVALUATION - INTERNAL METRICS")
    print("="*80)
    print(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Setup
        print("\n" + "="*80)
        print("⚙️  SETUP")
        print("="*80)
        
        config = load_all_configs()
        set_seed_simple(config['project']['random_seed'])
        setup_output_directory(OUTPUT_DIR)
        
        # Load data
        print("\n" + "="*80)
        print("📊 DATA LOADING")
        print("="*80)
        
        df = load_data(config, sample_size=SAMPLE_SIZE)
        print(f"✅ Loaded: {len(df):,} transactions")
        
        # Feature engineering
        print("\n" + "="*80)
        print("⚙️  FEATURE ENGINEERING")
        print("="*80)
        
        engineer = FeatureEngineer(config)
        df_processed = engineer.engineer_features(df)
        print(f"✅ Features: {len(df_processed.columns)}")
        
        # Clustering
        print("\n" + "="*80)
        print("🔬 CLUSTERING EXPERIMENTS")
        print("="*80)
        
        clustering_exp = ClusteringExperiment(config, test_mode=False)
        clustering_results = clustering_exp.run_all(df_processed)
        
        X_scaled, X_raw = clustering_exp.prepare_features(df_processed)
        
        # Validation
        validate_inputs(df_processed, X_scaled, clustering_results)
        
        # Evaluation
        df_results = evaluate_clustering_internal(clustering_results, X_scaled)
        
        # Visualizations
        print("\n" + "="*80)
        print("🎨 CREATING VISUALIZATIONS")
        print("="*80)
        
        for i, result in enumerate(clustering_results):
            method_name = result['algorithm']
            labels = result['labels']
            fraud_mask = result['fraud_mask']
            
            print(f"\n[{i+1}/{len(clustering_results)}] {method_name}")
            create_tsne_visualization(X_scaled, labels, fraud_mask, method_name, OUTPUT_DIR, TSNE_SAMPLE_SIZE)
        
        # Comparison plots
        create_comparison_plots(df_results, OUTPUT_DIR)
        
        # Ranking
        best_method = create_ranking_report(df_results, OUTPUT_DIR)
        
        # Summary
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETE!")
        print("="*80)
        print(f"\n⏱️  Time: {elapsed/60:.2f} minutes")
        print(f"\n📁 Outputs: {OUTPUT_DIR}/")
        print(f"🏆 Winner: {best_method['method']}")
        print(f"\n💡 Next: Update config and run GNN training")
        
        return 0
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ ERROR!")
        print("="*80)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
