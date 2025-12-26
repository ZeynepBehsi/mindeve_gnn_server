"""
Complete Clustering Evaluation Script
- Internal metrics only (no ground truth needed)
- t-SNE visualizations for each method
- Comprehensive CSV reports
- All validations and error checking
- Compatible with existing project structure

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

# Project imports (files are in /mnt/project/ directly, not in src/)
from config_loader import load_all_configs
# Don't import set_seed - we'll define our own torch-free version
from loader import load_data
from preprocessor import FeatureEngineer
from clustering import ClusteringExperiment


# ============================================================================
# TORCH-FREE SEED FUNCTION (for clustering evaluation only)
# ============================================================================

def set_seed_simple(seed: int = 42):
    """Set random seeds without torch (clustering doesn't need it)"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    print(f"✅ Random seed set to {seed} (numpy + random)")


# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLE_SIZE = 100000  # 100K for testing
OUTPUT_DIR = Path("outputs/clustering_evaluation")
TSNE_SAMPLE_SIZE = 5000  # Sample for t-SNE (computational efficiency)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_inputs(df: pd.DataFrame, X: np.ndarray, clustering_results: list):
    """
    Validate all inputs before processing
    
    Args:
        df: Input dataframe
        X: Feature matrix
        clustering_results: List of clustering results
    
    Raises:
        ValueError: If validation fails
    """
    print("\n" + "="*80)
    print("🔍 VALIDATING INPUTS")
    print("="*80)
    
    # Check dataframe
    if df is None or len(df) == 0:
        raise ValueError("❌ DataFrame is empty!")
    print(f"✅ DataFrame: {len(df):,} rows, {len(df.columns)} columns")
    
    # Check feature matrix
    if X is None or len(X) == 0:
        raise ValueError("❌ Feature matrix is empty!")
    if X.shape[0] != len(df):
        raise ValueError(f"❌ Shape mismatch: X has {X.shape[0]} rows, df has {len(df)} rows")
    print(f"✅ Feature matrix: {X.shape}")
    
    # Check clustering results
    if not clustering_results or len(clustering_results) == 0:
        raise ValueError("❌ No clustering results found!")
    print(f"✅ Clustering results: {len(clustering_results)} experiments")
    
    # Validate each result structure
    required_keys = ['algorithm', 'params', 'labels', 'fraud_mask', 'metrics']
    for i, result in enumerate(clustering_results):
        missing = [key for key in required_keys if key not in result]
        if missing:
            raise ValueError(f"❌ Result {i} missing keys: {missing}")
        
        # Check labels shape
        if len(result['labels']) != len(df):
            raise ValueError(f"❌ Result {i} labels shape mismatch")
        
        # Check fraud_mask shape
        if len(result['fraud_mask']) != len(df):
            raise ValueError(f"❌ Result {i} fraud_mask shape mismatch")
    
    print("✅ All validations passed!")


def setup_output_directory(output_dir: Path):
    """
    Create output directory structure
    
    Args:
        output_dir: Base output directory
    """
    # Create directories
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

def evaluate_clustering_internal(
    clustering_results: list, 
    X: np.ndarray
) -> pd.DataFrame:
    """
    Evaluate clustering with internal metrics only
    
    Args:
        clustering_results: List of clustering experiment results
        X: Standardized feature matrix
    
    Returns:
        DataFrame with evaluation results
    """
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
            # ================================================================
            # 1. INTERNAL QUALITY METRICS
            # ================================================================
            
            # Silhouette Score (-1 to 1, higher is better)
            try:
                if -1 in labels:
                    # DBSCAN: exclude outliers
                    mask = labels != -1
                    if mask.sum() > 100:
                        silhouette = silhouette_score(X[mask], labels[mask])
                    else:
                        silhouette = 0.0
                        print("   ⚠️  Too few non-outlier samples for silhouette")
                else:
                    silhouette = silhouette_score(X, labels)
            except Exception as e:
                silhouette = 0.0
                print(f"   ⚠️  Silhouette calculation failed: {e}")
            
            # Davies-Bouldin Index (>=0, lower is better)
            try:
                if -1 in labels:
                    mask = labels != -1
                    if mask.sum() > 100:
                        davies_bouldin = davies_bouldin_score(X[mask], labels[mask])
                    else:
                        davies_bouldin = 999.0
                else:
                    davies_bouldin = davies_bouldin_score(X, labels)
            except Exception as e:
                davies_bouldin = 999.0
                print(f"   ⚠️  Davies-Bouldin calculation failed: {e}")
            
            # Calinski-Harabasz Score (>=0, higher is better)
            try:
                if -1 in labels:
                    mask = labels != -1
                    if mask.sum() > 100:
                        calinski = calinski_harabasz_score(X[mask], labels[mask])
                    else:
                        calinski = 0.0
                else:
                    calinski = calinski_harabasz_score(X, labels)
            except Exception as e:
                calinski = 0.0
                print(f"   ⚠️  Calinski-Harabasz calculation failed: {e}")
            
            # ================================================================
            # 2. FRAUD LABELING STATISTICS
            # ================================================================
            
            fraud_rate = fraud_mask.mean()
            fraud_count = fraud_mask.sum()
            normal_count = len(fraud_mask) - fraud_count
            
            # ================================================================
            # 3. CLUSTER STATISTICS
            # ================================================================
            
            unique_labels = np.unique(labels[labels >= 0])
            n_clusters = len(unique_labels)
            
            if n_clusters > 0:
                cluster_sizes = [np.sum(labels == label) for label in unique_labels]
                min_cluster_size = min(cluster_sizes)
                max_cluster_size = max(cluster_sizes)
                avg_cluster_size = np.mean(cluster_sizes)
                std_cluster_size = np.std(cluster_sizes)
                
                # Find fraud cluster (smallest)
                fraud_cluster_idx = np.argmin(cluster_sizes)
                fraud_cluster_size = min_cluster_size
            else:
                min_cluster_size = max_cluster_size = avg_cluster_size = std_cluster_size = 0
                fraud_cluster_size = 0
            
            # Check for outliers (DBSCAN)
            n_outliers = np.sum(labels == -1)
            outlier_rate = n_outliers / len(labels) if len(labels) > 0 else 0
            
            # ================================================================
            # 4. VALIDITY CHECKS
            # ================================================================
            
            fraud_rate_valid = 0.07 <= fraud_rate <= 0.15
            silhouette_valid = silhouette > 0.1
            davies_bouldin_valid = davies_bouldin < 2.0
            
            # ================================================================
            # 5. COMPOSITE SCORE CALCULATION
            # ================================================================
            
            # Normalize metrics to 0-1 scale
            silhouette_norm = max(0, min(1, (silhouette + 1) / 2))  # -1~1 → 0~1
            davies_bouldin_norm = 1 / (1 + davies_bouldin)  # Lower is better → invert
            calinski_norm = min(calinski / 1000, 1.0)  # Cap at 1.0
            
            # Fraud rate penalty/reward
            if fraud_rate < 0.07:
                fraud_rate_score = fraud_rate / 0.07  # Too low: penalty
            elif fraud_rate > 0.15:
                fraud_rate_score = 0.15 / fraud_rate  # Too high: penalty
            else:
                fraud_rate_score = 1.0  # Perfect range: no penalty
            
            # Weighted composite score
            composite_score = (
                silhouette_norm * 0.35 +       # Cluster separation (most important)
                davies_bouldin_norm * 0.25 +   # Cluster compactness
                calinski_norm * 0.20 +         # Variance ratio
                fraud_rate_score * 0.20        # Fraud rate validity
            )
            
            # ================================================================
            # 6. PRINT RESULTS
            # ================================================================
            
            print(f"\n  📐 Internal Quality Metrics:")
            print(f"    Silhouette Score:      {silhouette:>8.4f}  {'✅' if silhouette > 0.3 else '⚠️' if silhouette > 0.1 else '❌'}")
            print(f"    Davies-Bouldin Index:  {davies_bouldin:>8.4f}  {'✅' if davies_bouldin < 1.0 else '⚠️' if davies_bouldin < 2.0 else '❌'}")
            print(f"    Calinski-Harabasz:     {calinski:>8.0f}  {'✅' if calinski > 100 else '⚠️' if calinski > 50 else '❌'}")
            
            print(f"\n  🎯 Fraud Labeling Statistics:")
            print(f"    Fraud Count:           {fraud_count:>8,}")
            print(f"    Normal Count:          {normal_count:>8,}")
            print(f"    Fraud Rate:            {fraud_rate*100:>7.2f}%  {'✅' if fraud_rate_valid else '❌'}")
            
            print(f"\n  📊 Cluster Distribution:")
            print(f"    Num Clusters:          {n_clusters:>8}")
            print(f"    Min Cluster Size:      {min_cluster_size:>8,}")
            print(f"    Max Cluster Size:      {max_cluster_size:>8,}")
            print(f"    Avg Cluster Size:      {avg_cluster_size:>8,.0f}")
            print(f"    Std Cluster Size:      {std_cluster_size:>8,.0f}")
            
            if n_outliers > 0:
                print(f"    Outliers (label=-1):   {n_outliers:>8,} ({outlier_rate*100:.2f}%)")
            
            print(f"\n  🏆 Composite Score:      {composite_score:>8.4f}")
            
            # ================================================================
            # 7. STORE RESULTS
            # ================================================================
            
            evaluation_results.append({
                'method': method,
                'params': str(params),
                # Internal metrics
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'calinski_harabasz': calinski,
                # Fraud stats
                'fraud_rate': fraud_rate * 100,
                'fraud_count': fraud_count,
                'normal_count': normal_count,
                # Cluster stats
                'n_clusters': n_clusters,
                'min_cluster_size': min_cluster_size,
                'max_cluster_size': max_cluster_size,
                'avg_cluster_size': avg_cluster_size,
                'n_outliers': n_outliers,
                'outlier_rate': outlier_rate * 100,
                # Score
                'composite_score': composite_score,
                # Validity flags
                'silhouette_valid': silhouette_valid,
                'davies_bouldin_valid': davies_bouldin_valid,
                'fraud_rate_valid': fraud_rate_valid
            })
            
        except Exception as e:
            print(f"\n  ❌ ERROR evaluating {method}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return pd.DataFrame(evaluation_results)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_tsne_visualization(
    X: np.ndarray,
    labels: np.ndarray,
    fraud_mask: np.ndarray,
    method_name: str,
    output_dir: Path,
    sample_size: int = 5000
):
    """
    Create t-SNE visualization for clustering result
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        fraud_mask: Binary fraud labels
        method_name: Algorithm name
        output_dir: Output directory
        sample_size: Max samples for t-SNE (computational efficiency)
    """
    print(f"\n  🎨 Creating t-SNE visualization for {method_name}...")
    
    try:
        # Sample if too large
        n_samples = len(X)
        if n_samples > sample_size:
            sample_idx = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[sample_idx]
            labels_sample = labels[sample_idx]
            fraud_sample = fraud_mask[sample_idx]
            print(f"     Sampling: {n_samples:,} → {sample_size:,} samples")
        else:
            X_sample = X
            labels_sample = labels
            fraud_sample = fraud_mask
        
        # Run t-SNE
        print(f"     Running t-SNE (perplexity=30, n_iter=1000)...")
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            n_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        X_tsne = tsne.fit_transform(X_sample)
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f't-SNE Visualization: {method_name}', fontsize=14, fontweight='bold')
        
        # ============================================================
        # Plot 1: Color by Cluster Labels
        # ============================================================
        
        # Filter out outliers for better visualization
        mask_valid = labels_sample >= 0
        
        if mask_valid.sum() > 0:
            unique_labels = np.unique(labels_sample[mask_valid])
            n_clusters = len(unique_labels)
            
            if n_clusters > 1:
                colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
                
                for i, label in enumerate(unique_labels):
                    mask = labels_sample == label
                    axes[0].scatter(
                        X_tsne[mask, 0], X_tsne[mask, 1],
                        c=[colors[i]], label=f'Cluster {label}',
                        s=20, alpha=0.6, edgecolors='none'
                    )
            else:
                axes[0].scatter(
                    X_tsne[mask_valid, 0], X_tsne[mask_valid, 1],
                    c='steelblue', label='All samples',
                    s=20, alpha=0.6, edgecolors='none'
                )
        
        # Plot outliers (DBSCAN)
        if np.any(labels_sample == -1):
            mask_outliers = labels_sample == -1
            axes[0].scatter(
                X_tsne[mask_outliers, 0], X_tsne[mask_outliers, 1],
                c='red', marker='x', label='Outliers', s=30, alpha=0.8
            )
        
        axes[0].set_title('Colored by Cluster Labels')
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        axes[0].legend(loc='upper right', fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        # ============================================================
        # Plot 2: Color by Fraud Labels
        # ============================================================
        
        scatter = axes[1].scatter(
            X_tsne[:, 0], X_tsne[:, 1],
            c=fraud_sample,
            cmap='RdYlGn_r',  # Red=fraud, Green=normal
            s=20, alpha=0.6, edgecolors='none'
        )
        
        axes[1].set_title('Colored by Fraud Labels')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        
        # Legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='green', markersize=10, label='Normal'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='red', markersize=10, label='Fraud')
        ]
        axes[1].legend(handles=handles, loc='upper right', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        safe_name = method_name.replace(' ', '_').replace('/', '_').lower()
        filepath = output_dir / "tsne_plots" / f"{safe_name}_tsne.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     ✅ Saved: {filepath.name}")
        
    except Exception as e:
        print(f"     ❌ t-SNE visualization failed: {e}")
        import traceback
        traceback.print_exc()


def create_pca_visualization(
    X: np.ndarray,
    labels: np.ndarray,
    fraud_mask: np.ndarray,
    method_name: str,
    output_dir: Path
):
    """
    Create PCA visualization for clustering result
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        fraud_mask: Binary fraud labels
        method_name: Algorithm name
        output_dir: Output directory
    """
    print(f"\n  📊 Creating PCA visualization for {method_name}...")
    
    try:
        # Run PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'PCA Visualization: {method_name}', fontsize=14, fontweight='bold')
        
        # Plot 1: Color by Cluster Labels
        mask_valid = labels >= 0
        
        if mask_valid.sum() > 0:
            unique_labels = np.unique(labels[mask_valid])
            n_clusters = len(unique_labels)
            
            if n_clusters > 1:
                colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
                
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    axes[0].scatter(
                        X_pca[mask, 0], X_pca[mask, 1],
                        c=[colors[i]], label=f'Cluster {label}',
                        s=20, alpha=0.6, edgecolors='none'
                    )
            else:
                axes[0].scatter(
                    X_pca[mask_valid, 0], X_pca[mask_valid, 1],
                    c='steelblue', label='All samples',
                    s=20, alpha=0.6, edgecolors='none'
                )
        
        # Plot outliers
        if np.any(labels == -1):
            mask_outliers = labels == -1
            axes[0].scatter(
                X_pca[mask_outliers, 0], X_pca[mask_outliers, 1],
                c='red', marker='x', label='Outliers', s=30, alpha=0.8
            )
        
        axes[0].set_title('Colored by Cluster Labels')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[0].legend(loc='upper right', fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Color by Fraud Labels
        scatter = axes[1].scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=fraud_mask,
            cmap='RdYlGn_r',
            s=20, alpha=0.6, edgecolors='none'
        )
        
        axes[1].set_title('Colored by Fraud Labels')
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        
        # Legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='green', markersize=10, label='Normal'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='red', markersize=10, label='Fraud')
        ]
        axes[1].legend(handles=handles, loc='upper right', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        safe_name = method_name.replace(' ', '_').replace('/', '_').lower()
        filepath = output_dir / "pca_plots" / f"{safe_name}_pca.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     ✅ Saved: {filepath.name}")
        
    except Exception as e:
        print(f"     ❌ PCA visualization failed: {e}")


def create_comparison_plots(df_results: pd.DataFrame, output_dir: Path):
    """
    Create comprehensive comparison plots
    
    Args:
        df_results: Evaluation results dataframe
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("📊 CREATING COMPARISON PLOTS")
    print("="*80)
    
    try:
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Clustering Methods Comparison', fontsize=16, fontweight='bold')
        
        methods = df_results['method'].values
        n_methods = len(methods)
        colors = plt.cm.viridis(np.linspace(0, 1, n_methods))
        
        # ============================================================
        # Plot 1: Composite Score Ranking
        # ============================================================
        
        axes[0, 0].barh(range(n_methods), df_results['composite_score'], color=colors, alpha=0.7)
        axes[0, 0].set_yticks(range(n_methods))
        axes[0, 0].set_yticklabels(methods, fontsize=8)
        axes[0, 0].set_xlabel('Composite Score')
        axes[0, 0].set_title('🏆 Overall Ranking (Higher = Better)')
        axes[0, 0].invert_yaxis()
        
        for i, v in enumerate(df_results['composite_score']):
            axes[0, 0].text(v, i, f' {v:.3f}', va='center', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # ============================================================
        # Plot 2: Silhouette Score
        # ============================================================
        
        axes[0, 1].barh(range(n_methods), df_results['silhouette'], color=colors, alpha=0.7)
        axes[0, 1].set_yticks(range(n_methods))
        axes[0, 1].set_yticklabels(methods, fontsize=8)
        axes[0, 1].set_xlabel('Silhouette Score')
        axes[0, 1].set_title('📐 Cluster Separation Quality')
        axes[0, 1].axvline(x=0.3, color='green', linestyle='--', alpha=0.5, label='Good (>0.3)')
        axes[0, 1].axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='Fair (>0.1)')
        axes[0, 1].legend(fontsize=7)
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # ============================================================
        # Plot 3: Davies-Bouldin Index
        # ============================================================
        
        # Filter out extreme values for better visualization
        df_filtered = df_results[df_results['davies_bouldin'] < 10].copy()
        
        if len(df_filtered) > 0:
            methods_filtered = df_filtered['method'].values
            n_filtered = len(methods_filtered)
            colors_filtered = plt.cm.viridis(np.linspace(0, 1, n_filtered))
            
            axes[0, 2].barh(range(n_filtered), df_filtered['davies_bouldin'], 
                           color=colors_filtered, alpha=0.7)
            axes[0, 2].set_yticks(range(n_filtered))
            axes[0, 2].set_yticklabels(methods_filtered, fontsize=8)
            axes[0, 2].set_xlabel('Davies-Bouldin Index')
            axes[0, 2].set_title('📏 Cluster Compactness (Lower = Better)')
            axes[0, 2].axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Good (<1.0)')
            axes[0, 2].axvline(x=2.0, color='orange', linestyle='--', alpha=0.5, label='Fair (<2.0)')
            axes[0, 2].legend(fontsize=7)
            axes[0, 2].invert_yaxis()
            axes[0, 2].grid(True, alpha=0.3, axis='x')
        else:
            axes[0, 2].text(0.5, 0.5, 'No valid data\n(all values > 10)', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
        
        # ============================================================
        # Plot 4: Fraud Rate
        # ============================================================
        
        axes[1, 0].barh(range(n_methods), df_results['fraud_rate'], color=colors, alpha=0.7)
        axes[1, 0].set_yticks(range(n_methods))
        axes[1, 0].set_yticklabels(methods, fontsize=8)
        axes[1, 0].set_xlabel('Fraud Rate (%)')
        axes[1, 0].set_title('🎯 Fraud Labeling Rate')
        axes[1, 0].axvspan(7, 15, alpha=0.2, color='green', label='Ideal Range (7-15%)')
        axes[1, 0].legend(fontsize=7)
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # ============================================================
        # Plot 5: Calinski-Harabasz Score
        # ============================================================
        
        axes[1, 1].barh(range(n_methods), df_results['calinski_harabasz'], color=colors, alpha=0.7)
        axes[1, 1].set_yticks(range(n_methods))
        axes[1, 1].set_yticklabels(methods, fontsize=8)
        axes[1, 1].set_xlabel('Calinski-Harabasz Score')
        axes[1, 1].set_title('📊 Variance Ratio (Higher = Better)')
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        # ============================================================
        # Plot 6: All Metrics Comparison (Normalized)
        # ============================================================
        
        # Normalize all metrics to 0-1
        metrics_norm = df_results[['silhouette', 'davies_bouldin', 'calinski_harabasz', 'fraud_rate']].copy()
        metrics_norm['silhouette'] = (metrics_norm['silhouette'] + 1) / 2
        metrics_norm['davies_bouldin'] = 1 / (1 + metrics_norm['davies_bouldin'])
        metrics_norm['calinski_harabasz'] = metrics_norm['calinski_harabasz'] / metrics_norm['calinski_harabasz'].max()
        
        # Fraud rate scoring
        fraud_rates = metrics_norm['fraud_rate'].values
        fraud_score_norm = []
        for fr in fraud_rates:
            if fr < 7:
                fraud_score_norm.append(fr / 7)
            elif fr > 15:
                fraud_score_norm.append(15 / fr)
            else:
                fraud_score_norm.append(1.0)
        metrics_norm['fraud_rate'] = fraud_score_norm
        
        x = np.arange(len(methods))
        width = 0.2
        
        axes[1, 2].barh(x - 1.5*width, metrics_norm['silhouette'], width, 
                       label='Silhouette', alpha=0.8, color='steelblue')
        axes[1, 2].barh(x - 0.5*width, metrics_norm['davies_bouldin'], width, 
                       label='Davies-Bouldin', alpha=0.8, color='coral')
        axes[1, 2].barh(x + 0.5*width, metrics_norm['calinski_harabasz'], width, 
                       label='Calinski-Harabasz', alpha=0.8, color='lightgreen')
        axes[1, 2].barh(x + 1.5*width, metrics_norm['fraud_rate'], width, 
                       label='Fraud Rate Score', alpha=0.8, color='gold')
        
        axes[1, 2].set_yticks(x)
        axes[1, 2].set_yticklabels(methods, fontsize=8)
        axes[1, 2].set_xlabel('Normalized Score (0-1)')
        axes[1, 2].set_title('📈 All Metrics Comparison (Normalized)')
        axes[1, 2].legend(fontsize=7, loc='lower right')
        axes[1, 2].invert_yaxis()
        axes[1, 2].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save
        filepath = output_dir / "reports" / "comparison_plots.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved comparison plots: {filepath.name}")
        
    except Exception as e:
        print(f"❌ Comparison plots failed: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# REPORTING FUNCTIONS
# ============================================================================

def create_ranking_report(df_results: pd.DataFrame, output_dir: Path):
    """
    Create ranking report and identify best method
    
    Args:
        df_results: Evaluation results dataframe
        output_dir: Output directory
    
    Returns:
        Best method row (Series)
    """
    print("\n" + "="*80)
    print("📋 CLUSTERING METHODS RANKING")
    print("="*80)
    
    # Sort by composite score
    df_sorted = df_results.sort_values('composite_score', ascending=False)
    
    # Display table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    pd.set_option('display.max_rows', None)
    
    print("\n" + df_sorted.to_string(index=False))
    
    # Save detailed CSV
    csv_path = output_dir / "reports" / "clustering_ranking_detailed.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"\n✅ Saved detailed ranking: {csv_path.name}")
    
    # Save summary CSV (key metrics only)
    summary_cols = [
        'method', 'composite_score', 'silhouette', 'davies_bouldin', 
        'fraud_rate', 'n_clusters', 'fraud_count'
    ]
    df_summary = df_sorted[summary_cols].copy()
    
    csv_summary_path = output_dir / "reports" / "clustering_ranking_summary.csv"
    df_summary.to_csv(csv_summary_path, index=False)
    print(f"✅ Saved summary ranking: {csv_summary_path.name}")
    
    # Winner announcement
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
    print(f"    Fraud Rate:       {best['fraud_rate']:>7.2f}%")
    print(f"    Fraud Count:      {best['fraud_count']:>8,}")
    print(f"    Normal Count:     {best['normal_count']:>8,}")
    print(f"\n  📊 Clusters:")
    print(f"    Num Clusters:     {best['n_clusters']:>8}")
    
    # Top 3 with medals
    print("\n" + "="*80)
    print("🥇🥈🥉 TOP 3 METHODS")
    print("="*80)
    
    for i, (idx, row) in enumerate(df_sorted.head(3).iterrows(), 1):
        medal = ['🥇', '🥈', '🥉'][i-1]
        print(f"\n{medal} #{i}: {row['method']}")
        print(f"    Score: {row['composite_score']:.4f}")
        print(f"    Silhouette: {row['silhouette']:.4f} | Fraud Rate: {row['fraud_rate']:.2f}%")
    
    # Save winner details
    winner_path = output_dir / "reports" / "best_method.txt"
    with open(winner_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BEST CLUSTERING METHOD FOR GNN TRAINING\n")
        f.write("="*80 + "\n\n")
        f.write(f"Method: {best['method']}\n")
        f.write(f"Composite Score: {best['composite_score']:.4f}\n\n")
        f.write(f"Quality Metrics:\n")
        f.write(f"  Silhouette:       {best['silhouette']:.4f}\n")
        f.write(f"  Davies-Bouldin:   {best['davies_bouldin']:.4f}\n")
        f.write(f"  Calinski-Harabasz: {best['calinski_harabasz']:.0f}\n\n")
        f.write(f"Fraud Labeling:\n")
        f.write(f"  Fraud Rate:  {best['fraud_rate']:.2f}%\n")
        f.write(f"  Fraud Count: {best['fraud_count']:,}\n\n")
        f.write(f"Next Step:\n")
        f.write(f"  1. Update config/clustering_config.yaml\n")
        f.write(f"  2. Enable only '{best['method']}' algorithm\n")
        f.write(f"  3. Run GNN training: python scripts/test_gnn.py\n")
    
    print(f"\n✅ Saved winner details: {winner_path.name}")
    
    return best


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print("\n" + "="*80)
    print("🔬 COMPLETE CLUSTERING EVALUATION")
    print("   Internal Metrics + t-SNE + PCA Visualizations")
    print("="*80)
    print(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # ====================================================================
        # 1. SETUP
        # ====================================================================
        
        print("\n" + "="*80)
        print("⚙️  SETUP & CONFIGURATION")
        print("="*80)
        
        config = load_all_configs()
        
        # If config is empty, files might be in current directory
        if not config or len(config) == 0:
            print("⚠️  Config empty, trying current directory...")
            from config_loader import ConfigLoader
            loader = ConfigLoader(config_dir='.')
            config = loader.load_all()
        
        set_seed_simple(config['project']['random_seed'])
        setup_output_directory(OUTPUT_DIR)
        
        print(f"✅ Configuration loaded")
        print(f"✅ Random seed: {config['project']['random_seed']}")
        print(f"✅ Output directory: {OUTPUT_DIR}/")
        
        # ====================================================================
        # 2. DATA LOADING
        # ====================================================================
        
        print("\n" + "="*80)
        print("📊 DATA LOADING")
        print("="*80)
        
        df = load_data(config, sample_size=SAMPLE_SIZE)
        print(f"✅ Loaded: {len(df):,} transactions")
        
        # ====================================================================
        # 3. FEATURE ENGINEERING
        # ====================================================================
        
        print("\n" + "="*80)
        print("⚙️  FEATURE ENGINEERING")
        print("="*80)
        
        engineer = FeatureEngineer(config)
        df_processed = engineer.engineer_features(df)
        print(f"✅ Engineered: {len(df_processed.columns)} features")
        
        # ====================================================================
        # 4. CLUSTERING EXPERIMENTS
        # ====================================================================
        
        print("\n" + "="*80)
        print("🔬 CLUSTERING EXPERIMENTS")
        print("="*80)
        
        clustering_exp = ClusteringExperiment(config, test_mode=False)
        clustering_results = clustering_exp.run_all(df_processed)
        
        print(f"\n✅ Completed {len(clustering_results)} clustering experiments")
        
        # Get feature matrix
        X_scaled, X_raw = clustering_exp.prepare_features(df_processed)
        
        # ====================================================================
        # 5. VALIDATION
        # ====================================================================
        
        validate_inputs(df_processed, X_scaled, clustering_results)
        
        # ====================================================================
        # 6. EVALUATION
        # ====================================================================
        
        df_results = evaluate_clustering_internal(clustering_results, X_scaled)
        
        # ====================================================================
        # 7. VISUALIZATIONS
        # ====================================================================
        
        print("\n" + "="*80)
        print("🎨 CREATING VISUALIZATIONS")
        print("="*80)
        
        for i, result in enumerate(clustering_results):
            method_name = result['algorithm']
            labels = result['labels']
            fraud_mask = result['fraud_mask']
            
            print(f"\n[{i+1}/{len(clustering_results)}] {method_name}")
            
            # t-SNE visualization
            create_tsne_visualization(
                X_scaled, labels, fraud_mask, method_name, 
                OUTPUT_DIR, TSNE_SAMPLE_SIZE
            )
            
            # PCA visualization
            create_pca_visualization(
                X_scaled, labels, fraud_mask, method_name, OUTPUT_DIR
            )
        
        print(f"\n✅ All visualizations created")
        
        # ====================================================================
        # 8. COMPARISON PLOTS
        # ====================================================================
        
        create_comparison_plots(df_results, OUTPUT_DIR)
        
        # ====================================================================
        # 9. RANKING REPORT
        # ====================================================================
        
        best_method = create_ranking_report(df_results, OUTPUT_DIR)
        
        # ====================================================================
        # 10. SUMMARY
        # ====================================================================
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETE!")
        print("="*80)
        
        print(f"\n⏱️  Total time: {elapsed_time/60:.2f} minutes")
        
        print(f"\n📁 Output files:")
        print(f"   {OUTPUT_DIR}/")
        print(f"   ├── tsne_plots/          ({len(clustering_results)} files)")
        print(f"   ├── pca_plots/           ({len(clustering_results)} files)")
        print(f"   └── reports/")
        print(f"       ├── clustering_ranking_detailed.csv")
        print(f"       ├── clustering_ranking_summary.csv")
        print(f"       ├── comparison_plots.png")
        print(f"       └── best_method.txt")
        
        print(f"\n🏆 Winner: {best_method['method']}")
        print(f"   Composite Score: {best_method['composite_score']:.4f}")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Review: {OUTPUT_DIR}/reports/best_method.txt")
        print(f"   2. Check visualizations in tsne_plots/ and pca_plots/")
        print(f"   3. Update config/clustering_config.yaml to use only '{best_method['method']}'")
        print(f"   4. Run GNN training: python scripts/test_gnn.py")
        
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


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)