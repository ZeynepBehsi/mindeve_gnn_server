"""
Clustering visualization with PCA, t-SNE, UMAP, Silhouette
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_samples


class ClusteringVisualizer:
    """Visualize clustering results"""
    
    def __init__(self, config: dict):
        self.config = config
        self.viz_config = config['visualization']
        
        # Style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
    
    def plot_all(self, X: np.ndarray, labels: np.ndarray, 
                 fraud_mask: np.ndarray, algorithm_name: str,
                 save_dir: str):
        """
        Create all visualizations for one clustering result
        
        Args:
            X: Feature matrix (standardized)
            labels: Cluster labels
            fraud_mask: Binary fraud labels
            algorithm_name: Algorithm name for title
            save_dir: Output directory
        """
        print(f"\n  📊 Creating visualizations for {algorithm_name}...")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample for speed if needed
        n_samples = len(X)
        max_samples = self.viz_config.get('tsne', {}).get('max_samples_test', 5000)
        
        if n_samples > max_samples:
            print(f"    Sampling {max_samples} points for t-SNE/UMAP...")
            sample_idx = np.random.choice(n_samples, max_samples, replace=False)
            X_sample = X[sample_idx]
            labels_sample = labels[sample_idx]
            fraud_sample = fraud_mask[sample_idx]
        else:
            X_sample = X
            labels_sample = labels
            fraud_sample = fraud_mask
        
        # Create 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f'{algorithm_name} - Clustering Analysis', fontsize=16, fontweight='bold')
        
        # 1. PCA 2D
        self._plot_pca(X, labels, fraud_mask, axes[0, 0])
        
        # 2. t-SNE 2D
        self._plot_tsne(X_sample, labels_sample, fraud_sample, axes[0, 1])
        
        # 3. UMAP 2D
        self._plot_umap(X_sample, labels_sample, fraud_sample, axes[1, 0])
        
        # 4. Silhouette Plot
        self._plot_silhouette(X, labels, axes[1, 1])
        
        plt.tight_layout()
        
        # Save
        filename = f"{algorithm_name.replace(' ', '_').lower()}_full_viz.{self.viz_config['save_format']}"
        filepath = save_dir / filename
        plt.savefig(filepath, dpi=self.viz_config['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Saved: {filepath.name}")
    
    def _plot_pca(self, X: np.ndarray, labels: np.ndarray, 
                  fraud_mask: np.ndarray, ax):
        """PCA 2D projection"""
        pca_config = self.viz_config.get('pca', {})
        n_components = pca_config.get('n_components', 2)
        
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Plot by fraud label (more intuitive)
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=fraud_mask,
            cmap='RdYlGn_r',  # Red=fraud, Green=normal
            s=10,
            alpha=0.6,
            edgecolors='none'
        )
        
        ax.set_title('PCA 2D Projection (by Fraud Label)')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        
        # Legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='green', markersize=10, label='Normal'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='red', markersize=10, label='Fraud')
        ]
        ax.legend(handles=handles, loc='upper right')
        
        ax.grid(True, alpha=0.3)
    
    
    def _plot_tsne(self, X: np.ndarray, labels: np.ndarray, 
               fraud_mask: np.ndarray, ax):
        """t-SNE 2D projection"""
        tsne_config = self.viz_config.get('tsne', {})
        
        tsne = TSNE(
            n_components=tsne_config.get('n_components', 2),
            perplexity=tsne_config.get('perplexity', 30),
            max_iter=tsne_config.get('n_iter', 1000),  # ✅ Düzeltildi
            random_state=tsne_config.get('random_state', 42),
            n_jobs=-1  # ✅ Bonus: Hızlandırma
        )
        
        X_tsne = tsne.fit_transform(X)
        
        scatter = ax.scatter(
            X_tsne[:, 0], X_tsne[:, 1],
            c=fraud_mask,
            cmap='RdYlGn_r',
            s=10,
            alpha=0.6,
            edgecolors='none'
        )
        
        ax.set_title('t-SNE 2D Projection (by Fraud Label)')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        
        # Legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor='green', markersize=10, label='Normal'),
            plt.Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor='red', markersize=10, label='Fraud')
        ]
        ax.legend(handles=handles, loc='upper right')
        
        ax.grid(True, alpha=0.3)

    #------
    
    def _plot_umap(self, X: np.ndarray, labels: np.ndarray, 
                   fraud_mask: np.ndarray, ax):
        """UMAP 2D projection"""
        umap_config = self.viz_config.get('umap', {})
        
        reducer = umap.UMAP(
            n_components=umap_config.get('n_components', 2),
            n_neighbors=umap_config.get('n_neighbors', 15),
            min_dist=umap_config.get('min_dist', 0.1),
            random_state=umap_config.get('random_state', 42)
        )
        
        X_umap = reducer.fit_transform(X)
        
        scatter = ax.scatter(
            X_umap[:, 0], X_umap[:, 1],
            c=fraud_mask,
            cmap='RdYlGn_r',
            s=10,
            alpha=0.6,
            edgecolors='none'
        )
        
        ax.set_title('UMAP 2D Projection (by Fraud Label)')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        
        # Legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='green', markersize=10, label='Normal'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='red', markersize=10, label='Fraud')
        ]
        ax.legend(handles=handles, loc='upper right')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_silhouette(self, X: np.ndarray, labels: np.ndarray, ax):
        """Silhouette plot"""
        silhouette_config = self.viz_config.get('silhouette', {})
        sample_size = silhouette_config.get('sample_size', 10000)
        
        # Sample if too large
        if len(X) > sample_size:
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[sample_idx]
            labels_sample = labels[sample_idx]
        else:
            X_sample = X
            labels_sample = labels
        
        # Compute silhouette scores
        try:
            # Exclude outliers for DBSCAN
            if -1 in labels_sample:
                mask = labels_sample != -1
                if mask.sum() > 100:
                    sample_silhouette_values = silhouette_samples(X_sample[mask], labels_sample[mask])
                    labels_plot = labels_sample[mask]
                else:
                    ax.text(0.5, 0.5, 'Not enough points for silhouette plot', 
                           ha='center', va='center', transform=ax.transAxes)
                    return
            else:
                sample_silhouette_values = silhouette_samples(X_sample, labels_sample)
                labels_plot = labels_sample
            
            # Plot
            y_lower = 10
            unique_labels = np.unique(labels_plot)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
            
            for i, (label, color) in enumerate(zip(unique_labels, colors)):
                cluster_silhouette_values = sample_silhouette_values[labels_plot == label]
                cluster_silhouette_values.sort()
                
                size = cluster_silhouette_values.shape[0]
                y_upper = y_lower + size
                
                ax.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                    label=f'Cluster {label}'
                )
                
                # Label cluster
                ax.text(-0.05, y_lower + 0.5 * size, str(label))
                y_lower = y_upper + 10
            
            # Mean line
            mean_score = sample_silhouette_values.mean()
            ax.axvline(x=mean_score, color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {mean_score:.3f}')
            
            ax.set_title('Silhouette Plot')
            ax.set_xlabel('Silhouette Coefficient')
            ax.set_ylabel('Cluster Label')
            ax.set_xlim([-1, 1])
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Silhouette plot error:\n{str(e)[:50]}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def plot_comparison_summary(self, results: list, save_dir: str):
        """
        Create summary comparison plot for all algorithms
        
        Args:
            results: List of clustering results
            save_dir: Output directory
        """
        print(f"\n  📊 Creating comparison summary...")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        df_results = []
        for r in results:
            row = {
                'Algorithm': r['algorithm'],
                'Params': str(r['params']),
                'Silhouette': r['metrics']['silhouette'],
                'Davies-Bouldin': r['metrics']['davies_bouldin'],
                'Calinski-Harabasz': r['metrics']['calinski_harabasz'],
                'Fraud Rate (%)': r['metrics']['fraud_rate'] * 100,
                'Time (s)': r['metrics']['train_time']
            }
            df_results.append(row)
        
        import pandas as pd
        df = pd.DataFrame(df_results)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Clustering Algorithms Comparison', fontsize=16, fontweight='bold')
        
        # 1. Silhouette scores
        df_sorted = df.sort_values('Silhouette', ascending=False)
        colors = ['green' if s > 0.3 else 'orange' if s > 0.1 else 'red' 
                 for s in df_sorted['Silhouette']]
        
        axes[0, 0].barh(range(len(df_sorted)), df_sorted['Silhouette'], color=colors, alpha=0.7)
        axes[0, 0].set_yticks(range(len(df_sorted)))
        axes[0, 0].set_yticklabels([f"{row['Algorithm']} ({i})" 
                                    for i, row in enumerate(df_sorted.to_dict('records'))],
                                   fontsize=8)
        axes[0, 0].set_xlabel('Silhouette Score')
        axes[0, 0].set_title('Silhouette Score (Higher = Better)')
        axes[0, 0].axvline(x=0.3, color='green', linestyle='--', alpha=0.5, label='Good (>0.3)')
        axes[0, 0].axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='Fair (>0.1)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Davies-Bouldin (lower is better)
        df_sorted = df[df['Davies-Bouldin'] < 900].sort_values('Davies-Bouldin')
        if len(df_sorted) > 0:
            colors = ['green' if d < 1.0 else 'orange' if d < 2.0 else 'red' 
                     for d in df_sorted['Davies-Bouldin']]
            
            axes[0, 1].barh(range(len(df_sorted)), df_sorted['Davies-Bouldin'], color=colors, alpha=0.7)
            axes[0, 1].set_yticks(range(len(df_sorted)))
            axes[0, 1].set_yticklabels([f"{row['Algorithm']} ({i})" 
                                        for i, row in enumerate(df_sorted.to_dict('records'))],
                                       fontsize=8)
            axes[0, 1].set_xlabel('Davies-Bouldin Index')
            axes[0, 1].set_title('Davies-Bouldin Index (Lower = Better)')
            axes[0, 1].axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Good (<1.0)')
            axes[0, 1].axvline(x=2.0, color='orange', linestyle='--', alpha=0.5, label='Fair (<2.0)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Fraud Rate
        axes[1, 0].scatter(df.index, df['Fraud Rate (%)'], s=100, alpha=0.6, c=df['Silhouette'], cmap='viridis')
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Min (0.5%)')
        axes[1, 0].axhline(y=5.0, color='red', linestyle='--', alpha=0.5, label='Max (5%)')
        axes[1, 0].set_xlabel('Experiment Index')
        axes[1, 0].set_ylabel('Fraud Rate (%)')
        axes[1, 0].set_title('Fraud Rate Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Training Time
        df_sorted = df.sort_values('Time (s)')
        axes[1, 1].barh(range(len(df_sorted)), df_sorted['Time (s)'], alpha=0.7, color='steelblue')
        axes[1, 1].set_yticks(range(len(df_sorted)))
        axes[1, 1].set_yticklabels([f"{row['Algorithm']} ({i})" 
                                    for i, row in enumerate(df_sorted.to_dict('records'))],
                                   fontsize=8)
        axes[1, 1].set_xlabel('Training Time (seconds)')
        axes[1, 1].set_title('Computational Efficiency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filepath = save_dir / f"comparison_summary.{self.viz_config['save_format']}"
        plt.savefig(filepath, dpi=self.viz_config['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Saved: {filepath.name}")
        
        # Also save CSV
        csv_path = save_dir / "comparison_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"    ✅ Saved: {csv_path.name}")