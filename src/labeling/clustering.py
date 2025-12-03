"""
Clustering algorithms for fraud labeling
Supports: K-Means, DBSCAN, Isolation Forest, GMM
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import time
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


class ClusteringAlgorithm:
    """Base class for clustering algorithms"""
    
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.labels_ = None
        self.model = None
        self.train_time = 0.0
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_fraud_mask(self) -> np.ndarray:
        """Get binary fraud mask (1=fraud, 0=normal)"""
        raise NotImplementedError
    
    def compute_metrics(self, X: np.ndarray) -> Dict[str, float]:
        """Compute clustering quality metrics"""
        metrics = {}
        
        # Silhouette score
        try:
            if len(np.unique(self.labels_)) > 1:
                # Exclude outliers for DBSCAN
                if hasattr(self, 'name') and 'dbscan' in self.name.lower():
                    mask = self.labels_ != -1
                    if mask.sum() > 100:
                        metrics['silhouette'] = silhouette_score(X[mask], self.labels_[mask])
                    else:
                        metrics['silhouette'] = 0.0
                else:
                    metrics['silhouette'] = silhouette_score(X, self.labels_)
            else:
                metrics['silhouette'] = 0.0
        except Exception:
            metrics['silhouette'] = 0.0
        
        # Davies-Bouldin Index
        try:
            if len(np.unique(self.labels_)) > 1:
                if hasattr(self, 'name') and 'dbscan' in self.name.lower():
                    mask = self.labels_ != -1
                    if mask.sum() > 100:
                        metrics['davies_bouldin'] = davies_bouldin_score(X[mask], self.labels_[mask])
                    else:
                        metrics['davies_bouldin'] = 999.0
                else:
                    metrics['davies_bouldin'] = davies_bouldin_score(X, self.labels_)
            else:
                metrics['davies_bouldin'] = 999.0
        except Exception:
            metrics['davies_bouldin'] = 999.0
        
        # Calinski-Harabasz Score
        try:
            if len(np.unique(self.labels_)) > 1:
                if hasattr(self, 'name') and 'dbscan' in self.name.lower():
                    mask = self.labels_ != -1
                    if mask.sum() > 100:
                        metrics['calinski_harabasz'] = calinski_harabasz_score(X[mask], self.labels_[mask])
                    else:
                        metrics['calinski_harabasz'] = 0.0
                else:
                    metrics['calinski_harabasz'] = calinski_harabasz_score(X, self.labels_)
            else:
                metrics['calinski_harabasz'] = 0.0
        except Exception:
            metrics['calinski_harabasz'] = 0.0
        
        # Fraud rate
        fraud_mask = self.get_fraud_mask()
        metrics['fraud_rate'] = fraud_mask.mean()
        
        # Training time
        metrics['train_time'] = self.train_time
        
        return metrics


class KMeansClusterer(ClusteringAlgorithm):
    """K-Means clustering"""
    
    def __init__(self, config: dict, k: int = 2, init: str = 'k-means++'):
        super().__init__(f"KMeans_k{k}_{init}", config)
        self.k = k
        self.init = init
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        start = time.time()
        
        self.model = KMeans(
            n_clusters=self.k,
            init=self.init,
            random_state=self.config.get('random_state', 42),
            n_init=self.config.get('n_init', 10),
            max_iter=self.config.get('max_iter', 300)
        )
        
        self.labels_ = self.model.fit_predict(X)
        self.train_time = time.time() - start
        
        return self.labels_
    
    def get_fraud_mask(self) -> np.ndarray:
        """Smallest cluster is fraud"""
        cluster_sizes = np.bincount(self.labels_)
        fraud_cluster = np.argmin(cluster_sizes)
        return (self.labels_ == fraud_cluster).astype(int)


class DBSCANClusterer(ClusteringAlgorithm):
    """DBSCAN clustering"""
    
    def __init__(self, config: dict, eps: float = 0.5, min_samples: int = 20):
        super().__init__(f"DBSCAN_eps{eps}_min{min_samples}", config)
        self.eps = eps
        self.min_samples = min_samples
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        start = time.time()
        
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.config.get('metric', 'euclidean'),
            n_jobs=self.config.get('n_jobs', -1)
        )
        
        self.labels_ = self.model.fit_predict(X)
        self.train_time = time.time() - start
        
        return self.labels_
    
    def get_fraud_mask(self) -> np.ndarray:
        """Outliers (-1) are fraud"""
        return (self.labels_ == -1).astype(int)


class IsolationForestClusterer(ClusteringAlgorithm):
    """Isolation Forest"""
    
    def __init__(self, config: dict, contamination: float = 0.02, n_estimators: int = 100):
        super().__init__(f"IsoForest_cont{contamination}_n{n_estimators}", config)
        self.contamination = contamination
        self.n_estimators = n_estimators
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        start = time.time()
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.config.get('max_samples', 'auto'),
            random_state=self.config.get('random_state', 42),
            n_jobs=self.config.get('n_jobs', -1)
        )
        
        # fit_predict returns 1 (inlier) and -1 (outlier)
        self.labels_ = self.model.fit_predict(X)
        self.train_time = time.time() - start
        
        return self.labels_
    
    def get_fraud_mask(self) -> np.ndarray:
        """Outliers (-1) are fraud, convert to binary"""
        return (self.labels_ == -1).astype(int)


class GMMClusterer(ClusteringAlgorithm):
    """Gaussian Mixture Model"""
    
    def __init__(self, config: dict, n_components: int = 2, covariance_type: str = 'full'):
        super().__init__(f"GMM_n{n_components}_{covariance_type}", config)
        self.n_components = n_components
        self.covariance_type = covariance_type
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        start = time.time()
        
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_init=self.config.get('n_init', 10),
            random_state=self.config.get('random_state', 42)
        )
        
        self.labels_ = self.model.fit_predict(X)
        self.train_time = time.time() - start
        
        return self.labels_
    
    def get_fraud_mask(self) -> np.ndarray:
        """Smallest cluster is fraud"""
        cluster_sizes = np.bincount(self.labels_)
        fraud_cluster = np.argmin(cluster_sizes)
        return (self.labels_ == fraud_cluster).astype(int)


class ClusteringExperiment:
    """Run clustering experiments with multiple algorithms"""
    
    def __init__(self, config: dict, test_mode: bool = False):
        self.config = config
        self.test_mode = test_mode
        self.results = []
        self.scaler = StandardScaler()
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract and standardize clustering features
        
        Returns:
            X_scaled: Standardized features
            X_raw: Raw features (for backup)
        """
        clustering_features = self.config['features']['clustering_features']
        
        # Check all features exist
        missing = [f for f in clustering_features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        X = df[clustering_features].values
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"  Feature matrix: {X_scaled.shape}")
        print(f"  Features: {', '.join(clustering_features)}")
        
        return X_scaled, X
    
    def run_kmeans(self, X: np.ndarray) -> list:
        """Run K-Means experiments"""
        print("\n  🔵 K-Means...")
        
        algo_config = self.config['algorithms']['kmeans']
        if not algo_config.get('enabled', True):
            print("    Skipped (disabled)")
            return []
        
        results = []
        
        if self.test_mode:
            # Test mode: only 1 config
            quick_config = self.config['test_mode']['quick_test']['kmeans']
            k = quick_config['k']
            init = quick_config.get('init', 'k-means++')
            
            clusterer = KMeansClusterer(algo_config, k=k, init=init)
            clusterer.fit_predict(X)
            metrics = clusterer.compute_metrics(X)
            
            results.append({
                'algorithm': 'KMeans',
                'params': {'k': k, 'init': init},
                'metrics': metrics,
                'labels': clusterer.labels_,
                'fraud_mask': clusterer.get_fraud_mask()
            })
            
            print(f"    k={k}: {metrics['fraud_rate']*100:.2f}% fraud, "
                  f"sil={metrics['silhouette']:.3f}")
        else:
            # Full mode: grid search
            for k in algo_config['k_values']:
                init = algo_config.get('init', 'k-means++')
                
                clusterer = KMeansClusterer(algo_config, k=k, init=init)
                clusterer.fit_predict(X)
                metrics = clusterer.compute_metrics(X)
                
                results.append({
                    'algorithm': 'KMeans',
                    'params': {'k': k, 'init': init},
                    'metrics': metrics,
                    'labels': clusterer.labels_,
                    'fraud_mask': clusterer.get_fraud_mask()
                })
                
                print(f"    k={k}, init={init}: {metrics['fraud_rate']*100:.2f}% fraud, "
                      f"sil={metrics['silhouette']:.3f}")
        
        return results
    
    def run_dbscan(self, X: np.ndarray) -> list:
        """Run DBSCAN experiments"""
        print("\n  🎯 DBSCAN...")
        
        algo_config = self.config['algorithms']['dbscan']
        if not algo_config.get('enabled', True):
            print("    Skipped (disabled)")
            return []
        
        results = []
        
        if self.test_mode:
            # Test mode: only 1 config
            quick_config = self.config['test_mode']['quick_test']['dbscan']
            eps = quick_config['eps']
            min_samples = quick_config['min_samples']
            
            clusterer = DBSCANClusterer(algo_config, eps=eps, min_samples=min_samples)
            clusterer.fit_predict(X)
            metrics = clusterer.compute_metrics(X)
            
            results.append({
                'algorithm': 'DBSCAN',
                'params': {'eps': eps, 'min_samples': min_samples},
                'metrics': metrics,
                'labels': clusterer.labels_,
                'fraud_mask': clusterer.get_fraud_mask()
            })
            
            print(f"    eps={eps}, min={min_samples}: {metrics['fraud_rate']*100:.2f}% fraud, "
                  f"sil={metrics['silhouette']:.3f}")
        else:
            # Full mode: grid search
            for eps in algo_config['eps_values']:
                for min_samples in algo_config['min_samples']:
                    clusterer = DBSCANClusterer(algo_config, eps=eps, min_samples=min_samples)
                    clusterer.fit_predict(X)
                    metrics = clusterer.compute_metrics(X)
                    
                    fraud_rate = metrics['fraud_rate']
                    
                    # Only keep results in valid fraud rate range
                    fraud_rate_val = self.config['ensemble']['fraud_rate_validation']
                    if fraud_rate_val['min'] <= fraud_rate <= fraud_rate_val['max']:
                        results.append({
                            'algorithm': 'DBSCAN',
                            'params': {'eps': eps, 'min_samples': min_samples},
                            'metrics': metrics,
                            'labels': clusterer.labels_,
                            'fraud_mask': clusterer.get_fraud_mask()
                        })
                        
                        print(f"    eps={eps}, min={min_samples}: {fraud_rate*100:.2f}% fraud, "
                              f"sil={metrics['silhouette']:.3f} ✓")
                    else:
                        print(f"    eps={eps}, min={min_samples}: {fraud_rate*100:.2f}% fraud (out of range)")
        
        return results
    
    def run_isolation_forest(self, X: np.ndarray) -> list:
        """Run Isolation Forest experiments"""
        print("\n  🌲 Isolation Forest...")
        
        algo_config = self.config['algorithms']['isolation_forest']
        if not algo_config.get('enabled', True):
            print("    Skipped (disabled)")
            return []
        
        results = []
        
        if self.test_mode:
            # Test mode: only 1 config
            quick_config = self.config['test_mode']['quick_test']['isolation_forest']
            contamination = quick_config['contamination']
            n_estimators = quick_config['n_estimators']
            
            clusterer = IsolationForestClusterer(algo_config, contamination=contamination, n_estimators=n_estimators)
            clusterer.fit_predict(X)
            metrics = clusterer.compute_metrics(X)
            
            results.append({
                'algorithm': 'IsolationForest',
                'params': {'contamination': contamination, 'n_estimators': n_estimators},
                'metrics': metrics,
                'labels': clusterer.labels_,
                'fraud_mask': clusterer.get_fraud_mask()
            })
            
            print(f"    cont={contamination}, n={n_estimators}: {metrics['fraud_rate']*100:.2f}% fraud")
        else:
            # Full mode: grid search
            for contamination in algo_config['contamination']:
                for n_estimators in algo_config['n_estimators']:
                    clusterer = IsolationForestClusterer(algo_config, contamination=contamination, n_estimators=n_estimators)
                    clusterer.fit_predict(X)
                    metrics = clusterer.compute_metrics(X)
                    
                    results.append({
                        'algorithm': 'IsolationForest',
                        'params': {'contamination': contamination, 'n_estimators': n_estimators},
                        'metrics': metrics,
                        'labels': clusterer.labels_,
                        'fraud_mask': clusterer.get_fraud_mask()
                    })
                    
                    print(f"    cont={contamination}, n={n_estimators}: {metrics['fraud_rate']*100:.2f}% fraud")
        
        return results
    
    def run_gmm(self, X: np.ndarray) -> list:
        """Run GMM experiments"""
        print("\n  📊 Gaussian Mixture Model...")
        
        algo_config = self.config['algorithms']['gmm']
        if not algo_config.get('enabled', True):
            print("    Skipped (disabled)")
            return []
        
        results = []
        
        if self.test_mode:
            # Test mode: only 1 config
            quick_config = self.config['test_mode']['quick_test']['gmm']
            n_components = quick_config['n_components']
            covariance_type = quick_config['covariance_type']
            
            clusterer = GMMClusterer(algo_config, n_components=n_components, covariance_type=covariance_type)
            clusterer.fit_predict(X)
            metrics = clusterer.compute_metrics(X)
            
            results.append({
                'algorithm': 'GMM',
                'params': {'n_components': n_components, 'covariance_type': covariance_type},
                'metrics': metrics,
                'labels': clusterer.labels_,
                'fraud_mask': clusterer.get_fraud_mask()
            })
            
            print(f"    n={n_components}, cov={covariance_type}: {metrics['fraud_rate']*100:.2f}% fraud, "
                  f"sil={metrics['silhouette']:.3f}")
        else:
            # Full mode: grid search
            for n_components in algo_config['n_components']:
                for covariance_type in algo_config['covariance_type']:
                    clusterer = GMMClusterer(algo_config, n_components=n_components, covariance_type=covariance_type)
                    clusterer.fit_predict(X)
                    metrics = clusterer.compute_metrics(X)
                    
                    results.append({
                        'algorithm': 'GMM',
                        'params': {'n_components': n_components, 'covariance_type': covariance_type},
                        'metrics': metrics,
                        'labels': clusterer.labels_,
                        'fraud_mask': clusterer.get_fraud_mask()
                    })
                    
                    print(f"    n={n_components}, cov={covariance_type}: {metrics['fraud_rate']*100:.2f}% fraud, "
                          f"sil={metrics['silhouette']:.3f}")
        
        return results
    
    def run_all(self, df: pd.DataFrame) -> list:
        """Run all clustering algorithms"""
        print(f"\n{'='*60}")
        print(f"🔬 CLUSTERING EXPERIMENTS")
        print(f"{'='*60}")
        
        if self.test_mode:
            print("  ⚡ TEST MODE: Running quick configs only")
        
        # Prepare features
        X_scaled, X_raw = self.prepare_features(df)
        
        # Run algorithms
        all_results = []
        all_results.extend(self.run_kmeans(X_scaled))
        all_results.extend(self.run_dbscan(X_scaled))
        all_results.extend(self.run_isolation_forest(X_scaled))
        all_results.extend(self.run_gmm(X_scaled))
        
        print(f"\n✅ Total experiments: {len(all_results)}")
        
        self.results = all_results
        return all_results
    
    def get_best_result(self, metric: str = 'silhouette') -> dict:
        """Get best result by metric"""
        if not self.results:
            return None
        
        if metric == 'davies_bouldin':
            # Lower is better
            best = min(self.results, key=lambda x: x['metrics'][metric])
        else:
            # Higher is better
            best = max(self.results, key=lambda x: x['metrics'][metric])
        
        return best
    
    def create_ensemble(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create ensemble fraud labels from all algorithms"""
        print(f"\n{'='*60}")
        print(f"🗳️  ENSEMBLE VOTING")
        print(f"{'='*60}")
        
        weights = self.config['ensemble']['weights']
        
        # Get best result from each algorithm
        best_by_algo = {}
        for algo_name in ['kmeans', 'dbscan', 'isolation_forest', 'gmm']:
            algo_results = [r for r in self.results if r['algorithm'].lower().startswith(algo_name.split('_')[0])]
            if algo_results:
                best = max(algo_results, key=lambda x: x['metrics']['silhouette'])
                best_by_algo[algo_name] = best
                print(f"  {algo_name:20s}: {best['params']}")
        
        # Weighted voting
        n_samples = len(self.results[0]['fraud_mask'])
        fraud_score = np.zeros(n_samples)
        
        for algo_name, result in best_by_algo.items():
            weight = weights.get(algo_name, 0.25)
            fraud_score += weight * result['fraud_mask']
        
        # Apply threshold
        threshold = self.config['ensemble']['threshold']
        fraud_label = (fraud_score >= threshold).astype(int)
        
        fraud_rate = fraud_label.mean()
        print(f"\n  Threshold: {threshold}")
        print(f"  Final fraud rate: {fraud_rate*100:.2f}%")
        print(f"  Fraud count: {fraud_label.sum():,}")
        
        return fraud_label, fraud_score