#!/usr/bin/env python3
"""
Test Phase 3: Clustering experiments
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.data.loader import load_data
from src.data.preprocessor import FeatureEngineer
from src.labeling.clustering import ClusteringExperiment
from src.labeling.visualizer import ClusteringVisualizer


def main():
    print("="*80)
    print("🔬 PHASE 3: CLUSTERING EXPERIMENTS (TEST MODE)")
    print("="*80)
    
    # 1. Load configs
    print("\n1️⃣  Loading configs...")
    loader = ConfigLoader()
    base_config = loader.load('base')
    clustering_config = loader.load('clustering')
    
    # Merge configs
    config = {**base_config, **clustering_config}
    
    # Force test mode
    config['test_mode']['enabled'] = True
    print("  ⚡ Test mode: ENABLED (quick configs only)")
    
    # 2. Set seed
    set_seed(config['project']['random_seed'])
    
    # 3. Logger
    logger = get_logger('phase3_clustering', config['logging'])
    logger.info("Phase 3 started")
    
    # 4. Load data
    logger.info("Loading data...")
    df = load_data(config, nrows=10000)
    
    # 5. Feature engineering
    logger.info("Engineering features...")
    engineer = FeatureEngineer(config)
    df = engineer.engineer_features(df)
    
    # 6. Run clustering experiments
    logger.info("Running clustering experiments...")
    experiment = ClusteringExperiment(config, test_mode=True)
    results = experiment.run_all(df)
    
    # 7. Create ensemble
    logger.info("Creating ensemble labels...")
    fraud_label, fraud_score = experiment.create_ensemble()
    
    # Add to dataframe
    df['fraud_label'] = fraud_label
    df['fraud_score'] = fraud_score
    
    # 8. Visualizations
    logger.info("Creating visualizations...")
    visualizer = ClusteringVisualizer(config)
    
    # Get best result from each algorithm
    best_results = []
    for algo_name in ['KMeans', 'DBSCAN', 'IsolationForest', 'GMM']:
        algo_results = [r for r in results if r['algorithm'] == algo_name]
        if algo_results:
            best = max(algo_results, key=lambda x: x['metrics']['silhouette'])
            best_results.append(best)
    
    # Plot each best result
    for result in best_results:
        # Extract features again for visualization
        X_scaled, _ = experiment.prepare_features(df)
        
        visualizer.plot_all(
            X=X_scaled,
            labels=result['labels'],
            fraud_mask=result['fraud_mask'],
            algorithm_name=result['algorithm'],
            save_dir='outputs/figures/clustering'
        )
    
    # Comparison plot
    visualizer.plot_comparison_summary(results, 'outputs/figures/clustering')
    
    # 9. Summary
    print(f"\n{'='*80}")
    print(f"✅ PHASE 3 TEST COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Results:")
    print(f"  Total experiments: {len(results)}")
    print(f"  Best by silhouette: {experiment.get_best_result('silhouette')['algorithm']}")
    print(f"  Final fraud labels: {fraud_label.sum():,} / {len(fraud_label):,} ({fraud_label.mean()*100:.2f}%)")
    
    print(f"\n📁 Outputs:")
    print(f"  Figures: outputs/figures/clustering/")
    print(f"  Logs: outputs/logs/")
    
    logger.info("Phase 3 completed successfully")


if __name__ == "__main__":
    main()