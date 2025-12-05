# scripts/predict.py
"""
Fraud prediction on new transactions
Usage:
  python scripts/predict.py --input data/new_transactions.csv --model outputs/models/sage_final.pt
"""

import sys
sys.path.append('.')

import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import pickle
from pathlib import Path

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.features.feature_engineer import FeatureEngineer
from src.models.gnn_models import GraphSAGE, GAT, GCN


def load_model(model_path: str, config: dict, device: str):
    """Load trained GNN model"""
    # Detect model type from filename
    model_name = Path(model_path).stem.split('_')[0]
    
    if model_name == 'sage':
        model = GraphSAGE(config)
    elif model_name == 'gat':
        model = GAT(config)
    elif model_name == 'gcn':
        model = GCN(config)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def map_to_nodes(ids: pd.Series, mapping: dict, default_idx: int = 0):
    """Map transaction IDs to node indices"""
    return torch.LongTensor([
        mapping.get(id_val, default_idx) for id_val in ids
    ])


def predict_fraud(
    input_csv: str,
    model_path: str,
    output_csv: str = None,
    threshold: float = 0.5
):
    """
    Predict fraud on new transactions
    
    Args:
        input_csv: Path to CSV with new transactions
        model_path: Path to trained model
        output_csv: Path to save predictions (default: input_predictions.csv)
        threshold: Probability threshold for fraud (default: 0.5)
    """
    
    print("\n" + "=" * 80)
    print("🔮 FRAUD PREDICTION")
    print("=" * 80)
    
    # Setup
    loader = ConfigLoader()
    config = loader.load_all()
    logger = setup_logger('predict', config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Input: {input_csv}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Device: {device}")
    
    # Load new transactions
    print("\n1️⃣  Loading new transactions...")
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df):,} transactions")
    
    # Feature engineering
    print("\n2️⃣  Engineering features...")
    feature_eng = FeatureEngineer(config)
    df = feature_eng.engineer_features(df)
    logger.info(f"Features: {len(df.columns)} columns")
    
    # Load graph and mappings
    print("\n3️⃣  Loading graph and mappings...")
    graph_path = Path(config['data']['processed_data_path']) / 'hetero_graph.pt'
    mappings_path = Path(config['data']['processed_data_path']) / 'node_mappings.pkl'
    
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")
    if not mappings_path.exists():
        raise FileNotFoundError(f"Mappings not found: {mappings_path}")
    
    graph = torch.load(graph_path, map_location=device)
    with open(mappings_path, 'rb') as f:
        node_mappings = pickle.load(f)
    
    logger.info("Graph and mappings loaded")
    
    # Map transaction IDs to node indices
    print("\n4️⃣  Mapping to graph nodes...")
    customer_idx = map_to_nodes(
        df['cust_id'], 
        node_mappings['customer_to_idx']
    ).to(device)
    
    product_idx = map_to_nodes(
        df['product_code'], 
        node_mappings['product_to_idx']
    ).to(device)
    
    store_idx = map_to_nodes(
        df['store_code'], 
        node_mappings['store_to_idx']
    ).to(device)
    
    logger.info(f"Mapped {len(customer_idx)} transactions")
    
    # Load model
    print("\n5️⃣  Loading model...")
    model = load_model(model_path, config, device)
    logger.info(f"Model loaded: {Path(model_path).stem}")
    
    # Predict
    print("\n6️⃣  Running predictions...")
    with torch.no_grad():
        # Get node embeddings
        x_dict = model(graph.x_dict, graph.edge_index_dict)
        
        # Predict transactions
        logits = model.predict_transaction(
            x_dict, 
            customer_idx, 
            product_idx, 
            store_idx
        )
        
        # Get probabilities
        proba = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        predictions = (proba >= threshold).astype(int)
    
    logger.info("Predictions complete")
    
    # Add results to dataframe
    df['fraud_probability'] = proba
    df['fraud_prediction'] = predictions
    df['fraud_risk_level'] = pd.cut(
        proba, 
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    # Save results
    if output_csv is None:
        output_csv = str(Path(input_csv).stem) + '_predictions.csv'
    
    print(f"\n7️⃣  Saving results...")
    df.to_csv(output_csv, index=False)
    logger.info(f"Results saved: {output_csv}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PREDICTION COMPLETE")
    print("=" * 80)
    
    print(f"\n📊 Summary:")
    print(f"   Total transactions: {len(df):,}")
    print(f"   Fraud detected: {predictions.sum():,} ({predictions.mean()*100:.2f}%)")
    print(f"   High risk: {(df['fraud_risk_level'] == 'High').sum():,}")
    print(f"   Medium risk: {(df['fraud_risk_level'] == 'Medium').sum():,}")
    print(f"   Low risk: {(df['fraud_risk_level'] == 'Low').sum():,}")
    
    print(f"\n📁 Output: {output_csv}")
    
    # Top fraud cases
    print("\n🚨 Top 10 Fraud Cases:")
    top_fraud = df.nlargest(10, 'fraud_probability')[
        ['trans_id', 'cust_id', 'product_code', 'total_price', 'fraud_probability']
    ]
    print(top_fraud.to_string(index=False))
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Fraud prediction on new transactions')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--model', '-m', required=True, help='Trained model path')
    parser.add_argument('--output', '-o', help='Output CSV file')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Fraud threshold (0-1)')
    
    args = parser.parse_args()
    
    try:
        predict_fraud(
            input_csv=args.input,
            model_path=args.model,
            output_csv=args.output,
            threshold=args.threshold
        )
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()