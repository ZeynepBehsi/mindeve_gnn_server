"""
Heterogeneous graph construction for fraud detection
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
import pickle


class GraphBuilder:
    """Build heterogeneous graph from transaction data"""
    
    def __init__(self, config: dict):
        self.config = config
        self.graph_config = config['graph_construction']
        
        # Mappings
        self.customer_to_idx = {}
        self.product_to_idx = {}
        self.store_to_idx = {}
    
    def build_graph(self, df: pd.DataFrame, fraud_labels: np.ndarray) -> HeteroData:
        """
        Build heterogeneous graph from transaction data
        
        Args:
            df: DataFrame with features
            fraud_labels: Binary fraud labels (from clustering)
        
        Returns:
            HeteroData graph object
        """
        print(f"\n{'='*60}")
        print(f"🏗️  BUILDING HETEROGENEOUS GRAPH")
        print(f"{'='*60}")
        
        # Filter guest users
        df = df[df['cust_id'] != 0].copy()
        fraud_labels = fraud_labels[df.index]
        df = df.reset_index(drop=True)
        
        print(f"  Transactions (after filtering): {len(df):,}")
        
        # 1. Create node mappings
        print(f"\n  📍 Creating node mappings...")
        self._create_mappings(df)
        
        # 2. Create node features
        print(f"  🎯 Creating node features...")
        customer_features = self._create_customer_features(df)
        product_features = self._create_product_features(df)
        store_features = self._create_store_features(df)
        
        # 3. Create edges
        print(f"  🔗 Creating edges...")
        edge_index_dict = self._create_edges(df)
        
        # 4. Build HeteroData object
        print(f"  🏗️  Assembling graph...")
        graph = HeteroData()
        
        # Add node features
        graph['customer'].x = customer_features
        graph['customer'].num_nodes = len(self.customer_to_idx)
        
        graph['product'].x = product_features
        graph['product'].num_nodes = len(self.product_to_idx)
        
        graph['store'].x = store_features
        graph['store'].num_nodes = len(self.store_to_idx)
        
        # Add edges
        for edge_type, edge_index in edge_index_dict.items():
            graph[edge_type].edge_index = edge_index
        
        # Add transaction-level labels
        transaction_mapping = self._create_transaction_mapping(df, fraud_labels)
        
        print(f"\n✅ Graph built successfully")
        print(f"  Nodes: {graph['customer'].num_nodes:,} customers, "
              f"{graph['product'].num_nodes:,} products, "
              f"{graph['store'].num_nodes:,} stores")
        print(f"  Edges: {sum(ei.shape[1] for ei in edge_index_dict.values()):,} total")
        print(f"  Fraud rate: {fraud_labels.mean()*100:.2f}%")
        
        return graph, transaction_mapping
    
    def _create_mappings(self, df: pd.DataFrame):
        """Create node ID mappings"""
        unique_customers = df['cust_id'].unique()
        unique_products = df['product_code'].unique()
        unique_stores = df['store_code'].unique()
        
        self.customer_to_idx = {cust: idx for idx, cust in enumerate(unique_customers)}
        self.product_to_idx = {prod: idx for idx, prod in enumerate(unique_products)}
        self.store_to_idx = {store: idx for idx, store in enumerate(unique_stores)}
        
        # Add to dataframe
        df['customer_idx'] = df['cust_id'].map(self.customer_to_idx)
        df['product_idx'] = df['product_code'].map(self.product_to_idx)
        df['store_idx'] = df['store_code'].map(self.store_to_idx)
        
        print(f"    Customers: {len(self.customer_to_idx):,}")
        print(f"    Products: {len(self.product_to_idx):,}")
        print(f"    Stores: {len(self.store_to_idx):,}")
    
    def _create_customer_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Create customer node features"""
        features = df.groupby('customer_idx').agg({
            'transaction_count': 'first',
            'total_spent': 'first',
            'avg_transaction_value': 'first',
            'transaction_velocity': 'first',
            'unique_stores': 'first',
            'return_rate': 'first'
        }).fillna(0)
        
        # Normalize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.values)
        
        return torch.FloatTensor(features_scaled)
    
    def _create_product_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Create product node features"""
        features = df.groupby('product_idx').agg({
            'unit_price': 'mean',
            'amount': 'mean',
            'total_price': 'mean',
            'product_popularity': 'first'
        }).fillna(0)
        
        # Normalize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.values)
        
        return torch.FloatTensor(features_scaled)
    
    def _create_store_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Create store node features"""
        features = df.groupby('store_idx').agg({
            'trans_id': 'count',
            'total_price': 'sum',
            'cust_id': 'nunique'
        }).fillna(0)
        
        features.columns = ['transaction_count', 'total_revenue', 'unique_customers']
        
        # Normalize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.values)
        
        return torch.FloatTensor(features_scaled)
    
    def _create_edges(self, df: pd.DataFrame) -> Dict:
        """Create heterogeneous edges"""
        edge_index_dict = {}
        
        # Customer -> Product (buys)
        cp_edges = df[['customer_idx', 'product_idx']].drop_duplicates()
        edge_index_dict[('customer', 'buys', 'product')] = torch.LongTensor([
            cp_edges['customer_idx'].values,
            cp_edges['product_idx'].values
        ])
        
        # Product -> Customer (bought_by)
        edge_index_dict[('product', 'bought_by', 'customer')] = torch.LongTensor([
            cp_edges['product_idx'].values,
            cp_edges['customer_idx'].values
        ])
        
        # Customer -> Store (visits)
        cs_edges = df[['customer_idx', 'store_idx']].drop_duplicates()
        edge_index_dict[('customer', 'visits', 'store')] = torch.LongTensor([
            cs_edges['customer_idx'].values,
            cs_edges['store_idx'].values
        ])
        
        # Store -> Customer (visited_by)
        edge_index_dict[('store', 'visited_by', 'customer')] = torch.LongTensor([
            cs_edges['store_idx'].values,
            cs_edges['customer_idx'].values
        ])
        
        # Product -> Store (sold_at)
        ps_edges = df[['product_idx', 'store_idx']].drop_duplicates()
        edge_index_dict[('product', 'sold_at', 'store')] = torch.LongTensor([
            ps_edges['product_idx'].values,
            ps_edges['store_idx'].values
        ])
        
        # Store -> Product (sells)
        edge_index_dict[('store', 'sells', 'product')] = torch.LongTensor([
            ps_edges['store_idx'].values,
            ps_edges['product_idx'].values
        ])
        
        print(f"    Customer-Product: {cp_edges.shape[0]:,}")
        print(f"    Customer-Store: {cs_edges.shape[0]:,}")
        print(f"    Product-Store: {ps_edges.shape[0]:,}")
        
        return edge_index_dict
    
    def _create_transaction_mapping(self, df: pd.DataFrame, 
                                   fraud_labels: np.ndarray) -> pd.DataFrame:
        """Create transaction-level mapping for training"""
        mapping = df[['trans_id', 'customer_idx', 'product_idx', 'store_idx']].copy()
        mapping['fraud_label'] = fraud_labels
        
        return mapping
    
    def save_graph(self, graph: HeteroData, transaction_mapping: pd.DataFrame, 
                   save_dir: str):
        """Save graph and mappings"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save graph
        torch.save(graph, save_dir / 'hetero_graph.pt')
        
        # Save transaction mapping
        transaction_mapping.to_pickle(save_dir / 'transaction_mapping.pkl')
        
        # Save node mappings
        mappings = {
            'customer_to_idx': self.customer_to_idx,
            'product_to_idx': self.product_to_idx,
            'store_to_idx': self.store_to_idx
        }
        with open(save_dir / 'node_mappings.pkl', 'wb') as f:
            pickle.dump(mappings, f)
        
        print(f"\n✅ Saved:")
        print(f"  Graph: {save_dir / 'hetero_graph.pt'}")
        print(f"  Transaction mapping: {save_dir / 'transaction_mapping.pkl'}")
        print(f"  Node mappings: {save_dir / 'node_mappings.pkl'}")