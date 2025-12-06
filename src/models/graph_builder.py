"""
Heterogeneous Graph Builder
Constructs graph structure from transaction data
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, Tuple
import pickle
from pathlib import Path


class GraphBuilder:
    """
    Build heterogeneous graph from transaction data
    
    Node types: customer, product, store
    Edge types: customer-buys-product, customer-visits-store, product-sold_at-store
    """
    
    def __init__(self, config: dict):
        self.config = config
    
    def build_graph(self, df: pd.DataFrame, fraud_labels: np.ndarray) -> Tuple[HeteroData, pd.DataFrame]:
        """
        Build heterogeneous graph from transaction data
        
        Args:
            df: Transaction dataframe with engineered features
            fraud_labels: Binary fraud labels for each transaction
        
        Returns:
            Tuple of (HeteroData graph, transaction_mapping dict)
        """
        
        print(f"\n{'='*60}")
        print(f"🏗️  BUILDING HETEROGENEOUS GRAPH")
        print(f"{'='*60}")
        
        # Add fraud labels to dataframe
        df = df.copy()
        df['fraud_label'] = fraud_labels
        
        # Filter out invalid transactions (if any)
        df = df[(df['cust_id'] > 0) & (df['product_code'] > 0) & (df['store_code'] > 0)]
        print(f"  Transactions (after filtering): {len(df):,}")
        
        # Create node mappings
        print(f"\n  📍 Creating node mappings...")
        df = self._create_node_mappings(df)
        
        # Create node features
        print(f"  🎯 Creating node features...")
        customer_features = self._create_customer_features(df)
        product_features = self._create_product_features(df)
        store_features = self._create_store_features(df)
        
        # Create edges
        print(f"  🔗 Creating edges...")
        edge_index_dict = self._create_edges(df)
        
        # Assemble graph
        print(f"  🏗️  Assembling graph...")
        graph = self._assemble_graph(
            customer_features, product_features, store_features,
            edge_index_dict, df
        )
        
        # Create transaction mapping
        transaction_mapping = self._create_transaction_mapping(df)
        
        # Print summary
        num_customers = df['customer_idx'].nunique()
        num_products = df['product_idx'].nunique()
        num_stores = df['store_idx'].nunique()
        total_edges = sum([edge_index.shape[1] for edge_index in edge_index_dict.values()])
        fraud_rate = df['fraud_label'].mean()
        
        print(f"\n✅ Graph built successfully")
        print(f"  Nodes: {num_customers:,} customers, {num_products:,} products, {num_stores:,} stores")
        print(f"  Edges: {total_edges:,} total")
        print(f"  Fraud rate: {fraud_rate:.2%}")
        
        # Save graph and mapping
        self._save_graph(graph, transaction_mapping)
        
        return graph, transaction_mapping
    
    def _create_node_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create index mappings for each node type"""
        
        # Customer mapping
        unique_customers = df['cust_id'].unique()
        customer_mapping = {cust_id: idx for idx, cust_id in enumerate(unique_customers)}
        df['customer_idx'] = df['cust_id'].map(customer_mapping)
        
        # Product mapping
        unique_products = df['product_code'].unique()
        product_mapping = {prod_id: idx for idx, prod_id in enumerate(unique_products)}
        df['product_idx'] = df['product_code'].map(product_mapping)
        
        # Store mapping
        unique_stores = df['store_code'].unique()
        store_mapping = {store_id: idx for idx, store_id in enumerate(unique_stores)}
        df['store_idx'] = df['store_code'].map(store_mapping)
        
        print(f"    Customers: {len(unique_customers):,}")
        print(f"    Products: {len(unique_products):,}")
        print(f"    Stores: {len(unique_stores):,}")
        
        return df
    
    def _create_customer_features(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Create customer node features from transaction data
        
        Args:
            df: DataFrame with customer_idx column
        
        Returns:
            Tensor of customer features [num_customers, num_features]
        """
        
        # Define feature columns (must match preprocessor output)
        feature_cols = [
            'customer_total_spending',
            'customer_transaction_count',
            'avg_transaction_value',
            'customer_unique_products',
            'unique_stores',
            'return_rate',
            'transaction_velocity',
            'customer_discount_rate',
            'customer_campaign_rate'
        ]
        
        # Check which features exist
        existing_cols = [col for col in feature_cols if col in df.columns]
        missing_cols = [col for col in feature_cols if col not in df.columns]
        
        if missing_cols:
            print(f"    ⚠️  Missing customer features: {missing_cols}")
        
        # Aggregate customer-level features
        features = df.groupby('customer_idx')[existing_cols].mean().fillna(0)
        
        # Convert to tensor
        return torch.FloatTensor(features.values)
    
    def _create_product_features(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Create product node features from transaction data
        
        Args:
            df: DataFrame with product_idx column
        
        Returns:
            Tensor of product features [num_products, num_features]
        """
        
        # Define feature columns
        feature_cols = [
            'product_popularity',
            'product_avg_price',
            'product_unique_customers',
            'product_discount_frequency'
        ]
        
        # Check which features exist
        existing_cols = [col for col in feature_cols if col in df.columns]
        missing_cols = [col for col in feature_cols if col not in df.columns]
        
        if missing_cols:
            print(f"    ⚠️  Missing product features: {missing_cols}")
        
        # Aggregate product-level features
        features = df.groupby('product_idx')[existing_cols].mean().fillna(0)
        
        # Convert to tensor
        return torch.FloatTensor(features.values)
    
    def _create_store_features(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Create store node features from transaction data
        
        Args:
            df: DataFrame with store_idx column
        
        Returns:
            Tensor of store features [num_stores, num_features]
        """
        
        # Check if we have multiple stores
        num_stores = df['store_code'].nunique()
        
        if num_stores == 1:
            # Single store: create dummy features
            print(f"    ⚠️  Single store detected, using dummy features")
            return torch.FloatTensor([[1.0, 1.0, 1.0]])
        
        # Aggregate store-level features
        features = df.groupby('store_idx').agg({
            'total_price': ['sum', 'mean'],
            'customer_idx': 'nunique',
            'product_idx': 'nunique'
        }).fillna(0)
        
        # Flatten multi-level columns
        features.columns = ['_'.join(str(col)).strip() for col in features.columns.values]
        
        # Convert to tensor
        return torch.FloatTensor(features.values)
    

    def _create_edges(self, df: pd.DataFrame) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """
        Create edge indices for all edge types (including reverse edges)
        
        Args:
            df: DataFrame with node index columns
        
        Returns:
            Dictionary of edge indices for each edge type
        """
        
        edge_index_dict = {}
        
        # Customer-Product edges (buys) + Reverse
        customer_product = df[['customer_idx', 'product_idx']].drop_duplicates()
        edge_index_dict[('customer', 'buys', 'product')] = torch.LongTensor([
            customer_product['customer_idx'].values,
            customer_product['product_idx'].values
        ])
        edge_index_dict[('product', 'bought_by', 'customer')] = torch.LongTensor([
            customer_product['product_idx'].values,
            customer_product['customer_idx'].values
        ])
        print(f"    Customer-Product: {customer_product.shape[0]:,}")
        
        # Customer-Store edges (visits) + Reverse
        customer_store = df[['customer_idx', 'store_idx']].drop_duplicates()
        edge_index_dict[('customer', 'visits', 'store')] = torch.LongTensor([
            customer_store['customer_idx'].values,
            customer_store['store_idx'].values
        ])
        edge_index_dict[('store', 'visited_by', 'customer')] = torch.LongTensor([
            customer_store['store_idx'].values,
            customer_store['customer_idx'].values
        ])
        print(f"    Customer-Store: {customer_store.shape[0]:,}")
        
        # Product-Store edges (sold_at) + Reverse
        product_store = df[['product_idx', 'store_idx']].drop_duplicates()
        edge_index_dict[('product', 'sold_at', 'store')] = torch.LongTensor([
            product_store['product_idx'].values,
            product_store['store_idx'].values
        ])
        edge_index_dict[('store', 'sells', 'product')] = torch.LongTensor([
            product_store['store_idx'].values,
            product_store['product_idx'].values
        ])
        print(f"    Product-Store: {product_store.shape[0]:,}")
        
        return edge_index_dict


    def _assemble_graph(
        self,
        customer_features: torch.Tensor,
        product_features: torch.Tensor,
        store_features: torch.Tensor,
        edge_index_dict: Dict,
        df: pd.DataFrame
    ) -> HeteroData:
        """
        Assemble heterogeneous graph data object
        
        Args:
            customer_features: Customer node features
            product_features: Product node features
            store_features: Store node features
            edge_index_dict: Dictionary of edge indices
            df: Transaction dataframe with labels
        
        Returns:
            HeteroData object
        """
        
        graph = HeteroData()
        
        # Add node features
        graph['customer'].x = customer_features
        graph['product'].x = product_features
        graph['store'].x = store_features
        
        # Add edges
        for edge_type, edge_index in edge_index_dict.items():
            graph[edge_type].edge_index = edge_index
        
        # Add fraud labels at transaction level
        # We'll aggregate to customer level for training
        customer_fraud = df.groupby('customer_idx')['fraud_label'].max().values
        graph['customer'].y = torch.LongTensor(customer_fraud)
        
        return graph
    

    def _create_transaction_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create mapping from transactions to graph nodes
        
        Args:
            df: Transaction dataframe
        
        Returns:
            DataFrame with transaction-to-node mappings
        """
        
        # Create DataFrame with essential columns
        mapping_df = pd.DataFrame({
            'trans_id': df['trans_id'].values if 'trans_id' in df.columns else np.arange(len(df)),
            'customer_idx': df['customer_idx'].values,
            'product_idx': df['product_idx'].values,
            'store_idx': df['store_idx'].values,
            'fraud_label': df['fraud_label'].values
        })
        
        return mapping_df

    
    def _save_graph(self, graph: HeteroData, transaction_mapping: pd.DataFrame):
        """Save graph and mapping to disk"""
        
        # Get save paths from config
        processed_dir = Path(self.config['data']['processed_data_path'])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save graph
        graph_path = processed_dir / "hetero_graph.pt"
        torch.save(graph, graph_path)
        
        # Save mapping
        mapping_path = processed_dir / "transaction_mapping.pkl"
        with open(mapping_path, 'wb') as f:
            pickle.dump(transaction_mapping, f)
        
        print(f"\n✅ Saved:")
        print(f"  Graph: {graph_path}")
        print(f"  Transaction mapping: {mapping_path}")
    
    def load_graph(self) -> Tuple[HeteroData, Dict]:
        """
        Load saved graph and mapping
        
        Returns:
            Tuple of (graph, transaction_mapping)
        """
        
        processed_dir = Path(self.config['data']['processed_data_path'])
        
        # Load graph
        graph_path = processed_dir / "hetero_graph.pt"
        graph = torch.load(graph_path)
        
        # Load mapping
        mapping_path = processed_dir / "transaction_mapping.pkl"
        with open(mapping_path, 'rb') as f:
            transaction_mapping = pickle.load(f)
        
        return graph, transaction_mapping