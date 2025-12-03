"""
GNN models: GraphSAGE, GAT, GCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, HeteroConv


class HeteroGNN(nn.Module):
    """Base heterogeneous GNN"""
    
    def __init__(self, config: dict, conv_type: str = 'sage'):
        super().__init__()
        
        self.config = config
        self.conv_type = conv_type
        
        # Model config
        model_config = config['architectures'][conv_type]
        self.hidden_channels = model_config['hidden_channels']
        self.num_layers = model_config['num_layers']
        self.dropout = model_config['dropout']
        
        # Input projections
        self.customer_proj = nn.Linear(6, self.hidden_channels)  # 6 customer features
        self.product_proj = nn.Linear(4, self.hidden_channels)   # 4 product features
        self.store_proj = nn.Linear(3, self.hidden_channels)     # 3 store features
        
        # Graph convolutions
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(self.num_layers):
            # Create heterogeneous convolution
            conv_dict = self._create_conv_dict(conv_type, model_config)
            conv = HeteroConv(conv_dict, aggr='mean')
            self.convs.append(conv)
            
            # Layer normalization for each node type
            norm_dict = nn.ModuleDict({
                'customer': nn.LayerNorm(self.hidden_channels),
                'product': nn.LayerNorm(self.hidden_channels),
                'store': nn.LayerNorm(self.hidden_channels)
            })
            self.norms.append(norm_dict)
        
        # Transaction classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_channels * 3, self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels // 2, 2)
        )
    
    def _create_conv_dict(self, conv_type: str, model_config: dict) -> dict:
        """Create convolution dict for all edge types"""
        conv_dict = {}
        
        edge_types = [
            ('customer', 'buys', 'product'),
            ('product', 'bought_by', 'customer'),
            ('customer', 'visits', 'store'),
            ('store', 'visited_by', 'customer'),
            ('product', 'sold_at', 'store'),
            ('store', 'sells', 'product')
        ]
        
        for edge_type in edge_types:
            if conv_type == 'sage':
                conv_dict[edge_type] = SAGEConv(
                    self.hidden_channels, 
                    self.hidden_channels,
                    aggr=model_config.get('aggregation', 'mean')
                )
            elif conv_type == 'gat':
                conv_dict[edge_type] = GATConv(
                    self.hidden_channels,
                    self.hidden_channels,
                    heads=model_config.get('num_heads', 8),
                    concat=False,
                    dropout=self.dropout
                )
            elif conv_type == 'gcn':
                conv_dict[edge_type] = GCNConv(
                    self.hidden_channels,
                    self.hidden_channels,
                    add_self_loops=model_config.get('add_self_loops', True)
                )
        
        return conv_dict
    
    def forward(self, x_dict, edge_index_dict):
        """Forward pass through GNN layers"""
        # Project input features
        x_dict = {
            'customer': self.customer_proj(x_dict['customer']),
            'product': self.product_proj(x_dict['product']),
            'store': self.store_proj(x_dict['store'])
        }
        
        # Graph convolutions
        for i, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict_new = conv(x_dict, edge_index_dict)
            
            # Apply normalization and activation
            for node_type in x_dict_new:
                x_dict_new[node_type] = norm_dict[node_type](x_dict_new[node_type])
                x_dict_new[node_type] = F.relu(x_dict_new[node_type])
                x_dict_new[node_type] = F.dropout(
                    x_dict_new[node_type],
                    p=self.dropout,
                    training=self.training
                )
            
            x_dict = x_dict_new
        
        return x_dict
    
    def predict_transaction(self, x_dict, customer_idx, product_idx, store_idx):
        """Predict fraud for transactions"""
        # Get embeddings
        customer_emb = x_dict['customer'][customer_idx]
        product_emb = x_dict['product'][product_idx]
        store_emb = x_dict['store'][store_idx]
        
        # Concatenate
        trans_emb = torch.cat([customer_emb, product_emb, store_emb], dim=1)
        
        # Classify
        logits = self.classifier(trans_emb)
        return logits


class GraphSAGE(HeteroGNN):
    """GraphSAGE model"""
    def __init__(self, config: dict):
        super().__init__(config, conv_type='sage')


class GAT(HeteroGNN):
    """Graph Attention Network"""
    def __init__(self, config: dict):
        super().__init__(config, conv_type='gat')


class GCN(HeteroGNN):
    """Graph Convolutional Network"""
    def __init__(self, config: dict):
        super().__init__(config, conv_type='gcn')