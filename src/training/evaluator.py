"""
GNN evaluation with comprehensive metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from typing import Dict


class GNNEvaluator:
    """Evaluate GNN models"""
    
    def __init__(self, model, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
    
    @torch.no_grad()
    def evaluate_full(self, graph, test_data: Dict) -> Dict:
        """Comprehensive evaluation"""
        self.model.eval()
        
        print(f"\n{'='*60}")
        print(f"📊 EVALUATION")
        print(f"{'='*60}")
        
        # Move to device
        graph = graph.to(self.device)
        test_cust_idx = torch.LongTensor(test_data['customer_idx']).to(self.device)
        test_prod_idx = torch.LongTensor(test_data['product_idx']).to(self.device)
        test_store_idx = torch.LongTensor(test_data['store_idx']).to(self.device)
        test_labels = test_data['labels']
        
        # Forward pass
        x_dict = self.model(graph.x_dict, graph.edge_index_dict)
        logits = self.model.predict_transaction(x_dict, test_cust_idx, test_prod_idx, test_store_idx)
        
        # Predictions
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        
        # Metrics
        precision = precision_score(test_labels, preds, zero_division=0)
        recall = recall_score(test_labels, preds, zero_division=0)
        f1 = f1_score(test_labels, preds, zero_division=0)
        auc = roc_auc_score(test_labels, probs) if len(np.unique(test_labels)) > 1 else 0.5
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, preds)
        
        # Top-K precision
        top_k_metrics = self._compute_top_k(test_labels, probs)
        
        results = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'top_k': top_k_metrics,
            'predictions': preds,
            'probabilities': probs
        }
        
        # Print results
        self._print_results(results)
        
        return results
    
    def _compute_top_k(self, labels: np.ndarray, probs: np.ndarray) -> Dict:
        """Compute Top-K precision"""
        top_k_results = {}
        
        for k in [100, 500, 1000]:
            if len(probs) >= k:
                top_k_idx = np.argsort(probs)[-k:]
                precision_at_k = labels[top_k_idx].mean()
                top_k_results[f'top_{k}'] = precision_at_k
        
        return top_k_results
    
    def _print_results(self, results: Dict):
        """Print evaluation results"""
        print(f"\n  📈 Classification Metrics:")
        print(f"    Precision: {results['precision']:.4f}")
        print(f"    Recall: {results['recall']:.4f}")
        print(f"    F1-Score: {results['f1']:.4f}")
        print(f"    AUC-ROC: {results['auc']:.4f}")
        
        print(f"\n  📊 Confusion Matrix:")
        cm = results['confusion_matrix']
        print(f"    TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
        print(f"    FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")
        
        if results['top_k']:
            print(f"\n  🎯 Top-K Precision:")
            for k, precision in results['top_k'].items():
                print(f"    {k.replace('_', '-').upper()}: {precision:.4f}")