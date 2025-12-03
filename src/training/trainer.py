"""
GNN training with SMOTE-ENN and LR scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
import time
from typing import Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE


class GNNTrainer:
    """Train GNN models with advanced techniques"""
    
    def __init__(self, model, config: dict, device: str = 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Training config
        self.train_config = config['training']
        self.num_epochs = self.train_config['num_epochs']
        self.batch_size = self.train_config['batch_size']
        self.learning_rate = self.train_config['learning_rate']
        
        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.train_config['weight_decay']
        )
        
        # Learning rate schedulers
        self.setup_schedulers()
        
        # Early stopping
        self.patience = self.train_config['early_stopping']['patience']
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_state = None
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_auc': [],
            'learning_rate': []
        }
    
    def setup_schedulers(self):
        """Setup learning rate schedulers"""
        sched_config = self.train_config['lr_scheduler']
        
        # Warmup scheduler
        warmup_epochs = sched_config['warmup']['epochs']
        self.warmup_scheduler = None
        if warmup_epochs > 0:
            warmup_lr = sched_config['warmup']['start_lr']
            def warmup_fn(epoch):
                return (self.learning_rate - warmup_lr) / warmup_epochs * epoch + warmup_lr
            self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: warmup_fn(epoch) / self.learning_rate
            )
        
        # Cosine annealing with warm restarts
        self.cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=sched_config['cosine']['T_0'],
            T_mult=sched_config['cosine']['T_mult'],
            eta_min=sched_config['cosine']['eta_min']
        )
        
        # Plateau scheduler
        self.plateau_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=sched_config['plateau']['factor'],
            patience=sched_config['plateau']['patience'],
            min_lr=sched_config['plateau']['min_lr']
        )
        
        self.warmup_epochs = warmup_epochs
    
    def apply_smote_enn(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE-ENN for data balancing"""
        balance_config = self.train_config['data_balancing']
        
        if not balance_config['enabled']:
            return X, y
        
        print(f"\n  ⚖️  Applying {balance_config['method']}...")
        print(f"    Before: {len(y):,} samples, {y.sum():,} fraud ({y.mean()*100:.2f}%)")
        
        if balance_config['method'] == 'smote_enn':
            smote_enn = SMOTEENN(
                sampling_strategy=balance_config['smote_ratio'],
                random_state=42
            )
            X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        elif balance_config['method'] == 'smote':
            smote = SMOTE(
                sampling_strategy=balance_config['smote_ratio'],
                random_state=42
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y
        
        print(f"    After: {len(y_resampled):,} samples, {y_resampled.sum():,} fraud ({y_resampled.mean()*100:.2f}%)")
        
        return X_resampled, y_resampled
    
    def train_epoch(self, graph, train_indices, train_labels, train_cust_idx, 
                   train_prod_idx, train_store_idx):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle training data
        perm = torch.randperm(len(train_indices))
        train_indices = train_indices[perm]
        train_labels = train_labels[perm]
        train_cust_idx = train_cust_idx[perm]
        train_prod_idx = train_prod_idx[perm]
        train_store_idx = train_store_idx[perm]
        
        # Mini-batch training
        for i in range(0, len(train_indices), self.batch_size):
            batch_end = min(i + self.batch_size, len(train_indices))
            
            batch_cust = train_cust_idx[i:batch_end]
            batch_prod = train_prod_idx[i:batch_end]
            batch_store = train_store_idx[i:batch_end]
            batch_labels = train_labels[i:batch_end]
            
            # Forward pass (full graph)
            self.optimizer.zero_grad()
            x_dict = self.model(graph.x_dict, graph.edge_index_dict)
            
            # Predict on batch
            logits = self.model.predict_transaction(x_dict, batch_cust, batch_prod, batch_store)
            
            # Loss (weighted cross entropy)
            loss = F.cross_entropy(logits, batch_labels)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, graph, val_indices, val_labels, val_cust_idx, 
                val_prod_idx, val_store_idx):
        """Evaluate on validation set"""
        self.model.eval()
        
        # Forward pass
        x_dict = self.model(graph.x_dict, graph.edge_index_dict)
        
        # Predict on all validation
        logits = self.model.predict_transaction(x_dict, val_cust_idx, val_prod_idx, val_store_idx)
        
        # Loss
        loss = F.cross_entropy(logits, val_labels)
        
        # Predictions
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = val_labels.cpu().numpy()
        
        # Metrics
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
        
        return {
            'loss': loss.item(),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'probs': probs,
            'preds': preds
        }
    
    def train(self, graph, train_data: dict, val_data: dict, verbose: bool = True):
        """Full training loop"""
        print(f"\n{'='*60}")
        print(f"🏋️  TRAINING {self.model.__class__.__name__}")
        print(f"{'='*60}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Device: {self.device}")
        
        # Move graph to device
        graph = graph.to(self.device)
        
        # Prepare training data
        train_indices = torch.arange(len(train_data['labels'])).to(self.device)
        train_labels = torch.LongTensor(train_data['labels']).to(self.device)
        train_cust_idx = torch.LongTensor(train_data['customer_idx']).to(self.device)
        train_prod_idx = torch.LongTensor(train_data['product_idx']).to(self.device)
        train_store_idx = torch.LongTensor(train_data['store_idx']).to(self.device)
        
        # Prepare validation data
        val_indices = torch.arange(len(val_data['labels'])).to(self.device)
        val_labels = torch.LongTensor(val_data['labels']).to(self.device)
        val_cust_idx = torch.LongTensor(val_data['customer_idx']).to(self.device)
        val_prod_idx = torch.LongTensor(val_data['product_idx']).to(self.device)
        val_store_idx = torch.LongTensor(val_data['store_idx']).to(self.device)
        
        print(f"\n  Training samples: {len(train_labels):,} ({train_labels.sum().item():,} fraud)")
        print(f"  Validation samples: {len(val_labels):,} ({val_labels.sum().item():,} fraud)")
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(
                graph, train_indices, train_labels,
                train_cust_idx, train_prod_idx, train_store_idx
            )
            
            # Validate
            val_metrics = self.evaluate(
                graph, val_indices, val_labels,
                val_cust_idx, val_prod_idx, val_store_idx
            )
            
            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]['lr']
            
            if epoch < self.warmup_epochs and self.warmup_scheduler:
                self.warmup_scheduler.step()
            else:
                self.cosine_scheduler.step()
                self.plateau_scheduler.step(val_metrics['loss'])
            
            # History
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['learning_rate'].append(current_lr)
            
            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.best_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
            
            # Logging
            if verbose and (epoch + 1) % 5 == 0:
                epoch_time = time.time() - epoch_start
                print(f"\n  Epoch {epoch+1:3d}/{self.num_epochs} ({epoch_time:.1f}s)")
                print(f"    Train Loss: {train_loss:.4f}")
                print(f"    Val Loss: {val_metrics['loss']:.4f} | "
                      f"F1: {val_metrics['f1']:.4f} | "
                      f"AUC: {val_metrics['auc']:.4f}")
                print(f"    LR: {current_lr:.6f} | "
                      f"Patience: {self.patience_counter}/{self.patience}")
            
            # Early stopping check
            if self.patience_counter >= self.patience:
                print(f"\n  ⚠️  Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        
        # Load best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            print(f"\n  ✅ Loaded best model (val_loss={self.best_val_loss:.4f})")
        
        print(f"\n  ⏱️  Total training time: {total_time/60:.1f} minutes")
        
        return self.history