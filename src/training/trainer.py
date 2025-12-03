"""
GNN Model Trainer with MLflow tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Tuple
import time
from pathlib import Path

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow not available. Install with: pip install mlflow")

from imblearn.combine import SMOTEENN


class GNNTrainer:
    """
    GNN Trainer with:
    - SMOTE-ENN balancing
    - 3-stage LR scheduling (warmup + cosine + plateau)
    - Early stopping
    - Mini-batch training
    - MLflow experiment tracking
    """
    
    def __init__(self, model: nn.Module, config: dict, logger=None):
        self.model = model
        self.config = config
        self.logger = logger
        self.device = torch.device(
            config['compute']['device'] if config['compute']['device'] != 'auto' 
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Training config
        train_config = config['training']
        self.num_epochs = train_config.get('num_epochs', 100)
        self.batch_size = train_config.get('batch_size', 1024)
        self.learning_rate = train_config.get('learning_rate', 0.001)
        self.weight_decay = train_config.get('weight_decay', 1e-4)
        
        # Test mode
        if config.get('test_mode', {}).get('enabled', False):
            test_config = config['test_mode']
            if test_config.get('fast_dev_run', False):
                self.num_epochs = test_config.get('epochs', 10)
                self.batch_size = test_config.get('batch_size', 512)
                if self.logger:
                    self.logger.info(f"⚡ Test mode: epochs={self.num_epochs}, batch_size={self.batch_size}")
        
        # Data balancing
        self.use_smote = train_config.get('data_balancing', {}).get('enabled', False)
        self.smote_ratio = train_config.get('data_balancing', {}).get('smote_ratio', 0.88)
        
        # Early stopping
        es_config = train_config.get('early_stopping', {})
        self.early_stopping_patience = es_config.get('patience', 15)
        self.min_delta = es_config.get('min_delta', 1e-4)
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_auc': [],
            'learning_rate': [],
            'epoch_time': [],
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }
        
        # MLflow
        self.use_mlflow = MLFLOW_AVAILABLE and config.get('mlflow', {}).get('enabled', True)
        self.mlflow_run = None
        
    def _setup_mlflow(self, model_name: str):
        """Setup MLflow experiment tracking"""
        if not self.use_mlflow:
            return
        
        try:
            # Set experiment
            experiment_name = self.config.get('mlflow', {}).get('experiment_name', 'mindeve_gnn')
            mlflow.set_experiment(experiment_name)
            
            # Start run
            run_name = f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
            self.mlflow_run = mlflow.start_run(run_name=run_name)
            
            # Log config params
            mlflow.log_params({
                'model_architecture': model_name,
                'hidden_channels': self.model.hidden_channels if hasattr(self.model, 'hidden_channels') else 'N/A',
                'num_layers': self.model.num_layers if hasattr(self.model, 'num_layers') else 'N/A',
                'dropout': self.model.dropout if hasattr(self.model, 'dropout') else 'N/A',
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'weight_decay': self.weight_decay,
                'use_smote': self.use_smote,
                'smote_ratio': self.smote_ratio,
                'early_stopping_patience': self.early_stopping_patience,
                'device': str(self.device)
            })
            
            # Log dataset info
            mlflow.log_params({
                'dataset': 'mindeve_retail',
                'date_range': f"{self.config['data']['date_range']['start']} to {self.config['data']['date_range']['end']}"
            })
            
            if self.logger:
                self.logger.info(f"✅ MLflow tracking enabled: {run_name}")
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"MLflow setup failed: {e}")
            self.use_mlflow = False
    
    def _log_metrics_to_mlflow(self, metrics: dict, step: int):
        """Log metrics to MLflow"""
        if not self.use_mlflow or self.mlflow_run is None:
            return
        
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"MLflow logging failed: {e}")
    
    def _balance_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE-ENN for data balancing"""
        if not self.use_smote:
            return X, y
        
        if self.logger:
            self.logger.info(f"Applying SMOTE-ENN (ratio={self.smote_ratio})...")
        
        original_fraud_count = (y == 1).sum()
        original_fraud_rate = y.mean()
        
        try:
            smote_enn = SMOTEENN(
                sampling_strategy=self.smote_ratio,
                random_state=self.config['project']['random_seed']
            )
            X_resampled, y_resampled = smote_enn.fit_resample(X, y)
            
            new_fraud_count = (y_resampled == 1).sum()
            new_fraud_rate = y_resampled.mean()
            
            if self.logger:
                self.logger.info(f"  Original: {len(y):,} samples, {original_fraud_count:,} fraud ({original_fraud_rate*100:.2f}%)")
                self.logger.info(f"  Balanced: {len(y_resampled):,} samples, {new_fraud_count:,} fraud ({new_fraud_rate*100:.2f}%)")
            
            # Log to MLflow
            if self.use_mlflow:
                mlflow.log_metrics({
                    'original_samples': len(y),
                    'balanced_samples': len(y_resampled),
                    'original_fraud_rate': original_fraud_rate,
                    'balanced_fraud_rate': new_fraud_rate
                })
            
            return X_resampled, y_resampled
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"SMOTE-ENN failed: {e}. Using original data.")
            return X, y
    
    def _setup_optimizer_and_schedulers(self):
        """Setup optimizer and learning rate schedulers"""
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # LR Schedulers
        lr_config = self.config['training'].get('lr_scheduler', {})
        
        # 1. Warmup
        warmup_epochs = lr_config.get('warmup', {}).get('epochs', 5)
        warmup_start_lr = lr_config.get('warmup', {}).get('start_lr', 1e-4)
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=warmup_start_lr / self.learning_rate,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        # 2. Cosine Annealing with Warm Restarts
        cosine_config = lr_config.get('cosine', {})
        self.cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=cosine_config.get('T_0', 10),
            T_mult=cosine_config.get('T_mult', 2),
            eta_min=cosine_config.get('eta_min', 1e-5)
        )
        
        # 3. Plateau (backup)
        plateau_config = lr_config.get('plateau', {})
        self.plateau_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=plateau_config.get('factor', 0.5),
            patience=plateau_config.get('patience', 10),
            
        )
        
        self.warmup_epochs = warmup_epochs
    
    def _get_current_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def train(
        self, 
        graph, 
        transaction_mapping, 
        train_idx: np.ndarray, 
        val_idx: np.ndarray
    ) -> Dict:
        """
        Train GNN model
        
        Args:
            graph: HeteroData graph
            transaction_mapping: DataFrame with transaction-node mapping
            train_idx: Training indices
            val_idx: Validation indices
        
        Returns:
            history: Training history dict
        """
        
        # Setup MLflow
        model_name = self.model.__class__.__name__
        if self.use_mlflow:
            self._setup_mlflow(model_name)
        
        print(f"\n{'='*60}")
        print(f"🏋️  TRAINING {model_name}")
        print(f"{'='*60}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Device: {self.device}")
        if self.use_mlflow:
            print(f"  MLflow: ✅ Enabled")
        
        if self.logger:
            self.logger.info(f"Training {model_name}")
            self.logger.info(f"  Epochs: {self.num_epochs}")
            self.logger.info(f"  Device: {self.device}")
        
        # Move model and graph to device
        self.model = self.model.to(self.device)
        graph = graph.to(self.device)
        
        # Prepare training data
        train_cust_idx = torch.LongTensor(
            transaction_mapping.loc[train_idx, 'customer_idx'].values
        ).to(self.device)
        train_prod_idx = torch.LongTensor(
            transaction_mapping.loc[train_idx, 'product_idx'].values
        ).to(self.device)
        train_store_idx = torch.LongTensor(
            transaction_mapping.loc[train_idx, 'store_idx'].values
        ).to(self.device)
        train_labels = torch.LongTensor(
            transaction_mapping.loc[train_idx, 'fraud_label'].values
        ).to(self.device)
        
        # Validation data
        val_cust_idx = torch.LongTensor(
            transaction_mapping.loc[val_idx, 'customer_idx'].values
        ).to(self.device)
        val_prod_idx = torch.LongTensor(
            transaction_mapping.loc[val_idx, 'product_idx'].values
        ).to(self.device)
        val_store_idx = torch.LongTensor(
            transaction_mapping.loc[val_idx, 'store_idx'].values
        ).to(self.device)
        val_labels = torch.LongTensor(
            transaction_mapping.loc[val_idx, 'fraud_label'].values
        ).to(self.device)
        
        print(f"\n  Training samples: {len(train_idx):,} ({train_labels.sum().item():,} fraud)")
        print(f"  Validation samples: {len(val_idx):,} ({val_labels.sum().item():,} fraud)")
        
        # Setup optimizer and schedulers
        self._setup_optimizer_and_schedulers()
        
        # Loss function (weighted for imbalance)
        fraud_rate = train_labels.float().mean().item()
        class_weights = torch.FloatTensor([1.0, (1-fraud_rate)/fraud_rate]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # ================================================================
            # TRAINING
            # ================================================================
            self.model.train()
            train_loss = 0
            num_batches = 0
            
            # Mini-batch training
            batch_size = self.batch_size
            num_samples = len(train_idx)
            indices = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                # Get batch data
                batch_cust = train_cust_idx[batch_indices]
                batch_prod = train_prod_idx[batch_indices]
                batch_store = train_store_idx[batch_indices]
                batch_labels = train_labels[batch_indices]
                
                # Forward pass
                self.optimizer.zero_grad()
                x_dict = self.model(graph.x_dict, graph.edge_index_dict)
                logits = self.model.predict_transaction(x_dict, batch_cust, batch_prod, batch_store)
                
                # Loss
                loss = criterion(logits, batch_labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            train_loss /= num_batches
            
            # ================================================================
            # VALIDATION
            # ================================================================
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                x_dict = self.model(graph.x_dict, graph.edge_index_dict)
                
                # Validate in batches
                num_val_batches = 0
                for i in range(0, len(val_idx), batch_size):
                    batch_cust = val_cust_idx[i:i+batch_size]
                    batch_prod = val_prod_idx[i:i+batch_size]
                    batch_store = val_store_idx[i:i+batch_size]
                    batch_labels = val_labels[i:i+batch_size]
                    
                    logits = self.model.predict_transaction(x_dict, batch_cust, batch_prod, batch_store)
                    loss = criterion(logits, batch_labels)
                    val_loss += loss.item()
                    num_val_batches += 1
                
                val_loss /= num_val_batches
                
                # Calculate F1 and AUC (on full validation set)
                all_logits = self.model.predict_transaction(x_dict, val_cust_idx, val_prod_idx, val_store_idx)
                predictions = all_logits.argmax(dim=1).cpu().numpy()
                proba = F.softmax(all_logits, dim=1)[:, 1].cpu().numpy()
                val_labels_np = val_labels.cpu().numpy()
                
                # Metrics
                from sklearn.metrics import f1_score, roc_auc_score
                val_f1 = f1_score(val_labels_np, predictions, zero_division=0)
                val_auc = roc_auc_score(val_labels_np, proba)
            
            # ================================================================
            # LR SCHEDULING
            # ================================================================
            current_lr = self._get_current_lr()
            
            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.cosine_scheduler.step()
            
            self.plateau_scheduler.step(val_loss)
            
            # ================================================================
            # HISTORY & LOGGING
            # ================================================================
            epoch_time = time.time() - epoch_start
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            self.history['val_auc'].append(val_auc)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)
            
            # Log to MLflow
            if self.use_mlflow:
                self._log_metrics_to_mlflow({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_f1': val_f1,
                    'val_auc': val_auc,
                    'learning_rate': current_lr,
                    'epoch_time': epoch_time
                }, step=epoch)
            
            # Print progress (every 5 epochs)
            if (epoch + 1) % 5 == 0:
                print(f"\n  Epoch {epoch+1:3d}/{self.num_epochs} ({epoch_time:.1f}s)")
                print(f"    Train Loss: {train_loss:.4f}")
                print(f"    Val Loss: {val_loss:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")
                print(f"    LR: {current_lr:.6f} | Patience: {patience_counter}/{self.early_stopping_patience}")
            
            # ================================================================
            # EARLY STOPPING
            # ================================================================
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                self.history['best_epoch'] = epoch + 1
                self.history['best_val_loss'] = best_val_loss
            else:
                patience_counter += 1
            
            if patience_counter >= self.early_stopping_patience:
                print(f"\n  ⚠️  Early stopping at epoch {epoch+1}")
                if self.logger:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # ================================================================
        # LOAD BEST MODEL
        # ================================================================
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n  ✅ Loaded best model (val_loss={best_val_loss:.4f})")
            if self.logger:
                self.logger.info(f"Loaded best model from epoch {self.history['best_epoch']}")
        
        # ================================================================
        # FINAL SUMMARY
        # ================================================================
        total_time = time.time() - start_time
        self.history['total_training_time'] = total_time
        
        print(f"\n  ⏱️  Total training time: {total_time/60:.1f} minutes")
        
        if self.logger:
            self.logger.info(f"Training complete: {total_time/60:.1f} minutes")
            self.logger.info(f"  Best epoch: {self.history['best_epoch']}")
            self.logger.info(f"  Best val loss: {best_val_loss:.4f}")
            self.logger.info(f"  Final val F1: {self.history['val_f1'][-1]:.4f}")
            self.logger.info(f"  Final val AUC: {self.history['val_auc'][-1]:.4f}")
        
        # Log final metrics to MLflow
        if self.use_mlflow:
            mlflow.log_metrics({
                'best_val_loss': best_val_loss,
                'best_epoch': self.history['best_epoch'],
                'total_training_time_minutes': total_time / 60,
                'final_val_f1': self.history['val_f1'][-1],
                'final_val_auc': self.history['val_auc'][-1]
            })
            
            # Log model to MLflow
            try:
                mlflow.pytorch.log_model(self.model, "model")
                if self.logger:
                    self.logger.info("✅ Model logged to MLflow")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to log model to MLflow: {e}")
            
            # End MLflow run
            mlflow.end_run()
        
        return self.history