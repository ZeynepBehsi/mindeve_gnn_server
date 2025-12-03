# MindEve GNN Fraud Detection

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Graph Neural Networks (GNN) kullanarak retail transaction verisi üzerinde fraud detection sistemi. Heterogeneous graph yapısı ile customer-product-store ilişkilerini modelleyerek fraud pattern'lerini tespit eder.

## 🎯 Proje Özeti

**Hedef:** 9.2M retail transaction (sample data) içerisinden fraud pattern'lerini tespit etmek  
**Yöntem:** Heterogeneous Graph Neural Networks + Unsupervised Clustering  
**Sonuç:** F1=0.9723, AUC=0.9958 (10K test sample)

### Temel Özellikler

- ✅ **Heterogeneous Graph Construction:** Customer-Product-Store ilişkileri
- ✅ **Multiple GNN Architectures:** GraphSAGE, GAT, GCN
- ✅ **Unsupervised Labeling:** K-Means, DBSCAN, Isolation Forest, GMM ensemble
- ✅ **Advanced Training:** SMOTE-ENN balancing, 3-stage LR scheduling
- ✅ **MLflow Integration:** Experiment tracking ve model management
- ✅ **Production Ready:** Configurable, modular, scalable

---

## 📊 Dataset

**Kaynak:** MindEve Retail Transaction Data (Mayıs 2024)

| Özellik | Değer |
|---------|-------|
| **Total Transactions** | 9.2M |
| **Customers** | 500K |
| **Products** | 139K |
| **Stores** | 500+ |
| **Date Range** | 01 Mayıs - 30 Mayıs 2024 |
| **Features** | 28 (engineered) |
| **Graph Edges** | 1.4M+ |

### Feature Engineering

**Customer Features (6):**
- `transaction_count`, `total_spent`, `avg_transaction_value`
- `transaction_velocity`, `unique_stores`, `return_rate`

**Product Features (4):**
- `unit_price`, `amount`, `total_price`, `product_popularity`

**Store Features (3):**
- `transaction_count`, `total_revenue`, `unique_customers`

**Anomaly Indicators:**
- `price_deviation`, `is_unusual_amount`, `is_night_transaction`
- `is_bulk_purchase`, `is_weekend`, `hour_of_day`

---

## 🏗️ Proje Yapısı

```
mindeve_gnn_server/
├── config/                     # YAML konfigürasyon dosyaları
│   ├── base_config.yaml       # Ana proje ayarları
│   ├── clustering_config.yaml # Clustering algoritma parametreleri
│   └── gnn_config.yaml        # GNN model hiperparametreleri
│
├── data/
│   ├── raw/                   # Orijinal CSV dosyaları (git'e eklenmez)
│   ├── processed/             # İşlenmiş data, graphs (git'e eklenmez)
│   ├── splits/                # Train/val/test splits (git'e eklenmez)
│   └── sample/                # Test için 10K sample (git'e eklenir)
│       └── 2024_05_sample.csv
│
├── src/
│   ├── data/                  # Data loading & preprocessing
│   │   ├── loader.py          # CSV yükleme
│   │   └── preprocessor.py   # FeatureEngineer class
│   │
│   ├── labeling/              # Unsupervised fraud labeling
│   │   └── clustering.py     # K-Means, DBSCAN, IsolationForest, GMM
│   │
│   ├── models/                # GNN model implementasyonları
│   │   ├── gnn_models.py     # GraphSAGE, GAT, GCN
│   │   └── graph_builder.py  # Heterogeneous graph construction
│   │
│   ├── training/              # Training & evaluation
│   │   ├── trainer.py        # GNNTrainer (SMOTE-ENN, LR scheduling)
│   │   └── evaluator.py      # GNNEvaluator (metrics, top-k)
│   │
│   └── utils/                 # Yardımcı fonksiyonlar
│       ├── config_loader.py  # YAML config loader
│       └── logger.py         # Logging setup
│
├── scripts/                   # Executable scripts
│   ├── test_gnn.py           # ✅ Test script (10K sample)
│   ├── run_full_pipeline.py  # Full pipeline runner
│   ├── predict.py            # Inference script
│   └── start_mlflow.sh       # MLflow UI launcher
│
├── outputs/                   # Çıktılar (git'e eklenmez)
│   ├── mlruns/               # MLflow experiment tracking
│   ├── models/               # Saved models
│   └── figures/              # Visualizations
│
├── tests/                     # Unit tests
│
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Bu dosya
```

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric
- CUDA 12.1+ (GPU kullanımı için)

### Adım 1: Repository'yi Clone

```bash
git clone https://github.com/ZeynepBehsi/mindeve_gnn_server.git
cd mindeve_gnn_server
```

### Adım 2: Conda Environment Oluştur

```bash
conda create -n mindeve python=3.10 -y
conda activate mindeve
```

### Adım 3: Dependencies Kurulumu

#### CPU (Local Development - Mac)

```bash
# PyTorch
pip install torch torchvision

# PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Diğer dependencies
pip install -r requirements.txt
```

#### GPU (Server - CUDA 12.1)

```bash
# PyTorch with CUDA
pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric with CUDA
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu121.html

# Diğer dependencies
pip install -r requirements.txt
```

### Adım 4: Konfigürasyon

Config dosyalarını kontrol et ve gerekirse düzenle:

```bash
# Ana config
vim config/base_config.yaml

# Clustering parametreleri
vim config/clustering_config.yaml

# GNN hiperparametreleri
vim config/gnn_config.yaml
```

---

## 🧪 Hızlı Test (10K Sample)

Sample data ile tüm pipeline'ı test et:

```bash
# Cache temizle
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Test çalıştır
python scripts/test_gnn.py
```

**Beklenen Çıktı:**
```
================================================================================
🚀 PHASE 4: GNN MODELS (TEST MODE)
================================================================================

✅ Data loaded (10,000 transactions)
✅ Features engineered (28 features)
✅ Clustering completed (47.38% fraud rate)
✅ Graph built (999 customers, 56,876 edges)
✅ Model trained (10 epochs, 0.1 min)
✅ Evaluation complete

📊 Final Results (GraphSAGE):
  Test F1:        0.9723
  Test Precision: 0.9723
  Test Recall:    0.9723
  Test AUC:       0.9958

🎯 Top-K Precision:
  TOP-100:  100%
  TOP-500:  100%
  TOP-1000: 68.4%
================================================================================
```

---

## 📈 MLflow UI

Experiment tracking ve model karşılaştırması için MLflow UI'ı başlat:

```bash
# MLflow UI'ı başlat
./scripts/start_mlflow.sh

# Manuel başlatma
mlflow ui --backend-store-uri ./outputs/mlruns --port 5001
```

Tarayıcıda aç: **http://localhost:5001**

### MLflow'da Takip Edilen Metrikler

- **Training:** `train_loss`, `epoch_time`
- **Validation:** `val_loss`, `val_f1`, `val_auc`
- **Test:** `test_precision`, `test_recall`, `test_f1`, `test_auc`
- **Top-K:** `top_100_precision`, `top_500_precision`, `top_1000_precision`
- **Hyperparameters:** learning rate, batch size, dropout, vb.

---

## 🎯 Pipeline Aşamaları

### Phase 1: Data Loading
```bash
python -c "from src.data.loader import load_data; from src.utils.config_loader import ConfigLoader; config = ConfigLoader().load_all(); df = load_data(config)"
```

### Phase 2: Feature Engineering
```bash
python -c "from src.data.preprocessor import FeatureEngineer; from src.utils.config_loader import ConfigLoader; config = ConfigLoader().load_all(); fe = FeatureEngineer(config)"
```

### Phase 3: Clustering & Labeling
```python
from src.labeling.clustering import ClusteringExperiment

clustering_exp = ClusteringExperiment(config, test_mode=True)
results = clustering_exp.run_all(df)
fraud_labels, fraud_scores = clustering_exp.create_ensemble()
```

**Kullanılan Algoritmalar:**
- K-Means (k=2)
- DBSCAN (eps=0.5, min_samples=20)
- Isolation Forest (contamination=0.02)
- Gaussian Mixture Model (n_components=2)
- **Ensemble Voting** (threshold=0.4)

### Phase 4: Graph Construction
```python
from src.models.graph_builder import GraphBuilder

graph_builder = GraphBuilder(config)
graph, transaction_mapping = graph_builder.build_graph(df, fraud_labels)
```

**Graph Yapısı:**
- **Nodes:** Customers, Products, Stores
- **Edges:**
  - Customer ↔ Product (buys/bought_by)
  - Customer ↔ Store (visits/visited_by)
  - Product ↔ Store (sold_at/sells)

### Phase 5: Model Training
```python
from src.models.gnn_models import GraphSAGE
from src.training.trainer import GNNTrainer

model = GraphSAGE(config).to(device)
trainer = GNNTrainer(model, config)
history = trainer.train(graph, transaction_mapping, train_idx, val_idx)
```

**Training Features:**
- ✅ SMOTE-ENN for class balancing
- ✅ 3-stage LR scheduling (warmup + cosine annealing)
- ✅ Early stopping (patience=15)
- ✅ MLflow integration
- ✅ Best model checkpointing

### Phase 6: Evaluation
```python
from src.training.evaluator import GNNEvaluator

evaluator = GNNEvaluator(model, device)
results = evaluator.evaluate_full(graph, test_data)
```

**Evaluation Metrics:**
- Classification: Precision, Recall, F1, AUC
- Confusion Matrix
- Top-K Precision (K=100, 500, 1000)

---

## 🧠 Model Architectures

### GraphSAGE (Default)
```yaml
architectures:
  sage:
    hidden_channels: 128
    num_layers: 2
    dropout: 0.3
    aggregation: mean
```

### GAT (Graph Attention Network)
```yaml
architectures:
  gat:
    hidden_channels: 128
    num_layers: 2
    num_heads: 8
    dropout: 0.3
```

### GCN (Graph Convolutional Network)
```yaml
architectures:
  gcn:
    hidden_channels: 128
    num_layers: 2
    dropout: 0.3
    add_self_loops: true
```

---

## ⚙️ Konfigürasyon

### Base Config (`config/base_config.yaml`)

```yaml
project:
  name: "mindeve_gnn_fraud"
  version: "1.0.0"
  random_seed: 42

data:
  raw_data_path: "data/raw/2024_05_transactions.csv"
  processed_data_path: "data/processed"
  sample_data_path: "data/sample/2024_05_sample.csv"
  splits:
    train: 0.7
    val: 0.15
    test: 0.15

compute:
  device: "auto"  # auto, cpu, cuda
  num_workers: 4
  pin_memory: true

logging:
  log_dir: "outputs/logs"
  log_level: "INFO"
  console: true
  file: true

mlflow:
  tracking_uri: "./outputs/mlruns"
  experiment_name: "gnn_fraud_detection"

test_mode:
  enabled: false
  sample_size: 10000
```

### Training Config

```yaml
training:
  num_epochs: 100
  batch_size: 1024
  learning_rate: 0.001
  weight_decay: 0.0001
  
  # SMOTE-ENN
  use_smote: true
  smote_k_neighbors: 5
  
  # Learning Rate Scheduling
  lr_scheduler:
    type: "3stage"  # warmup + cosine + warm_restarts
    warmup_epochs: 5
    cosine_epochs: 45
    restart_epochs: 50
    restart_mult: 2
  
  # Early Stopping
  early_stopping:
    patience: 15
    min_delta: 0.001
    monitor: "val_loss"
```

---

## 📊 Sonuçlar

### 10K Test Sample Sonuçları

| Model | F1 Score | Precision | Recall | AUC |
|-------|----------|-----------|--------|-----|
| **GraphSAGE** | **0.9723** | **0.9723** | **0.9723** | **0.9958** |
| Logistic Regression | 0.4100 | - | - | 0.9700 |
| Random Forest | 0.6500 | - | - | 0.9800 |
| XGBoost | 0.5900 | - | - | 0.9800 |

### Baseline Karşılaştırması

**GraphSAGE İyileştirmeleri:**
- vs Logistic Regression: **+137% F1 improvement**
- vs Random Forest: **+49% F1 improvement**
- vs XGBoost: **+65% F1 improvement**

### Top-K Precision

| K | Precision | Recall |
|---|-----------|--------|
| 100 | 100.0% | - |
| 500 | 100.0% | - |
| 1000 | 68.4% | - |

**İnsan Analisti İçin:** İlk 100 en riskli transaction'ı kontrol ettiğinde %100 doğruluk garantisi!

---

## 🔬 Advanced Features

### 1. SMOTE-ENN Balancing

Class imbalance problemini çözmek için SMOTE (Synthetic Minority Over-sampling) + ENN (Edited Nearest Neighbors) kombinasyonu:

```python
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(
    smote=SMOTE(k_neighbors=5),
    enn=EditedNearestNeighbours()
)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
```

### 2. 3-Stage Learning Rate Scheduling

**Stage 1: Warmup (Epoch 1-5)**
```python
lr = base_lr * (epoch / warmup_epochs)
```

**Stage 2: Cosine Annealing (Epoch 6-50)**
```python
lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * T_cur / T_max))
```

**Stage 3: Warm Restarts (Epoch 51-100)**
```python
# Her 10 epoch'ta bir restart
lr_scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
```

### 3. Heterogeneous Graph Convolution

Her edge type için ayrı convolution:

```python
conv_dict = {
    ('customer', 'buys', 'product'): SAGEConv(...),
    ('product', 'bought_by', 'customer'): SAGEConv(...),
    ('customer', 'visits', 'store'): SAGEConv(...),
    ('store', 'visited_by', 'customer'): SAGEConv(...),
    ('product', 'sold_at', 'store'): SAGEConv(...),
    ('store', 'sells', 'product'): SAGEConv(...)
}
hetero_conv = HeteroConv(conv_dict, aggr='mean')
```

---

## 🐛 Troubleshooting

### Import Errors

```bash
# PyTorch Geometric kurulumunu kontrol et
python -c "import torch_geometric; print(torch_geometric.__version__)"

# Yeniden kur
pip uninstall torch-scatter torch-sparse torch-cluster -y
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### CUDA Errors

```bash
# CUDA versiyonunu kontrol et
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"

# Doğru CUDA versiyonu için PyTorch kur
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Memory Errors

```yaml
# config/base_config.yaml'de batch size'ı azalt
training:
  batch_size: 512  # 1024'ten 512'ye düşür
```

### MLflow UI Çalışmıyor

```bash
# Port zaten kullanımda olabilir
lsof -i :5001
kill -9 <PID>

# Farklı port kullan
mlflow ui --backend-store-uri ./outputs/mlruns --port 5002
```

---

## 📝 Git Workflow

### İlk Commit

```bash
git add .
git commit -m "Initial commit: GNN fraud detection project structure"
git push origin main
```

### Feature Branch

```bash
# Yeni feature için branch
git checkout -b feature/new-gnn-architecture

# Değişiklikleri commit et
git add .
git commit -m "feat: add GIN (Graph Isomorphism Network) architecture"

# Push ve PR oluştur
git push origin feature/new-gnn-architecture
```

### Bug Fix

```bash
git checkout -b fix/evaluator-metrics-bug
git add .
git commit -m "fix: correct top-k precision calculation in evaluator"
git push origin fix/evaluator-metrics-bug
```

---

## 🚀 Production Deployment

### Server Setup (3x RTX A4000 GPU)

```bash
# 1. Server'a SSH
ssh user@server_ip

# 2. Repo clone
git clone https://github.com/ZeynepBehsi/mindeve_gnn_server.git
cd mindeve_gnn_server

# 3. Environment setup
conda create -n mindeve python=3.10 -y
conda activate mindeve

# 4. CUDA dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu121.html
pip install -r requirements.txt

# 5. Data transfer (buraya özel data transfer metodu ekle)
```

### Full Training

```bash
# Tmux session başlat
tmux new-session -s mindeve_training

# Full pipeline çalıştır
python scripts/run_full_pipeline.py --config config/base_config.yaml

# Detach: Ctrl+B, D
# Reattach: tmux attach -t mindeve_training
```

### Model Serving (Inference)

```bash
# Saved model ile inference
python scripts/predict.py \
  --model_path outputs/models/graphsage_best.pt \
  --data_path data/new_transactions.csv \
  --output_path predictions.csv
```

---

## 📚 Referanslar

**CARE-GNN Paper:**
- Dou, Y., et al. (2020). "Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters." CIKM 2020.
- [Paper Link](https://arxiv.org/abs/2008.08692)

**PyTorch Geometric:**
- [Documentation](https://pytorch-geometric.readthedocs.io/)
- [GitHub](https://github.com/pyg-team/pytorch_geometric)

**MLflow:**
- [Documentation](https://mlflow.org/docs/latest/index.html)
- [Tutorial](https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html)

---

## 🤝 Contributing

Pull request'ler memnuniyetle karşılanır! Büyük değişiklikler için lütfen önce bir issue açarak ne değiştirmek istediğinizi tartışın.

### Development Workflow

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 👥 İletişim

**Proje Sahibi:** Zeynep Behsi  
**GitHub:** [@ZeynepBehsi](https://github.com/ZeynepBehsi)  
**Repository:** [mindeve_gnn_server](https://github.com/ZeynepBehsi/mindeve_gnn_server)

---

## 🙏 Acknowledgments

- MindEve ekibine data ve computational resources için teşekkürler
- PyTorch Geometric topluluğuna excellent library için
- CARE-GNN paper yazarlarına methodology için

---

## 📅 Changelog

### v1.0.0 (2025-12-03) ✅

**İlk Stable Release:**
- ✅ Full pipeline implementation (data → clustering → graph → training → eval)
- ✅ 3 GNN architectures (GraphSAGE, GAT, GCN)
- ✅ 4 clustering algorithms + ensemble voting
- ✅ SMOTE-ENN balancing
- ✅ 3-stage LR scheduling
- ✅ MLflow integration
- ✅ Comprehensive evaluation metrics
- ✅ 10K test sample validated (F1=0.9723, AUC=0.9958)

**Test Results:**
```
Pipeline: ✅ BAŞARILI
Training Time: 0.1 min (10K sample, CPU)
Test Performance: F1=0.9723, AUC=0.9958
MLflow: ✅ Tracking active
```

---

<div align="center">

**⭐ Star this repository if you find it useful! ⭐**

Made with ❤️ by Zeynep Behsi

</div>