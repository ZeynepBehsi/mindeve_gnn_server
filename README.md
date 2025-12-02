# 🧠 MindEve GNN - Fraud Detection with Graph Neural Networks

## 📋 Proje Açıklaması
Graph Neural Networks (GNN) kullanarak retail transaction verisi üzerinde fraud detection.

**Hedef**: 89M transaction içerisinden fraud pattern'leri tespit etmek.

---

## 🏗️ Proje Yapısı
```
mindeve_gnn_server/
├── config/              # YAML configuration files
├── data/
│   ├── raw/            # Orijinal CSV (git'e eklenmez)
│   ├── processed/      # Graph, features (git'e eklenmez)
│   ├── splits/         # Train/val/test (git'e eklenmez)
│   └── sample/         # Test için 10K sample (git'e eklenir)
├── src/
│   ├── data/           # Data loading, preprocessing
│   ├── labeling/       # Clustering algorithms
│   ├── models/         # GNN architectures
│   ├── training/       # Training loops
│   ├── utils/          # Helpers, logging
│   └── experiments/    # Experiment runners
├── outputs/            # Models, figures, logs (git'e eklenmez)
├── scripts/            # Shell scripts (setup, run)
├── tests/              # Unit tests
└── requirements.txt    # Python dependencies
```

---

## 🚀 Kurulum

### Local (Mac) - Development
```bash
# 1. Conda environment
conda create -n mindeve python=3.10 -y
conda activate mindeve

# 2. PyTorch (CPU için Mac)
pip install torch torchvision

# 3. PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse

# 4. Diğer dependencies
pip install -r requirements.txt
```

### Server (GPU) - Production
```bash
# 1. Repo'yu clone
git clone https://github.com/USERNAME/mindeve_gnn_server.git
cd mindeve_gnn_server

# 2. Conda environment
conda create -n mindeve python=3.10 -y
conda activate mindeve

# 3. PyTorch + CUDA
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# 4. Dependencies
pip install -r requirements.txt

# 5. Data transfer (Rclone)
# (Ayrı dokümantasyon)
```

---

## 📊 Workflow

### 1️⃣ Local Testing (10K sample)
```bash
# Sample data oluştur
python scripts/create_sample.py --n_samples 10000

# Clustering test
python src/experiments/phase3_clustering.py --config config/clustering_config.yaml --test_mode

# Training test
python src/experiments/phase4_gnn_comparison.py --config config/gnn_config.yaml --test_mode
```

### 2️⃣ Server Production (89M full)
```bash
# Server'da
tmux new-session -s mindeve_training

# Full data training
python src/experiments/phase3_clustering.py --config config/clustering_config.yaml

# Detach: Ctrl+B, D
```

---

## 🔬 Experiments

### Phase 3: Clustering-based Labeling
- K-Means, DBSCAN, Isolation Forest, GMM
- Görselleştirme: PCA, t-SNE, UMAP, Silhouette

### Phase 4: GNN Architecture Comparison
- GraphSAGE, GAT, GCN
- SMOTE-ENN balancing
- LR scheduling (warmup + cosine)

### Phase 5: Fine-tuning
- Hyperparameter optimization
- Advanced scheduling
- Graph augmentation

---

## 📈 MLflow Tracking
```bash
# MLflow UI başlat
mlflow ui --backend-store-uri ./outputs/mlruns

# Browser: http://localhost:5000
```

---

## 👥 Contributors
- Zeynep (Research Lead)

## 📄 License
MIT
