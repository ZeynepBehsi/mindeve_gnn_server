# MindEve GNN Fraud Detection - Teknik Gelişim Raporu

**Proje:** Heterogeneous Graph Neural Network ile Perakende Fraud Tespiti  
**Veri Seti:** 89M transaction (2022-2025), 100K-5M sample testler  
**Tarih:** 12 Aralık 2025  
**Durum:** Parametre optimizasyonu ve clustering evaluation aşamasında

---

## 1. Başlangıç Durumu ve Temel Sorun

### Veri Yapısı
- 89 milyon transaction, 9.3 GB CSV
- 13 sütun: transaction ID, tarih, mağaza, müşteri, ürün, fiyat, indirim bilgileri
- Test ortamı: 100K-5M sample
- Tarih aralığı: 2022-2025

### İlk Model Konfigürasyonu
```yaml
Model: GraphSAGE (HeteroGNN)
Hidden Channels: 64
Layers: 2
Learning Rate: 0.001
Batch Size: 512
Loss: Focal Loss (gamma=2.0, alpha=0.75)
SMOTE Ratio: 0.30
```

### Tespit Edilen Kritik Sorun
Model hiç öğrenemiyordu:
- Training loss: 0.6932 (sabit, hiç düşmedi)
- Validation F1: 0.00-0.29 (rastgele tahmin seviyesi)
- Validation AUC: 0.50 (hiç öğrenme yok)
- Early stopping: 16-34 epoch (çok erken duruyordu)
- Tüm clustering methodları (Ensemble, K-Means, IsolationForest, GMM) başarısız

---

## 2. GPU Kullanım Sorunu

### Tespit
```bash
nvidia-smi
# GPU 0: 4 MiB / 11264 MiB
# GPU 1: 20 MiB / 11264 MiB
# Kullanılan process: YOK
```

Model CPU'da çalışıyordu, 100K sample 2-3 saat sürüyordu.

### Çözüm
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python script.py
```

GPU'da çalışma doğrulandı ancak clustering CPU-bound olduğu için büyük fark yaratmadı.

---

## 3. Clustering Performans Optimizasyonu

### İlk Durum
100K sample için clustering süreleri:
- K-Means: 2-3 dakika
- DBSCAN: 4-5 dakika
- GMM: 1-2 dakika
- IsolationForest: 30 saniye
- Toplam: 8-10 dakika

### Uygulanan Optimizasyonlar

#### K-Means
```python
# Önceki
MiniBatchKMeans(n_clusters=k, batch_size=10000, max_iter=100)
fit_predict(X_full)  # 100K üzerinde

# Sonrası
if len(X) > 100000:
    sample_size = min(50000, int(len(X) * 0.3))
    model.fit(X_sample)
    labels = model.predict(X_full)
```

#### DBSCAN
```python
# Öncesi
DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_full)

# Sonrası
labels_sample = DBSCAN().fit_predict(X_sample[50K])
nbrs = NearestNeighbors(n_neighbors=1).fit(X_sample)
distances, indices = nbrs.kneighbors(X_full)
labels_full = labels_sample[indices.flatten()]
```

#### GMM ve Isolation Forest
50K sample üzerinde eğitim, 1M üzerinde tahmin stratejisi uygulandı.

### Sonuç
Beklenen süre düşüşü: 8-10 dakika → 3-4 dakika

---

## 4. Learning Rate Scheduler Sorunu

### Tespit Edilen Problem
Üç farklı scheduler aynı anda çalışıyordu:
```python
Epoch 5:  LR = 0.008600   # Warmup sonu
Epoch 10: LR = 0.018100   # Cosine artışı (!)
Epoch 15: LR = 0.000658   # Ani düşüş
Epoch 20: LR = 0.000034   # Çok düşük
```

Bu, modelin öğrenememesinin ana sebeplerinden biriydi.

### Önerilen Çözüm
```yaml
training:
  learning_rate: 0.0001  # 0.001'den düşürüldü
  
  lr_scheduler:
    enabled: true
    type: "reduce_on_plateau"  # Basitleştirildi
    plateau:
      mode: "min"
      factor: 0.5
      patience: 5
      min_lr: 0.00001
```

Warmup + Cosine + Plateau karmaşası kaldırıldı, sadece ReduceLROnPlateau kullanılacak.

---

## 5. Loss Function ve Data Balancing

### Önceki Konfigürasyon
```yaml
loss:
  type: "focal"
  focal:
    alpha: 0.75
    gamma: 3.0  # Çok agresif
    
data_balancing:
  smote:
    sampling_strategy: 0.70  # Çok yüksek
```

### Yeni Konfigürasyon
```yaml
loss:
  type: "weighted_cross_entropy"
  class_weights:
    normal: 1.0
    fraud: 20.0  # 100'den düşürüldü
    
data_balancing:
  smote:
    sampling_strategy: 0.50  # 0.70'den düşürüldü
```

Focal loss çok agresif penalty uyguluyordu, weighted cross-entropy daha dengeli.

---

## 6. Model Kapasitesi Artışı

### Önceki Mimari
```yaml
architectures:
  sage:
    hidden_channels: 64
    num_layers: 2
    dropout: 0.1
```

### Yeni Mimari
```yaml
architectures:
  sage:
    hidden_channels: 128  # 2x artış
    num_layers: 3         # +1 layer
    dropout: 0.3          # Regularization artışı
```

Daha karmaşık fraud pattern'lerini yakalamak için model kapasitesi artırıldı.

---

## 7. Early Stopping Ayarları

### Önceki
```yaml
early_stopping:
  patience: 15
  min_delta: 0.001
```

Model 16-25 epoch civarında duruyordu, öğrenme şansı bulamıyordu.

### Yeni
```yaml
early_stopping:
  patience: 20  # Daha uzun bekleme
  min_delta: 0.0005  # Daha hassas delta
```

---

## 8. Clustering Evaluation Pipeline

### Motivasyon
12 clustering yöntemi × 20 dakika GNN training = 4 saat zaman kaybı. Önce clustering kalitesini değerlendirip en iyisini seçmek gerekiyordu.

### Geliştirilen Sistem
**evaluate_clustering_complete.py** scripti oluşturuldu:

#### Değerlendirilen Metodlar (12 konfigürasyon)
```
K-Means: k={2,3,4}
DBSCAN: eps={0.3,0.5,0.7}, min_samples={10,20,30}
Isolation Forest: contamination={0.01,0.03,0.05}
GMM: n_components={2,3}, covariance_type={full,diag}
```

#### Metrikler (Ground Truth Gerektirmeden)
```python
# Internal Quality
- Silhouette Score: [-1, 1], yüksek = iyi cluster separation
- Davies-Bouldin Index: [0, inf], düşük = compact clusters
- Calinski-Harabasz Score: [0, inf], yüksek = iyi variance ratio

# Fraud Labeling
- Fraud Rate: 7-15% ideal aralık
- Fraud Count: Mutlak sayı

# Composite Score
composite = (
    silhouette_norm * 0.35 +
    davies_bouldin_norm * 0.25 +
    calinski_harabasz_norm * 0.20 +
    fraud_rate_score * 0.20
)
```

#### Görselleştirmeler
- t-SNE projections (12 dosya): Cluster ve fraud pattern görselleştirme
- PCA projections (12 dosya): Lineer separation analizi
- Comparison plots (1 dosya): 4-panel karşılaştırma grafiği

#### Çıktılar
```
outputs/clustering_evaluation/
├── tsne_plots/ (12 PNG, 300 DPI)
├── pca_plots/ (12 PNG, 300 DPI)
└── reports/
    ├── clustering_ranking_detailed.csv
    ├── clustering_ranking_summary.csv
    ├── comparison_plots.png
    └── best_method.txt
```

### Teknik Sorunlar ve Çözümler

#### Output Buffering Sorunu
```bash
# Problem: Log dosyası boş kalıyordu
python script.py 2>&1 | tee log.txt  # 0 byte log

# Çözüm
PYTHONUNBUFFERED=1 python -u script.py 2>&1 | tee log.txt
```

#### Import Path Sorunu
```python
# Problem: ModuleNotFoundError: No module named 'src'
from src.utils.config_loader import load_all_configs

# Çözüm
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))
```

#### Config Directory Mismatch
```python
# Problem: Config files not found in 'config/' directory
# Çözüm: Current directory override
loader = ConfigLoader(config_dir='.')
```

---

## 9. Feature Engineering

### Oluşturulan Feature Kategorileri (55 feature)

**Fiyat Features (7)**
```
price_deviation, total_price, is_high_price, is_low_price,
is_unusual_amount, is_bulk_purchase, effective_price
```

**İndirim Features (7)**
```
discount_rate, discount_percentage, has_campaign, has_discount,
discount_per_unit, is_high_discount, campaign_no_discount
```

**Zaman Features (9)**
```
hour, day_of_week, day_of_month, month, year,
is_weekend, is_night_transaction, is_business_hours, is_holiday_season
```

**Müşteri Agregasyonları (9)**
```
customer_transaction_count, customer_total_spending, avg_transaction_value,
customer_unique_products, unique_stores, return_rate, transaction_velocity,
customer_discount_rate, customer_campaign_rate
```

**Ürün Agregasyonları (5)**
```
product_popularity, product_avg_price, product_price_deviation,
product_unique_customers, product_discount_frequency
```

**Anomaly Features (5)**
```
is_extreme_value, time_since_last_trans, is_rapid_transaction,
same_product_time_gap, is_repeated_product_purchase
```

### Clustering için Seçilen 15 Feature
```
price_deviation, return_rate, transaction_velocity, is_unusual_amount,
unique_stores, product_popularity, total_price, is_night_transaction,
is_bulk_purchase, avg_transaction_value, discount_rate, has_campaign,
is_high_discount, customer_discount_rate, product_discount_frequency
```

---

## 10. Graph Yapısı

### Node Types (Heterogeneous)
```
Customer nodes: 9 features (transaction behavior aggregates)
Product nodes: 4 features (popularity, price, customers, discount_freq)
Store nodes: 4 features (price stats, unique customers/products)
```

### Edge Types (Bidirectional)
```
customer ↔ product (buys / bought_by)
customer ↔ store (visits / visited_by)
product ↔ store (sold_at / sells)
```

### Boyutlar (100K sample)
```
Customers: ~16,500
Products: ~7,900
Stores: 7
Total edges: ~255,000
```

---

## 11. Güncel Durum ve Sonraki Adımlar

### Tamamlanan İşler
1. GPU kullanım kontrolü ve optimizasyonu
2. Clustering performans optimizasyonu (sampling stratejisi)
3. Learning rate scheduler problemi tespit edildi
4. Parametre optimizasyonları belirlendi (uygulanmadı)
5. Clustering evaluation pipeline geliştirildi
6. Output buffering ve import sorunları çözüldü

### Devam Eden İş
Clustering evaluation scripti 1M sample ile çalışıyor.

### Bekleyen İşler
1. Clustering evaluation sonuçlarını inceleme
2. En iyi clustering methodunu seçme
3. GNN config dosyasına parametre optimizasyonlarını uygulama
4. Seçilen clustering + optimize parametreler ile GNN eğitimi
5. Sonuçları değerlendirme ve paper için raporlama

### Beklenen İyileşmeler
```
Önceki: 12 combination × 20 dk = 4 saat
Yeni: 15 dk clustering evaluation + 20 dk GNN = 35 dakika
Zaman kazancı: ~3.5 saat
```

---

## Teknik Detaylar

### Donanım
```
GPU: 2× NVIDIA GeForce RTX 2080 Ti (11GB)
CUDA: 13.0
Driver: 580.95.05
Environment: conda (mindeve)
```

### Yazılım Stack
```
Python 3.10+
PyTorch 2.0+
PyTorch Geometric 2.4+
scikit-learn 1.3+
imbalanced-learn 0.11+
MLflow 2.8+
```

### Dosya Yapısı
```
mindeve_gnn_server-main/
├── config/
│   ├── base_config.yaml
│   ├── clustering_config.yaml
│   └── gnn_config.yaml
├── src/
│   ├── data/ (loader, preprocessor)
│   ├── labeling/ (clustering)
│   ├── models/ (gnn_models, graph_builder)
│   ├── training/ (trainer, evaluator)
│   └── utils/ (config_loader, seed)
├── scripts/
│   ├── evaluate_clustering_complete.py
│   ├── test_gnn.py
│   └── test_research_full.py
└── outputs/
    └── clustering_evaluation/
```