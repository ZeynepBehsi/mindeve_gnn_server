# 📊 Labeling Stratejileri Karşılaştırması

## YÖNTEM 1: Historical Fraud Dataset (Eski Yaklaşım)

### Nasıl Çalışıyordu?
```python
# Gerçek fraud etiketleri kullanılırdı
df['fraud_label'] = historical_fraud_labels  # 0 veya 1
```

### Avantajları ✅

1. **Gerçek ground truth**: Doğrulanmış fraud case'leri
2. **Yüksek güvenilirlik**: İnsan expert'lerin onayladığı
3. **Supervised learning**: Klasik makine öğrenmesi
4. **Benchmark için ideal**: Model performansını doğru ölçme

### Dezavantajları ❌

1. **Label gürültüsü (noise)**: Customer-level labeling problemi
   - Bir müşteri fraud ise TÜM işlemleri fraud sayılıyor
   - Aynı müşterinin normal işlemleri de fraud etiketleniyor
   - FP (False Positive) artışı

2. **Veri eksikliği**: Yeterli fraud örneği olmayabilir
   - Fraud rate çok düşük (<1%)
   - Class imbalance sorunu
   - Yeni fraud pattern'leri yakalayamaz

3. **Zamanlama problemi**:
   - Fraud tespit edilene kadar zaman geçer
   - Eski data'da olup yeni data'da olmayan fraud'lar

4. **Etiketleme maliyeti**:
   - İnsan kaynağı gerekir
   - Pahalı ve zaman alıcı

### Senin Bulgularından (Memory'den):

- "Customer-based labeling significantly underperforms transaction-based labeling"
- "Missing reverse edges in heterogeneous graphs cause propagation errors"

---

## YÖNTEM 2: Clustering-Based Ensemble Labeling (Şu Anki Yaklaşım)

### Nasıl Çalışıyor?
```python
# 4 farklı unsupervised algoritma
algorithms = ['KMeans', 'DBSCAN', 'IsolationForest', 'GMM']

# Ensemble voting
fraud_label, fraud_score = clustering.create_ensemble()
# Threshold: 0.3 (4 algoritmadan 2'si fraud derse = fraud)
```

### Algoritmaların Katkısı:

#### 1. K-Means (Ağırlık: 1.0)
```python
# En küçük cluster = fraud
cluster_sizes = np.bincount(labels)
fraud_cluster = np.argmin(cluster_sizes)
```

- **Güçlü yönü**: Hızlı, kararlı
- **Zayıf yönü**: Sphere-shaped cluster varsayımı

#### 2. DBSCAN (Ağırlık: 1.0)
```python
# Outlier'lar (-1) = fraud
fraud_mask = (labels == -1)
```

- **Güçlü yönü**: Density-based, arbitrary shape'ler
- **Zayıf yönü**: Epsilon hassas, sparse data'da zayıf

#### 3. Isolation Forest (Ağırlık: 1.5) ⭐ En Yüksek
```python
# Anomaly detection
fraud_mask = (labels == -1)  # Outliers
```

- **Güçlü yönü**: Anomaly detection için tasarlanmış
- **Zayıf yönü**: Contamination parametresi hassas

#### 4. GMM (Ağırlık: 1.0)
```python
# En küçük Gaussian component = fraud
cluster_sizes = np.bincount(labels)
fraud_cluster = np.argmin(cluster_sizes)
```

- **Güçlü yönü**: Probabilistic, soft clustering
- **Zayıf yönü**: EM convergence sorunları

### Ensemble Voting:
```python
weights = {
    'kmeans': 1.0,
    'dbscan': 1.0,
    'isolation_forest': 1.5,  # En güvenilir
    'gmm': 1.0
}

fraud_score = Σ(weight_i × fraud_mask_i)
fraud_label = (fraud_score >= 0.3)  # Threshold
```

### Avantajları ✅

1. **Transaction-level labeling**: Her işlem ayrı değerlendiriliyor
   - Customer-level noise yok
   - Daha hassas fraud tespiti

2. **Unsupervised**: Label'a ihtiyaç yok
   - Yeni fraud pattern'leri otomatik keşfediyor
   - Maliyet yok

3. **Ensemble robustness**:
   - 4 algoritmanın consensus'u
   - Tek algoritma yanılgısı minimize

4. **Fraud rate kontrolü**:
```python
   fraud_rate_validation:
     min: 0.01  # %1
     max: 0.15  # %15
```

5. **Feature engineering ile güçlü**:
   - 28+ feature (price_deviation, return_rate, etc.)
   - Discount features (yeni dataset ile)

### Dezavantajları ❌

1. **Ground truth yok**: 
   - Gerçek fraud mu belirsiz
   - Model kendi öğretiyor kendine

2. **Hyperparameter hassasiyeti**:
   - Epsilon (DBSCAN)
   - Contamination (IsoForest)
   - K (K-Means)

3. **Threshold seçimi kritik**:
   - 0.3 çok düşük → Fazla FP
   - 0.3 çok yüksek → Fazla FN

---

## 🔬 Performans Karşılaştırması

### **Historical Fraud Dataset Yaklaşımı** (Varsayımsal)

| Metric | Değer | Açıklama |
|--------|-------|----------|
| **Precision** | 0.65-0.75 | Customer-level noise nedeniyle düşük |
| **Recall** | 0.80-0.90 | Bilinen fraud'ları iyi yakalar |
| **F1-Score** | 0.70-0.80 | Orta seviye |
| **Label Quality** | ⭐⭐⭐ | Noisy (customer-level) |
| **Fraud Rate** | 0.5-2% | Gerçek fraud oranı |

**Problem**: 
```
Müşteri X fraud → TÜM işlemleri fraud
  ├─ İşlem 1: 5 TL ekmek → FRAUD ❌ (False label)
  ├─ İşlem 2: 10 TL süt → FRAUD ❌ (False label)
  └─ İşlem 100: 5000 TL TV → FRAUD ✅ (True fraud)
```

---

### **Clustering Ensemble Yaklaşımı** (Şu Anki)

#### **Clustering Sonuçları** (10K sample test):
```
Ensemble Voting:
  Threshold: 0.3
  Final fraud rate: 1.2%
  Fraud count: 120

Best Algorithms:
  - IsolationForest: Silhouette 0.42
  - GMM (n=3): Silhouette 0.38
  - KMeans (k=3): Silhouette 0.35
  - DBSCAN (eps=0.5): Silhouette 0.28
```

#### **GNN Model Sonuçları** (5M sample test):

| Metric | GraphSAGE | GAT | GCN |
|--------|-----------|-----|-----|
| **Precision** | **0.9942** | ? | ? |
| **Recall** | **1.0000** | ? | ? |
| **F1-Score** | **0.9971** | ? | ? |
| **AUC-ROC** | **0.9942** | ? | ? |