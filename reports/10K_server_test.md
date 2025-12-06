# MindEve GNN Fraud Detection - Proje Raporu

**Tarih:** 5 Aralık 2025  
**Durum:** 10K basic test tamamlandı. 5M Veri ile Training Devam Ediyor  

---

## 🎯 Proje Amacı

MindEve GNN Fraud Detection, retail (perakende) sektöründe Graph Neural Networks (GNN) kullanarak fraudulent (sahte) işlemleri tespit eden bir sistemdir. Proje, CARE-GNN metodolojisinin basitleştirilmiş versiyonunu 89 milyon transaction içeren gerçek retail veri setine adapte eder.


## 🧠 Kullanılan Algoritmalar ve Yöntemler

### 1. Graph Neural Network: GraphSAGE

**Ana Model:** HeteroGNN (Heterogeneous Graph Neural Network)

- **Architecture:** GraphSAGE (Graph Sample and Aggregate)
- **Hidden Channels:** 64
- **Layers:** 2
- **Dropout:** 0.3
- **Total Parameters:** 115,554

**Neden GraphSAGE?**
- Inductive learning (yeni node'lar eklenebilir)
- Büyük graph'larda scalable
- Neighborhood aggregation ile güçlü representation learning

**Desteklenen Alternatifler:** GAT (Graph Attention Networks), GCN (Graph Convolutional Networks)

---

### 2. Heterogeneous Graph Yapısı

**3 Node Type:**
- **Customer:** 1,155,458 müşteri (20M veri için)
- **Product:** 13,752 ürün
- **Store:** 307 mağaza

**6 Edge Type (Forward + Reverse):**

| Forward | Reverse |
|---------|---------|
| customer → buys → product | product → bought_by → customer |
| customer → visits → store | store → visited_by → customer |
| product → sold_at → store | store → sells → product |


**Neden Heterogeneous?**  
Farklı entity tipleri arası ilişkileri model edebilir, her node type'ın kendine özgü feature'ları olabilir.

---

### 3. Fraud Labeling Yöntemi

**Dual Labeling Strategy:**

1. **Transaction-Based Labeling (Ana):**
   - Anomaly detection kullanılarak transaction seviyesinde etiketleme
   - Discount patterns, amount thresholds, behavior anomalies

2. **Customer-Level Aggregation:**
   - Graph'ta customer node'ları için max aggregation
   - En az bir fraud transaction varsa customer risky


---

### 4. Unsupervised Fraud Detection (Clustering)

**Ensemble Voting System:** 4 algoritmanın majority voting'i

- **K-Means:** Centroid-based clustering
- **DBSCAN:** Density-based spatial clustering
- **Isolation Forest:** Anomaly detection
- **GMM:** Gaussian Mixture Model

**Önceki Başarı:** 85.6% ground truth overlap

---

### 5. Class Imbalance Handling

**Teknikler:**
- **SMOTE-ENN:** Synthetic oversampling + noise removal
- **Focal Loss:** Hard examples'a daha fazla ağırlık
- **Class Weights:** ~1:100 ratio
- **Stratified Splitting:** Fraud oranı korunarak split

---

### 6. Training Optimization

**Optimizer:** AdamW (weight decay: 0.01)

**Learning Rate Strategy (3-Stage):**
1. **Warmup:** 5 epoch (yavaş başlangıç)
2. **Cosine Annealing with Warm Restarts:** Cyclical learning
3. **Early Stopping:** Patience 15 epoch

**Training Config:**
- Epochs: 50
- Batch Size: 1024
- Initial LR: 0.001
- Min LR: 1e-6

---

### 7. Feature Engineering

**Customer Features (9):**
1. total_spending
2. transaction_count
3. avg_transaction_value
4. unique_products
5. unique_stores
6. return_rate
7. transaction_velocity
8. discount_rate (YENİ)
9. campaign_rate (YENİ)

**Product Features (4):**
1. popularity
2. avg_price
3. unique_customers
4. discount_frequency

**Store Features (4):**
- Aggregated: sales, customer, product statistics

**Toplam:** 55 engineered feature (13 raw kolondan)

---

### 8. Evaluation Metrics

**Primary:**
- **AUC-ROC:** Fraud/non-fraud discriminative power
- **Recall:** Fraud yakalama oranı (en önemli!)
- **Precision:** False positive oranı
- **F1-Score:** Precision-recall harmonic mean

**Additional:**
- Confusion Matrix (TP, TN, FP, FN)
- Top-K Precision (100, 500, 1000)
- Loss curves (train/val)

---

## 📊 10K Test Sonuçları (Proof of Concept)

### Test Konfigürasyonu

- **Sample:** 10,000 transactions
- **Training Time:** ~6 minutes
- **Device:** GPU (NVIDIA RTX 2080 Ti)
- **Epochs:** 50

### Veri İstatistikleri

**After filtering:** 8,688 transactions
- Fraud cases: 95 (0.95%)
- Customers: 2,207
- Products: 3,517
- Stores: 1 (single store - dummy features)
- Edges: 28,114

**Split (70/15/15):**
- Train: 6,081 (57 fraud)
- Val: 1,303 (10 fraud)
- Test: 1,304 (15 fraud)

### Training Performansı

- **Best Epoch:** 47
- **Train Loss:** 0.1814
- **Val Loss:** 0.2353
- **Training Time:** 6 seconds (50 epoch)

**Progression:**
- Epoch 5: AUC=0.8195
- Epoch 25: AUC=0.9432
- Epoch 50: AUC=0.9686

### Test Set Sonuçları

**Ana Metrikler:**

| Metric | Değer | Yorum |
|--------|-------|-------|
| AUC-ROC | 0.9942 | 🌟 Neredeyse perfect! |
| Recall | 1.0000 | 🌟 Hiç fraud kaçmadı! |
| Precision | 0.1415-0.1485 | ⚠️ Düşük (çok FP) |
| F1-Score | 0.2479-0.2586 | ⚠️ Precision'dan etkilenmiş |

**Confusion Matrix:**

```
                Predicted
                Non-Fraud  Fraud
Actual  
Non-Fraud       ~1,200     ~90      (TN / FP)
Fraud              0        15      (FN / TP)
```

- ✅ True Positives: 15 (tüm fraud'lar yakalandı)
- ✅ False Negatives: 0 (hiç fraud kaçmadı!)
- ⚠️ False Positives: ~90 (precision düşük)

**Top-K Precision:**
- Top-100: 15.00%
- Top-500: 3.00%
- Top-1000: 1.50%

### 10K Test Yorumu

**✅ Başarılar:**
- **Perfect Recall (100%):** Model hiçbir fraud'u kaçırmadı
- **Excellent AUC (0.9942):** Fraud detection capability çok yüksek
- **Stable Training:** Loss curves düzgün, overfit yok
- **Hızlı:** 6 dakikada eğitim

**⚠️ İyileştirme Alanları:**
- **Düşük Precision (14%):** Çok fazla false positive
- **Small Test Set:** Sadece 15 fraud (statistical power düşük)
- **Single Store:** Store-level patterns öğrenemedi

**🎯 Çıkarımlar:**
- Model fraud pattern'lerini başarıyla öğrenmiş
- Conservativ bir approach (güvenlik odaklı)
- Daha büyük dataset ile precision artabilir

---

## 🔄 5M Test Durumu (Devam Ediyor)

### Başlangıç

- **Start:** 05 Aralık 2025, 20:54
- **Current:** 05 Aralık 2025, 22:19
- **Elapsed:** 1 saat 25 dakika
- **Process ID:** 985033

### Veri Boyutu (10K → 5M Karşılaştırma)
"Karşılaştırma sonuçları, trainin bittiğinde eklenecek"


### Şu Anki Durum (🔄 Training)

**GPU Status:**
```
GPU 0: NVIDIA RTX 2080 Ti
├─ Utilization: 100% (tam güç!)
├─ Memory: 9.5 GB / 11 GB (85%)
├─ Temperature: 78°C (normal)
├─ Power: 170W / 250W
└─ Status: Training in progress
```

**Process:**
- CPU: 100%
- RAM: 13 GB
- Elapsed training: ~40 minutes
- Estimated remaining: 8-9 hours

**Timeline:**
- ✅ Config Loading (5 sec)
- ✅ Data Loading (2 min)
- ✅ Feature Engineering (15 min)
- ✅ Graph Building (40 min)
- 🔄 Training (50 epochs) (10-12 hours) ← **CURRENT**
- ⏳ Evaluation (10-15 min)
- ⏳ Save Results (1 min)

**Estimated completion:** 06 Aralık 2025, 06:00-08:00


