## Test Sonrası Parametre Değişiklikleri

### Clustering Configuration (clustering_config.yaml)

**Değişiklik Yapıldı:**
```yaml
# ÖNCE (Tüm algoritmalar aktif)
algorithms:
  kmeans:
    enabled: true
    configs:
      - k: 2
      - k: 3
      - k: 4
  
  dbscan:
    enabled: true
    configs:
      - eps: 0.3, min_samples: 10
      - eps: 0.5, min_samples: 20
      - eps: 0.7, min_samples: 30
  
  isolation_forest:
    enabled: true
    configs:
      - contamination: 0.01
      - contamination: 0.03
      - contamination: 0.05
  
  gmm:
    enabled: true
    configs:
      - n_components: 2, covariance_type: 'full'
      - n_components: 3, covariance_type: 'full'
      - n_components: 2, covariance_type: 'diag'
```
```yaml
# SONRA (Sadece kazanan)
algorithms:
  kmeans:
    enabled: true
    configs:
      - k: 2  # Kazanan: Composite Score 0.7762, Fraud 9.17%
        init: "k-means++"
        max_iter: 300
        n_init: 10
  
  dbscan:
    enabled: false  # Kapatıldı
  
  isolation_forest:
    enabled: false  # Kapatıldı (cluster metrikleri hesaplanamadı)
  
  gmm:
    enabled: false  # Kapatıldı
```

**Gerekçe:**
- KMeans k=2 en yüksek composite score (0.7762)
- İdeal fraud rate (%9.17)
- Dengeli performans (tüm metriklerde iyi)

---

### GNN Configuration (gnn_config.yaml)

**Model Kapasitesi Artırıldı:**
```yaml
# ÖNCE
architectures:
  sage:
    hidden_channels: 64
    num_layers: 2
    dropout: 0.1

# SONRA
architectures:
  sage:
    hidden_channels: 128  # 2x artış
    num_layers: 3         # +1 layer
    dropout: 0.3          # Regularization artırıldı
```

**Gerekçe:** Model daha karmaşık fraud pattern'leri yakalayabilmeli.

---

**Learning Rate Düşürüldü:**
```yaml
# ÖNCE
training:
  learning_rate: 0.001

# SONRA
training:
  learning_rate: 0.0001  # 10x düşürüldü
```

**Gerekçe:** Daha yavaş ama stabil öğrenme, loss düşme şansı artar.

---

**LR Scheduler Basitleştirildi:**
```yaml
# ÖNCE
lr_scheduler:
  type: "cosine_with_warmup"
  warmup:
    epochs: 5
  cosine:
    T_max: 40
  warm_restarts:
    enabled: true
    T_0: 10

# SONRA
lr_scheduler:
  type: "reduce_on_plateau"
  plateau:
    mode: "min"
    factor: 0.5
    patience: 5
    min_lr: 0.00001
```

**Gerekçe:** Warmup + Cosine + Plateau kombinasyonu LR'de zıplamalar yaratıyordu (0.0001 → 0.018 → 0.000034). ReduceLROnPlateau daha stabil.

---

**Loss Function Değiştirildi:**
```yaml
# ÖNCE
loss:
  type: "focal"
  focal:
    alpha: 0.75
    gamma: 3.0

# SONRA
loss:
  type: "weighted_cross_entropy"
  class_weights:
    normal: 1.0
    fraud: 20.0  # 100'den düşürüldü
```

**Gerekçe:** Focal loss gamma=3.0 çok agresifti, zor örneklere aşırı odaklanıyordu. Weighted cross-entropy daha dengeli.

---

**SMOTE Sampling Dengeli Hale Getirildi:**
```yaml
# ÖNCE
data_balancing:
  smote:
    sampling_strategy: 0.70  # %70 fraud hedefi

# SONRA
data_balancing:
  smote:
    sampling_strategy: 0.50  # %50 fraud hedefi
```

**Gerekçe:** 0.70 çok agresif, model öğrenemiyordu. 0.50 daha dengeli sınıf dağılımı sağlar.

---

**Early Stopping Patience Artırıldı:**
```yaml
# ÖNCE
early_stopping:
  patience: 15

# SONRA
early_stopping:
  patience: 20
```

**Gerekçe:** Model 16-25 epoch'ta duruyordu, öğrenme şansı bulamıyordu. Daha uzun bekleme süresi.

---

### Değişiklik Özeti

| Parametre | Önceki | Sonraki | Sebep |
|-----------|--------|---------|-------|
| **Clustering Method** | 12 config | KMeans k=2 | En yüksek composite score |
| **Hidden Channels** | 64 | 128 | Model kapasitesi artışı |
| **Num Layers** | 2 | 3 | Daha derin model |
| **Dropout** | 0.1 | 0.3 | Overfitting önleme |
| **Learning Rate** | 0.001 | 0.0001 | Daha stabil öğrenme |
| **LR Scheduler** | Cosine+Warmup+Plateau | ReduceLROnPlateau | Basitleştirme |
| **Loss Function** | Focal (gamma=3.0) | Weighted CrossEntropy | Daha dengeli |
| **Class Weight (Fraud)** | 100 | 20 | Aşırı weighting kaldırıldı |
| **SMOTE Ratio** | 0.70 | 0.50 | Dengeli sınıf dağılımı |
| **Early Stopping Patience** | 15 | 20 | Daha uzun öğrenme süresi |

---

### Beklenen İyileşmeler

**Önceki Sorunlar:**
- Training loss: 0.6932 (hiç düşmedi)
- Validation F1: 0.00-0.29 (rastgele tahmin seviyesi)
- Validation AUC: 0.50 (öğrenme yok)
- Early stopping: 16-34 epoch (çok erken)

**Yeni Parametrelerle Hedefler:**
- Training loss düşmeli (0.6932 → <0.4)
- Validation F1 artmalı (>0.40)
- Validation AUC artmalı (>0.65)
- Early stopping geç olmalı (>35 epoch)
- Model öğrenmeli (gradient update'ler etkili olmalı)