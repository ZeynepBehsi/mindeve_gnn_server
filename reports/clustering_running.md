# Clustering Evaluation Sonuçları - 1M Veri Seti

**Tamamlanma Tarihi:** 15 Aralık 2025, 02:36  
**Toplam Süre:** 2339.15 dakika (~39 saat)  
**Veri Boyutu:** 1,000,000 işlem  
**Test Edilen Method:** 10 konfigürasyon

---

## Kazanan Method

**KMeans (k=2)**
```
Composite Score: 0.7762
Silhouette: 0.4410
Davies-Bouldin: 1.0163
Calinski-Harabasz: 159,122
Fraud Rate: 9.17%
Fraud Count: 91,680
```

---

## Tüm Sonuçlar (Sıralı)

### Top 5 Performans

| Sıra | Method | Composite Score | Silhouette | Fraud Rate | Clusters |
|------|--------|----------------|------------|------------|----------|
| 1 | **KMeans k=2** | **0.7762** | 0.4410 | 9.17% | 2 |
| 2 | DBSCAN eps=0.7 | 0.7706 | 0.2573 | 10.44% | 12 |
| 3 | KMeans k=3 | 0.7605 | 0.4620 | 6.46% | 3 |
| 4 | GMM n=3 full | 0.7575 | 0.4625 | 6.46% | 3 |
| 5 | GMM n=2 full | 0.7413 | 0.3961 | 15.51% | 2 |

### Diğer Sonuçlar

| Method | Composite Score | Silhouette | Fraud Rate | Not |
|--------|----------------|------------|------------|-----|
| GMM n=2 diag | 0.6630 | 0.7712 | 0.59% | Çok yüksek silhouette ama fraud çok düşük |
| KMeans k=4 | 0.6151 | 0.4953 | 0.0001% | Hiç fraud bulamadı |
| IsolationForest cont=0.05 | 0.3181 | 0.0000 | 5.00% | Cluster metrikleri hesaplanamadı |
| IsolationForest cont=0.03 | 0.2610 | 0.0000 | 3.00% | Cluster metrikleri hesaplanamadı |
| IsolationForest cont=0.01 | 0.2016 | 0.0000 | 0.92% | Cluster metrikleri hesaplanamadı |

---

## Detaylı Analiz

### 1. KMeans Performansı

**k=2 (Kazanan):**
- En yüksek composite score (0.7762)
- İyi silhouette (0.4410)
- İdeal fraud rate (%9.17)
- Dengeli cluster boyutları (91,680 vs 908,320)

**k=3:**
- Daha yüksek silhouette (0.4620)
- Düşük fraud rate (%6.46) - ideal aralığın altı
- Composite score 3. sırada (0.7605)

**k=4:**
- En yüksek silhouette (0.4953)
- Kritik başarısızlık: 0.0001% fraud - neredeyse hiç fraud bulamadı
- Composite score düşük (0.6151)

### 2. DBSCAN Performansı

**eps=0.7, min_samples=30:**
- 2. en yüksek composite score (0.7706)
- Düşük silhouette (0.2573) - cluster separation zayıf
- İdeal fraud rate (%10.44)
- 12 cluster + 104,449 outlier (noise points = fraud)
- En iyi Davies-Bouldin score (0.6597)

**Diğer DBSCAN konfigürasyonları:**
- eps=0.3: %39.73 fraud (aralık dışı - elendi)
- eps=0.5: %19.42 fraud (aralık dışı - elendi)

### 3. GMM Performansı

**n=3, covariance=full:**
- 4. sıra composite score (0.7575)
- En yüksek silhouette (0.4625) - KMeans k=3 ile neredeyse aynı
- Düşük fraud rate (%6.46) - ideal aralığın altı
- 3 component: 64,617 / 782,392 / 152,991

**n=2, covariance=full:**
- Yüksek fraud rate (%15.51) - ideal aralığın üstünde
- Orta silhouette (0.3961)
- 5. sıra (0.7413)

**n=2, covariance=diag:**
- En yüksek silhouette (0.7712) - mükemmel cluster separation
- Kritik başarısızlık: %0.59 fraud - çok düşük
- Composite score düşük (0.6630)
- Fraud rate cezası nedeniyle sıralamada geride kaldı

### 4. Isolation Forest Başarısızlığı

**Tüm konfigürasyonlar:**
- Silhouette: 0.0000 (hesaplanamadı)
- Davies-Bouldin: 999.0 (geçersiz)
- Calinski-Harabasz: 0.0000
- Tek cluster oluşturdu (anomaly vs normal)
- Composite score çok düşük (0.20-0.32)

**Sebep:** Isolation Forest cluster tabanlı değil, anomaly detection yapıyor. Internal cluster metrikleri hesaplanamıyor.

---

## Fraud Rate Analizi

### İdeal Aralıkta (%7-15)
```
KMeans k=2:      9.17%  ✓ (kazanan)
DBSCAN eps=0.7: 10.44%  ✓
```

### Düşük (%7'nin altı)
```
KMeans k=3:      6.46%  (sınırda)
GMM n=3 full:    6.46%  (sınırda)
IsolationForest: 0.92-5.00%
GMM n=2 diag:    0.59%
KMeans k=4:      0.0001%
```

### Yüksek (%15'in üstü)
```
GMM n=2 full:    15.51%  (sınırda)
```

---

## Metrik Karşılaştırması

### Silhouette Score (Cluster Separation)
```
En İyi:
1. GMM n=2 diag:  0.7712  (ama fraud çok düşük)
2. KMeans k=4:    0.4953  (ama fraud yok)
3. GMM n=3 full:  0.4625  ✓
4. KMeans k=3:    0.4620  ✓
5. KMeans k=2:    0.4410  ✓ (kazanan)
```

### Davies-Bouldin Index (Cluster Compactness)
```
En İyi (düşük):
1. DBSCAN eps=0.7: 0.6597  ✓
2. KMeans k=4:     0.6299  (ama fraud yok)
3. GMM n=2 diag:   0.8358
4. KMeans k=2:     1.0163  ✓ (kazanan)
5. KMeans k=3:     1.0842
```

### Calinski-Harabasz Score (Variance Ratio)
```
En İyi (yüksek):
1. KMeans k=3:     181,329
2. GMM n=3 full:   180,572
3. GMM n=2 full:   163,070
4. KMeans k=2:     159,122  ✓ (kazanan)
5. KMeans k=4:     144,126
```

---

## Composite Score Formülü Etkisi
```
composite_score = (
    silhouette_normalized × 0.35 +       (en ağırlıklı)
    davies_bouldin_normalized × 0.25 +
    calinski_harabasz_normalized × 0.20 +
    fraud_rate_score × 0.20
)
```

**KMeans k=2 neden kazandı?**
- Silhouette iyi (0.4410)
- Davies-Bouldin iyi (1.0163)
- Fraud rate ideal (%9.17) 
- Dengeli performans - hiçbir metrikte çok kötü değil

**GMM n=2 diag neden kaybetti?**
- Mükemmel silhouette (0.7712)
- Ama fraud rate çok düşük (%0.59) - büyük ceza
- Composite score'da 6. sıra

**KMeans k=4 neden başarısız?**
- İyi silhouette (0.4953)
- Ama hiç fraud bulamadı (%0.0001) - maksimum ceza
- Kullanılamaz

---

## Çıktılar

### CSV Raporları
```
clustering_ranking_detailed.csv: 2.2 KB (15+ metrik)
clustering_ranking_summary.csv:  861 bytes (7 anahtar metrik)
best_method.txt:                 475 bytes (kazanan özet)
```

### Görselleştirmeler
```
tsne_plots/:
  - kmeans_tsne.png (2.0 MB)
  - dbscan_tsne.png (2.0 MB)
  - gmm_tsne.png (1.9 MB)
  - isolationforest_tsne.png (2.0 MB)

comparison_plots.png: 4-panel karşılaştırma grafiği
```

---

## COMPARISON SONUÇLARINA GÖRE YAPILAN DEĞİŞKLİKLER

### 1. Clustering Config Güncelleme

**Sadece k-means k=2 bırakıldı. diğer algoritmalar devre dışı bırakıldı. :**

```yaml
kmeans:
  enabled: true
  configs:
    - k: 2  # 1M test kazananı
      init: "k-means++"
      max_iter: 300
      n_init: 10
```

### 2. Alternatif Seçenekler

**Daha yüksek precision için:**
```yaml
# KMeans k=3 veya GMM n=3 denenebilir
# Fraud rate düşük (%6.46) ama silhouette daha iyi (0.46)
# Daha az false positive, daha fazla true negative
```

**Daha yüksek recall için:**
```yaml
# DBSCAN eps=0.7 denenebilir
# Fraud rate biraz yüksek (%10.44)
# Daha fazla fraud yakalar ama false positive de artar
```

### 3. GNN Training Parametreleri güncellenecek

```yaml
learning_rate: 0.0001
hidden_channels: 128
num_layers: 3
dropout: 0.3
loss: weighted_cross_entropy (fraud: 20.0)
smote: sampling_strategy 0.50
```

---

## Sonraki Adımlar


###  GNN Training Başlatılacak
```bash
screen -S gnn_training
conda activate mindeve
CUDA_VISIBLE_DEVICES=0 python scripts/test_gnn.py 2>&1 | tee gnn_kmeans_k2_optimized.log
# Ctrl+A, D (detach)
```


## Kritik Bulgular

### Başarılı Konfigürasyonlar
1. **KMeans k=2** - En dengeli, composite score kazananı
2. **DBSCAN eps=0.7** - En iyi Davies-Bouldin, alternatif seçenek
3. **KMeans k=3 / GMM n=3** - En iyi silhouette, düşük fraud rate

### Başarısız Konfigürasyonlar
1. **KMeans k=4** - Hiç fraud bulamadı
2. **GMM n=2 diag** - Fraud rate çok düşük
3. **IsolationForest (tümü)** - Cluster metrikleri hesaplanamadı

### Süre Analizi
- Toplam: 39 saat
- t-SNE: ~35 saat (10 method × 3.5 saat/method)
- Clustering + Evaluation: ~4 saat

**t-SNE çok yavaş - gelecekte devre dışı bırakılabilir.**