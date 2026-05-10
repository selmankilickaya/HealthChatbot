# Değerlendirme Metodolojisi

Bu klasör, **Türkçe Semptom Değerlendirici** sisteminin akademik değerlendirmesi için gerekli veri seti, betikler ve raporları içerir.

## Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `symptom_test_set.csv` | 30 örneklik altın standart veri seti (kategori bazında dengelenmiş) |
| `run_evaluation.py` | İki konfigürasyonu (kural tabanlı / hibrit) test eden ana betik |
| `results_rule_only.csv` | Sadece kural tabanlı yaklaşımın detaylı sonuçları (`run_evaluation.py` üretir) |
| `results_hybrid.csv` | Tam hibrit sistemin detaylı sonuçları (`run_evaluation.py` üretir) |
| `evaluation_report.md` | Otomatik üretilen Markdown formatında özet rapor |

## Çalıştırma

Proje kök dizininden:

```bash
# Hem kural tabanlı hem hibrit (Claude API gerektirir)
python evaluation/run_evaluation.py

# Sadece kural tabanlı (ücretsiz, hızlı)
python evaluation/run_evaluation.py --rule-only
```

## Test Seti Yapısı

Veri seti, üç hedef kategori (solunum, sindirim, ağrı) ve risk grubu vakalarından oluşur:

| Kategori | Örnek Sayısı |
|----------|---|
| Solunum yolu | 6 |
| Sindirim sistemi | 7 |
| Genel ağrı | 8 |
| Risk grubu (yaşlı, gebe, bebek, immün) | 4 |
| Karma / acil | 5 |
| **Toplam** | **30** |

Risk dağılımı sınıf dengesi gözetilerek hazırlanmıştır:
- Düşük: ~7 örnek
- Orta: ~9 örnek
- Acil: ~14 örnek (ağırlıklı, çünkü bu vakaları kaçırmak en yüksek riskli senaryodur)

## Hesaplanan Metrikler

### 1. Genel Doğruluk (Accuracy)
Doğru sınıflandırılan örneklerin oranı. Hocaya raporlanan ana metriktir.

### 2. Sınıf Bazlı Hassasiyet ve Duyarlılık (Precision / Recall)
- **Precision (P):** Sistemin bir sınıfa atadığı vakaların ne kadarı gerçekten o sınıfa ait.
- **Recall (R):** Bir sınıftaki gerçek vakaların ne kadarı doğru tespit edilmiş.

### 3. Karmaşıklık Matrisi (Confusion Matrix)
Hangi sınıfların birbirine karıştığını gösteren 3x3 matris.

### 4. Acil Vakaları Kaçırma Oranı (Critical False Negative Rate) ⭐
**Bu sistemin en kritik metriğidir.** Acil bir vakanın "Düşük" veya "Orta" olarak sınıflandırılması, kullanıcının yaşamı için tehlikelidir. Kabul edilebilir hedef: **%0**.

## Karşılaştırma Yaklaşımı

İki konfigürasyon karşılaştırılır:

**A) RULE_ONLY:** Sadece kural tabanlı kırmızı bayrak taraması + risk grubu yükseltmesi. LLM çağrısı yok.
- ✅ Avantaj: Anlık, ücretsiz, deterministik, çevrimdışı çalışabilir
- ❌ Dezavantaj: Belirsiz / dolaylı anlatımlarda zayıf

**B) HYBRID:** Kural tabanlı + LLM destekli + risk grubu. Tam sistem.
- ✅ Avantaj: Bağlamsal anlama, daha yüksek doğruluk
- ❌ Dezavantaj: API çağrısı, gecikme, maliyet

İki sonucun karşılaştırılması, **LLM bileşeninin sisteme katkısını** somut sayılarla göstermek için kullanılır.

## Kısıtlılıklar ve Sınırlamalar

1. **Sentetik veri seti:** Mevcut altın standart, hekim onaylı **gerçek** vakalar değildir. Saha testinde 10+ gerçek kullanıcıyla genişletilecektir.
2. **Tek değerlendirici:** Beklenen etiketler tek kişi tarafından atanmıştır; ideali çoklu hekim onayıdır (Cohen's kappa hesabı sonraki sürümde).
3. **Türkçe dil çeşitliliği:** Bölgesel ifadeler (örn. "*sancım dengeden çıktı*") test edilmemiştir.
4. **Çoklu tur diyalog:** Bu testler tek mesaj üzerinden çalışır; tam diyalog akışı saha testinde değerlendirilecektir.

## Etik Notlar

- Bu sistem **tanı koymaz**, yalnızca yönlendirme yapar.
- Saha testi öncesinde **etik kurul onayı** alınması zorunludur.
- KVKK kapsamında kullanıcı verileri anonimleştirilir.

---

**Hazırlayan:** Selman Kılıçkaya
**Lisans:** MIT
