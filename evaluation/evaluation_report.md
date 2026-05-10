# Değerlendirme Raporu

Türkçe Semptom Değerlendirici sisteminin altın standart veri seti üzerindeki performans değerlendirmesi.

## Test Seti Özeti

- **Toplam örnek:** 30
- **Düşük risk:** 7 örnek
- **Orta risk:** 7 örnek
- **Acil risk:** 16 örnek

## Performans Karşılaştırması

| Metrik | Sadece Kural |
|---|---|
| Genel Doğruluk | 80.0% |
| Acil Kaçırma | 2/16 (12.5%) |

## Sonuç ve Tartışma

Sadece kural tabanlı sistem, herhangi bir LLM çağrısı olmaksızın deterministik kırmızı bayrak taraması ve risk grubu yükseltmesi ile çalışmaktadır.

> Not: Bu değerlendirme örnek bir altın standart üzerinde yapılmıştır. Saha testlerinde hekim onaylı geniş bir veri seti ile yeniden ölçülmesi planlanmaktadır.
