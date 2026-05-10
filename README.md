# Türkçe Semptom Değerlendirici

LLM (Büyük Dil Modeli) ve RAG (Retrieval-Augmented Generation) mimarisi tabanlı, Türkçe dil destekli akıllı sağlık chatbotu prototipi.

> ⚠️ **Önemli:** Bu sistem bir tanı aracı **değildir**. Yalnızca bireyleri doğru sağlık kararına yönlendiren bir karar destek sistemidir. Acil durumlarda lütfen **112**'yi arayın.

## Proje Hakkında

Türkiye'de gece saatlerinde veya hafta sonlarında erişilebilir sağlık hizmeti bulamayan bireyler, semptomlarını doğru değerlendirememekten kaynaklı olarak gereksiz acil servis başvuruları yapmakta veya gerçekten acil olan durumları geç fark etmektedir. Bu proje, Türkçe doğal dilde diyalog kurabilen, WHO ve Sağlık Bakanlığı kaynaklarına dayalı bilgi sunan, risk seviyesi tahmini yapan bir karar destek chatbotu sunar.

### Hedef Kategoriler
- Solunum yolu hastalıkları
- Sindirim sistemi şikayetleri
- Genel ağrı sendromları

### Risk Sınıflandırması
- 🟢 **Düşük** — Evde takip edilebilir
- 🟡 **Orta** — En kısa sürede aile hekimine başvurulmalı
- 🔴 **Acil** — Derhal acil servise / 112'ye yönlendirme

## Teknoloji Yığını

| Katman | Teknoloji |
|--------|-----------|
| Dil | Python 3.11+ |
| LLM | OpenAI API / Anthropic Claude API |
| RAG Çerçevesi | LangChain |
| Vektör Veritabanı | ChromaDB |
| Embedding | Sentence-Transformers (Türkçe model) |
| Arayüz | Streamlit |
| Veri İşleme | Pandas, NumPy, PyPDF |

## Proje Yapısı

```
turkce-semptom-degerlendirici/
├── app/
│   ├── data_loader.py      # PDF/veri yükleme ve temizleme
│   ├── rag_engine.py       # ChromaDB + retrieval
│   ├── chatbot.py          # LangChain konuşma zinciri
│   └── risk_scorer.py      # Risk sınıflandırma motoru
├── data/
│   ├── raw/                # Ham veriler (WHO, Sağlık Bakanlığı PDF'leri)
│   └── processed/          # İşlenmiş veriler
├── tests/                  # Birim testleri
├── notebooks/              # Deney ve analiz defterleri
├── streamlit_app.py        # Streamlit giriş noktası
├── requirements.txt
├── .env.example
└── README.md
```

## Kurulum

### 1. Depoyu klonla
```bash
git clone https://github.com/KULLANICI_ADIN/turkce-semptom-degerlendirici.git
cd turkce-semptom-degerlendirici
```

### 2. Sanal ortam oluştur
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Bağımlılıkları yükle
```bash
pip install -r requirements.txt
```

### 4. API anahtarını ayarla
`.env.example` dosyasını `.env` olarak kopyala ve API anahtarını gir:
```bash
cp .env.example .env
```

### 5. Uygulamayı başlat
```bash
streamlit run streamlit_app.py
```

## Veri Gizliliği ve Etik

- Kullanıcı verileri **anonimleştirilir** ve oturum dışında saklanmaz.
- Sistem, KVKK kapsamında hassas sağlık verisi koruma ilkelerine uygun tasarlanmıştır.
- Saha testleri öncesinde **etik kurul onayı** alınacaktır.
- Sistem hiçbir koşulda tanı koymaz; sorumluluk reddi her oturumda gösterilir.

## Yol Haritası

- [x] Proje yapısının oluşturulması
- [ ] Veri hazırlık modülü (WHO + Sağlık Bakanlığı kaynakları)
- [ ] RAG altyapısı (ChromaDB + Türkçe embedding)
- [ ] LangChain konuşma motoru
- [ ] Risk skorlama hibrit motoru
- [ ] Streamlit arayüzü
- [ ] Saha testi (10+ gönüllü kullanıcı)
- [ ] Etik sınır analizi dokümantasyonu

## Akademik Bağlam

Bu proje [Ders Adı] kapsamında geliştirilmiştir.

## Lisans

MIT Lisansı — `LICENSE` dosyasına bakın.
