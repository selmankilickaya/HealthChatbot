"""
build_index.py
Tek seferlik indeksleme: data/raw/ → ChromaDB

Bu betiği veri eklediğinizde veya değiştirdiğinizde çalıştırın.

Çalıştırma (proje kökünden):
    python scripts/build_index.py
"""
from pathlib import Path
import sys

# Proje kökünü Python yoluna ekle
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.data_loader import load_and_chunk
from app.rag_engine import RAGEngine


def main():
    print("=" * 64)
    print("  RAG İNDEKSLEME — ChromaDB Vektör Veritabanı Oluşturma")
    print("=" * 64)

    # 1. Adım — belgeler hazırlanıyor
    print("\n[1/3] Belgeler yükleniyor ve parçalanıyor...")
    chunks = load_and_chunk()
    if not chunks:
        print("\n⚠️  Veri yok. data/raw/ klasörüne dosya ekleyin ve tekrar deneyin.")
        return
    print(f"      ✓ {len(chunks)} parça hazır.")

    # 2. Adım — RAG motoru başlatılıyor (model indirme dahil)
    print("\n[2/3] Embedding modeli hazırlanıyor...")
    print("      (ilk seferde ~120 MB indirilir, birkaç dakika sürebilir)")
    engine = RAGEngine()

    # 3. Adım — vektörleştirme
    print("\n[3/3] Vektörler oluşturuluyor ve ChromaDB'ye yazılıyor...")
    n = engine.index_documents(chunks)
    print(f"      ✓ {n} belge başarıyla indekslendi.")
    print(f"      📂 Konum: {engine.db_path}")

    # Doğrulama testi — örnek bir sorgu
    print("\n" + "=" * 64)
    print("  DOĞRULAMA TESTİ")
    print("=" * 64)

    test_queries = [
        "boğazım ağrıyor ve öksürüğüm var",
        "şiddetli karın ağrım var",
        "göğsümde basınç hissediyorum, kola vuruyor",
    ]

    for query in test_queries:
        print(f'\n🔎 Sorgu: "{query}"')
        results = engine.retrieve_with_scores(query, k=2)
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source_file", "?")
            snippet = doc.page_content[:140].replace("\n", " ")
            print(f"   [{i}] benzerlik: {score:.3f}  •  {source}")
            print(f"       {snippet}...")

    print("\n" + "=" * 64)
    print("  ✅ İndeksleme tamamlandı! Sistem sorgulara hazır.")
    print("=" * 64)


if __name__ == "__main__":
    main()
