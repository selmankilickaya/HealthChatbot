"""
data_loader.py
data/raw/ klasöründeki belgeleri yükler, temizler ve parçalara böler.

Desteklenen formatlar: .pdf, .txt, .md

Bağımsız çalıştırma:
    python -m app.data_loader
"""
from pathlib import Path
from typing import List
import re

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# Proje kök dizini (app/'in bir üstü)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def clean_text(text: str) -> str:
    """Metni temizler: fazla boşluk, kontrol karakterleri, vb."""
    # 3+ ardışık satır sonunu 2'ye indir
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 2+ ardışık boşluğu 1'e indir
    text = re.sub(r" {2,}", " ", text)
    # Sıfır genişlikli ve BOM karakterleri
    text = text.replace("\u200b", "").replace("\ufeff", "")
    # Satır başı/sonu boşluklarını kırp
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def load_documents(raw_dir: Path = RAW_DIR) -> List[Document]:
    """
    raw_dir içindeki tüm desteklenen dosyaları yükler.

    Döndürür:
        Document listesi (her belge .page_content ve .metadata içerir)
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Ham veri klasörü bulunamadı: {raw_dir}")

    documents: List[Document] = []

    for file_path in sorted(raw_dir.iterdir()):
        # Gizli dosyaları ve .gitkeep'i atla
        if file_path.name.startswith("."):
            continue
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
            elif suffix in (".txt", ".md"):
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
            else:
                print(f"  ⊘ Atlandı (desteklenmeyen format): {file_path.name}")
                continue

            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata["source_file"] = file_path.name

            documents.extend(docs)
            print(f"  ✓ Yüklendi: {file_path.name}  ({len(docs)} sayfa/blok)")

        except Exception as exc:
            print(f"  ✗ Hata — {file_path.name}: {exc}")

    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    Belgeleri RAG için anlamlı parçalara böler.

    chunk_size: Karakter cinsinden hedef parça boyutu
    chunk_overlap: Komşu parçalar arası örtüşme (bağlam kaybını önler)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    return chunks


def load_and_chunk() -> List[Document]:
    """Tek adımda tüm veriyi yükle ve parçala. Diğer modüller bunu çağırır."""
    docs = load_documents()
    chunks = chunk_documents(docs)
    return chunks


# ---------------------------------------------------------------------------
# Bağımsız çalıştırma — `python -m app.data_loader`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"📂 Veri klasörü: {RAW_DIR}\n")
    print("→ Belgeler yükleniyor...")
    docs = load_documents()
    print(f"\n✓ Toplam {len(docs)} belge yüklendi.\n")

    print("→ Parçalara bölünüyor...")
    chunks = chunk_documents(docs)
    print(f"✓ Toplam {len(chunks)} parça oluşturuldu.\n")

    if not chunks:
        print("⚠️  Parça yok — data/raw/ içine veri ekleyin.")
    else:
        print("─" * 60)
        print("ÖRNEK PARÇA (ilk parça):")
        print("─" * 60)
        first = chunks[0]
        print(f"📎 Kaynak: {first.metadata.get('source_file')}")
        print(f"📏 Uzunluk: {len(first.page_content)} karakter")
        print(f"\n{first.page_content[:400]}...")
        print("─" * 60)

        # Kaynak başına parça sayısı
        from collections import Counter

        source_counts = Counter(
            c.metadata.get("source_file", "?") for c in chunks
        )
        print("\n📊 Kaynak başına parça sayısı:")
        for src, n in source_counts.most_common():
            print(f"   {src:40s} {n:3d} parça")
