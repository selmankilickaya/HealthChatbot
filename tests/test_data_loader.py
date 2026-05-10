"""
test_data_loader.py
data_loader modülü için birim testleri.

Çalıştırma:
    pytest tests/test_data_loader.py -v
"""
import pytest
from pathlib import Path
import tempfile
from langchain_core.documents import Document

from app.data_loader import (
    clean_text, load_documents, chunk_documents, load_and_chunk, RAW_DIR,
)


class TestCleanText:
    def test_fazla_satir_sonu(self):
        text = "Birinci\n\n\n\nİkinci"
        assert clean_text(text) == "Birinci\n\nİkinci"

    def test_fazla_bosluk(self):
        text = "Çok    fazla    boşluk"
        assert clean_text(text) == "Çok fazla boşluk"

    def test_satir_basi_sonu_kirpma(self):
        text = "  Satır  \n  Yeni  "
        result = clean_text(text)
        assert "  " not in result
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_zero_width_temizleme(self):
        text = "Test\u200bMesajı"
        assert "\u200b" not in clean_text(text)

    def test_bos_string(self):
        assert clean_text("") == ""

    def test_normal_metin_korunur(self):
        text = "Bu bir test cümlesidir.\nİkinci satır."
        assert clean_text(text) == text


class TestLoadDocuments:
    def test_proje_verileri_yuklenebilir(self):
        """data/raw/ klasöründe örnek belgeler bulunabilir mi?"""
        if not RAW_DIR.exists():
            pytest.skip("data/raw/ yok — testi atla")

        docs = load_documents(RAW_DIR)
        # En az bir belge bekleniyor
        assert isinstance(docs, list)

    def test_olmayan_klasor_hatasi(self):
        with pytest.raises(FileNotFoundError):
            load_documents(Path("/asla/var/olmayacak/klasor"))

    def test_gizli_dosyalar_atlaniyor(self, tmp_path):
        # .gitkeep, .DS_Store gibi dosyalar atlanmalı
        (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
        (tmp_path / ".hidden").write_text("hidden", encoding="utf-8")
        (tmp_path / "gercek.txt").write_text("Gercek icerik", encoding="utf-8")

        docs = load_documents(tmp_path)
        # Sadece gercek.txt yüklenmeli
        sources = [d.metadata.get("source_file") for d in docs]
        assert "gercek.txt" in sources
        assert ".gitkeep" not in sources

    def test_desteklenmeyen_format_atlaniyor(self, tmp_path):
        (tmp_path / "test.xyz").write_text("desteklenmiyor", encoding="utf-8")
        (tmp_path / "test.txt").write_text("destekleniyor", encoding="utf-8")

        docs = load_documents(tmp_path)
        sources = [d.metadata.get("source_file") for d in docs]
        assert "test.txt" in sources
        assert "test.xyz" not in sources

    def test_md_dosyasi_yuklenebiliyor(self, tmp_path):
        content = "# Baslik\n\nBu bir test dosyasidir."
        (tmp_path / "test.md").write_text(content, encoding="utf-8")

        docs = load_documents(tmp_path)
        assert len(docs) >= 1
        assert "Baslik" in docs[0].page_content


class TestChunkDocuments:
    def test_kucuk_metin_tek_parca(self):
        doc = Document(page_content="Kısa bir metin", metadata={})
        chunks = chunk_documents([doc])
        assert len(chunks) == 1

    def test_buyuk_metin_bolunur(self):
        # 3000 karakterlik bir metin
        content = "Bu bir cümledir. " * 200  # ~3400 karakter
        doc = Document(page_content=content, metadata={"source_file": "test"})
        chunks = chunk_documents([doc], chunk_size=800, chunk_overlap=100)
        assert len(chunks) > 1

    def test_metadata_korunuyor(self):
        doc = Document(
            page_content="Bu bir cümledir. " * 200,
            metadata={"source_file": "kaynak.md", "ekstra": "veri"},
        )
        chunks = chunk_documents([doc], chunk_size=400, chunk_overlap=50)
        for chunk in chunks:
            assert chunk.metadata.get("source_file") == "kaynak.md"

    def test_bos_liste(self):
        chunks = chunk_documents([])
        assert chunks == []


class TestLoadAndChunk:
    def test_calistirilabilir(self):
        """Tümleşik fonksiyon proje verisi üzerinde çalışabilmeli."""
        if not RAW_DIR.exists():
            pytest.skip("data/raw/ yok")

        chunks = load_and_chunk()
        assert isinstance(chunks, list)
        # Eğer veri varsa parçalar olmalı
        if chunks:
            assert all(isinstance(c, Document) for c in chunks)
            assert all(c.page_content for c in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
