"""
rag_engine.py
ChromaDB tabanlı RAG motoru: belgeleri vektörleştirir ve anlamsal arama yapar.

Kullanım:
    engine = RAGEngine()
    if not engine.is_indexed():
        engine.index_documents(chunks)
    results = engine.retrieve("burnum akıyor", k=4)
"""
from pathlib import Path
from typing import List, Optional, Tuple
import os
import shutil

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_default_db_path() -> Path:
    """
    Veritabanı yolunu ortama göre seç:
    - Yerel ortam: data/chroma_db/ (kalıcı)
    - Streamlit Cloud: /tmp/chroma_db/ (yazılabilir, geçici)
    - Ortam değişkeni varsa onu kullan
    """
    # 1. Ortam değişkeni öncelikli
    env_path = os.getenv("CHROMA_DB_PATH")
    if env_path:
        return Path(env_path)

    # 2. Streamlit Cloud algıla — /mount/src altında çalışır, repo readonly
    if Path("/mount/src").exists() or os.getenv("STREAMLIT_SHARING") or os.getenv("STREAMLIT_RUNTIME"):
        return Path("/tmp/chroma_db")

    # 3. Yerel — proje klasörüne yaz
    return PROJECT_ROOT / "data" / "chroma_db"


DEFAULT_DB_PATH = _get_default_db_path()

# Türkçe destekli, çok dilli embedding modeli (HuggingFace)
# ~120 MB, ilk kullanımda otomatik indirilir, sonra önbellekten kullanılır
DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

COLLECTION_NAME = "saglik_belgeleri"


class RAGEngine:
    """Türkçe embedding + ChromaDB ile RAG motoru."""

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        verbose: bool = True,
    ):
        self.db_path = Path(db_path)
        self.embedding_model_name = embedding_model
        self.verbose = verbose

        if verbose:
            print(f"  → Embedding modeli yükleniyor: {embedding_model}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vectorstore: Optional[Chroma] = None

    @property
    def vectorstore(self) -> Chroma:
        """Vector store'u tembel başlat (sadece gerekince)."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(self.db_path),
            )
        return self._vectorstore

    def is_indexed(self) -> bool:
        """ChromaDB içinde indekslenmiş belge var mı?"""
        try:
            return self.vectorstore._collection.count() > 0
        except Exception:
            return False

    def document_count(self) -> int:
        """İndekslenmiş belge sayısını döndürür."""
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        """Mevcut indeksi tamamen siler (sıfırdan başlamak için)."""
        self._vectorstore = None
        if self.db_path.exists():
            shutil.rmtree(self.db_path)

    def index_documents(self, documents: List[Document]) -> int:
        """
        Belgeleri vektörleştir ve ChromaDB'ye yaz.
        Mevcut indeks varsa silinip yeniden oluşturulur.
        """
        if not documents:
            raise ValueError("Boş belge listesi verildi.")

        self.clear()
        self.db_path.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"  → {len(documents)} parça vektörleştiriliyor...")

        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(self.db_path),
        )
        return len(documents)

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Sorguya en yakın k belgeyi getirir.

        Args:
            query: Kullanıcı sorgusu (Türkçe doğal dil)
            k: Getirilecek belge sayısı

        Returns:
            En alakalı k belgenin listesi
        """
        if not self.is_indexed():
            raise RuntimeError(
                "ChromaDB boş. Önce indeksle: "
                "python scripts/build_index.py"
            )
        return self.vectorstore.similarity_search(query, k=k)

    def retrieve_with_scores(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        retrieve() ile aynı, ama her sonuçla birlikte benzerlik skoru döner.
        Skor düşük = daha benzer (mesafe metriği).
        """
        if not self.is_indexed():
            raise RuntimeError(
                "ChromaDB boş. Önce indeksle: "
                "python scripts/build_index.py"
            )
        return self.vectorstore.similarity_search_with_score(query, k=k)
