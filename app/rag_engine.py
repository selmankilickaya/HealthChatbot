"""
rag_engine.py
RAG motoru — In-memory (sadece bellekte) çalışır, diske yazmaz.

ChromaDB yerine basit ama hızlı bir bellek içi vector store kullanır.
Avantaj: dosya sistemi yazma izni gerekmez (Streamlit Cloud için ideal).
Dezavantaj: her başlangıçta yeniden indekslenir (~30 sn).

Kullanım:
    engine = RAGEngine()
    engine.index_documents(chunks)
    results = engine.retrieve("burnum akıyor", k=4)
"""
from pathlib import Path
from typing import List, Optional, Tuple
import os
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Türkçe destekli, çok dilli embedding modeli (HuggingFace)
DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


class RAGEngine:
    """In-memory RAG motoru. Diske yazmaz."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        verbose: bool = True,
    ):
        # db_path artık kullanılmıyor, geriye dönük uyumluluk için kabul ediliyor
        self.db_path = Path(db_path) if db_path else None
        self.embedding_model_name = embedding_model
        self.verbose = verbose

        if verbose:
            print(f"  → Embedding modeli yükleniyor: {embedding_model}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Bellek içi depo
        self._documents: List[Document] = []
        self._vectors: Optional[np.ndarray] = None

    def is_indexed(self) -> bool:
        """İndeks dolu mu?"""
        return len(self._documents) > 0 and self._vectors is not None

    def document_count(self) -> int:
        return len(self._documents)

    def clear(self) -> None:
        """İndeksi temizle."""
        self._documents = []
        self._vectors = None

    def index_documents(self, documents: List[Document]) -> int:
        """
        Belgeleri vektörleştir, bellekte sakla.
        """
        if not documents:
            raise ValueError("Boş belge listesi verildi.")

        self.clear()

        if self.verbose:
            print(f"  → {len(documents)} parça vektörleştiriliyor...")

        # Tüm metinleri tek seferde embedding'e ver (verimli)
        texts = [doc.page_content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)

        self._documents = documents
        self._vectors = np.array(vectors, dtype=np.float32)

        return len(documents)

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Kosinüs benzerliği ile en yakın k belgeyi getir.
        Embeddings L2-normalize edildiği için dot product = cosine similarity.
        """
        if not self.is_indexed():
            raise RuntimeError(
                "Bellek içi indeks boş. Önce index_documents() çağırın."
            )

        # Sorguyu vektörleştir
        query_vec = np.array(
            self.embeddings.embed_query(query), dtype=np.float32
        )

        # Kosinüs benzerliği (normalize edilmiş vektörler için dot product)
        similarities = self._vectors @ query_vec  # shape: (n_docs,)

        # En yüksek k tanesinin indeksini al
        top_k_indices = np.argsort(similarities)[::-1][:k]

        return [self._documents[i] for i in top_k_indices]

    def retrieve_with_scores(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        retrieve() ile aynı, ama her sonuçla birlikte mesafe skoru döner.
        Düşük skor = daha benzer (kosinüs mesafesi = 1 - similarity).
        """
        if not self.is_indexed():
            raise RuntimeError(
                "Bellek içi indeks boş. Önce index_documents() çağırın."
            )

        query_vec = np.array(
            self.embeddings.embed_query(query), dtype=np.float32
        )
        similarities = self._vectors @ query_vec
        top_k_indices = np.argsort(similarities)[::-1][:k]

        return [
            (self._documents[i], float(1.0 - similarities[i]))
            for i in top_k_indices
        ]
