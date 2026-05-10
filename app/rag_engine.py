"""
rag_engine.py
ChromaDB vektör veritabanı yönetimi ve anlamsal arama.
Sonraki adımda doldurulacak.
"""


class RAGEngine:
    """ChromaDB tabanlı RAG motoru."""

    def __init__(self, db_path: str, embedding_model: str):
        self.db_path = db_path
        self.embedding_model = embedding_model

    def index_documents(self, documents: list) -> None:
        raise NotImplementedError("4. Adımda implemente edilecek.")

    def retrieve(self, query: str, k: int = 4) -> list:
        raise NotImplementedError("4. Adımda implemente edilecek.")
