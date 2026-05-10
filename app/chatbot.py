"""
chatbot.py
LangChain tabanlı konuşma motoru — RAG ile zenginleştirilmiş çoklu tur diyalog.
Sonraki adımda doldurulacak.
"""


class HealthChatbot:
    """RAG-zenginleştirilmiş Türkçe sağlık chatbotu."""

    def __init__(self, rag_engine, llm_provider: str, model: str):
        self.rag_engine = rag_engine
        self.llm_provider = llm_provider
        self.model = model

    def respond(self, user_message: str, history: list) -> str:
        raise NotImplementedError("5. Adımda implemente edilecek.")
