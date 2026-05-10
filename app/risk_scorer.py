"""
risk_scorer.py
Hibrit risk skorlama: kural tabanlı + LLM destekli sınıflandırma.
Sonraki adımda doldurulacak.
"""
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "Düşük"
    MEDIUM = "Orta"
    URGENT = "Acil"


class RiskScorer:
    """Toplanan semptomları Düşük / Orta / Acil olarak sınıflandırır."""

    def score(self, symptoms: list, llm_assessment: str | None = None) -> RiskLevel:
        raise NotImplementedError("6. Adımda implemente edilecek.")
