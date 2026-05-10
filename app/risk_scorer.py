"""
risk_scorer.py
Hibrit risk skorlama motoru: kural tabanlı kırmızı bayrak taraması +
LLM destekli değerlendirme + risk grubu düzeltmesi.

Akış:
    1. Kullanıcı metninde KIRMIZI BAYRAK ara → varsa anında ACİL
    2. Yoksa LLM'e (RAG bağlamıyla) sorgula → DÜŞÜK / ORTA / ACİL
    3. Risk grubu varsa (yaş, gebelik, kronik hastalık) bir üst seviyeye yükselt
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Callable
import re
import json


# ---------------------------------------------------------------------------
# Risk seviyeleri
# ---------------------------------------------------------------------------
class RiskLevel(str, Enum):
    LOW = "Düşük"
    MEDIUM = "Orta"
    URGENT = "Acil"

    @property
    def emoji(self) -> str:
        return {
            RiskLevel.LOW: "🟢",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.URGENT: "🔴",
        }[self]

    @property
    def numeric(self) -> int:
        return {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.URGENT: 3}[self]

    @classmethod
    def from_numeric(cls, n: int) -> "RiskLevel":
        n = max(1, min(3, n))
        return {1: cls.LOW, 2: cls.MEDIUM, 3: cls.URGENT}[n]


# ---------------------------------------------------------------------------
# Türkçe-aware küçük harf dönüşümü
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """
    Türkçe metni güvenli biçimde küçük harfe çevirir.
    Python'un default lower'ı 'İ' → 'i̇' (combining dot) yapıyor; bu
    eşleşmeyi kırıyor. Önce manuel düzeltiyoruz.
    """
    text = text.replace("İ", "i").replace("I", "ı")
    return text.lower()


def has_any(text: str, keywords: List[str]) -> bool:
    """Metinde verilen kelimelerden HERHANGİ BİRİ geçiyor mu?"""
    return any(kw in text for kw in keywords)


def has_all(text: str, keywords: List[str]) -> bool:
    """Metinde verilen kelimelerin TAMAMI geçiyor mu?"""
    return all(kw in text for kw in keywords)


# ---------------------------------------------------------------------------
# KIRMIZI BAYRAK KURALLARI
# Her kural: (isim, açıklama, dedektör fonksiyonu)
# Dedektör: normalize edilmiş text alır, True/False döner.
# Birden fazla kural eşleşse bile her biri ayrı listelenir.
# ---------------------------------------------------------------------------
RED_FLAG_RULES: List[tuple[str, str, Callable[[str], bool]]] = [
    # 1) İntihar / kendine zarar — ruhsal acil
    (
        "intihar_kendine_zarar",
        "Acil ruhsal sağlık desteği gerekli",
        lambda t: has_any(t, [
            "intihar",
            "kendime zarar",
            "kendime kıymak",
            "ölmek istiyor",
            "yaşamak istemi",
            "hayata son",
        ]),
    ),

    # 2) Anafilaksi — alerjik acil
    (
        "anafilaksi",
        "Olası anafilaksi (yüzde/dilde şişme, boğaz tıkanıklığı)",
        lambda t: (
            (has_any(t, ["yüz", "dudak", "dil"]) and "şiş" in t)
            or (has_any(t, ["boğaz", "boğazım"]) and has_any(t, ["kapan", "tıkan"]))
            or "anafilak" in t
        ),
    ),

    # 3) Kalp belirtileri — göğüs ağrısı/baskısı/sıkışması
    (
        "kalp_belirtileri",
        "Olası kalp belirtileri (göğüs ağrısı/baskısı, kola/çeneye yayılma)",
        lambda t: (
            # Göğüs + ağrı/baskı/sıkışma/basınç/yanma
            (has_any(t, ["göğüs", "göğs", "gögüs"])
             and has_any(t, ["ağrı", "baskı", "sıkış", "basınç", "yanma", "ezi"]))
            # Sol kola / çeneye / sırta yayılan ağrı
            or (has_any(t, ["sol kol", "sol kolum", "sol kola", "sol koluma"])
                and has_any(t, ["vur", "yayıl", "ağrı"]))
            or (has_any(t, ["çene", "çenem"])
                and has_any(t, ["vur", "yayıl", "ağrı"]))
            # Soğuk ter + göğüs ağrısı/baskı
            or ("soğuk ter" in t and has_any(t, ["göğüs", "göğs", "ağrı", "sıkış", "basınç"]))
        ),
    ),

    # 4) Solunum yetmezliği — nefes darlığı, morarma
    (
        "solunum_yetmezligi",
        "Solunum yetmezliği belirtileri (nefes darlığı, morarma)",
        lambda t: (
            has_any(t, [
                "nefes alam",
                "nefes alma",
                "nefes darlığ",
                "nefes darlik",
                "soluk alam",
                "soluk alma",
                "morarma",
                "siyanoz",
                "boğuluyor",
                "boğulacak",
            ])
            # Nefes/soluk + zorlanma
            or (has_any(t, ["nefes", "soluk"])
                and has_any(t, ["zorlan", "yetmiyor", "yetersiz"]))
            or (has_any(t, ["dudak", "dudaklarım", "parmak"])
                and "mor" in t)
            or ("astım" in t and has_any(t, [
                "geçmiyor", "etki etmiyor", "fayda etmiyor",
                "hiç fayda", "iş görmüyor",
            ]))
        ),
    ),

    # 5) Nörolojik acil — inme, nöbet, ani şiddetli baş ağrısı
    (
        "norolojik_acil",
        "Olası nörolojik acil (inme, nöbet, ani şiddetli baş ağrısı)",
        lambda t: (
            has_any(t, [
                "bilinç kayb",
                "konuşam",
                "konuşmakta zorlan",
                "konuşmam zor",
                "felç",
                "inme",
                "nöbet",
                "ense sertli",
                "görme kayb",
                "çift gör",
            ])
            # Yüz felci belirtisi
            or (has_any(t, ["yüz", "yüzüm", "ağzım"])
                and has_any(t, ["kay", "düşt", "çarpı"]))
            # "Hayatımın en şiddetli baş ağrısı"
            or ("baş ağrı" in t and has_any(t, [
                "hayatımın en", "en şiddetli", "en kötü", "ani başla",
                "patlar gibi", "yıldırım gibi", "aniden başla",
            ]))
            # Tek taraflı güçsüzlük + ekstremite
            or (has_any(t, ["tek tarafım", "sağ tarafım", "sol tarafım"])
                and has_any(t, ["güç kayb", "kalkamı", "tutmu", "hissetmi"]))
            # Cauda equina: bel ağrısı + idrar/dışkı kaçırma
            or (has_any(t, ["bel", "belim"])
                and has_any(t, ["idrar tutamı", "idrar kaçır",
                                "dışkı tutamı", "dışkı kaçır",
                                "kaka tutamı"]))
        ),
    ),

    # 6) Şiddetli kanama / iç kanama
    (
        "siddetli_kanama",
        "Şiddetli kanama veya iç kanama belirtisi",
        lambda t: (
            has_any(t, [
                "şiddetli kanama",
                "kan tükür",
                "hemoptizi",
                "katran kıvam",
            ])
            # Durdurulamayan kanama
            or ("kanama" in t and has_any(t, ["durmu", "durduram", "duramı"]))
            # Kusmukta veya dışkıda kan
            or ("kus" in t and has_any(t, ["kan", "kanlı"]))
            or (has_any(t, ["dışkı", "kaka"]) and has_any(t, ["kan", "siyah", "katran"]))
            or "kanlı kus" in t
        ),
    ),

    # 7) Akut karın — şiddetli karın ağrısı, defans
    (
        "akut_karin",
        "Olası akut karın (şiddetli karın ağrısı, dokunma hassasiyeti)",
        lambda t: (
            # Karın + şiddet ifadesi + ağrı
            (has_any(t, ["karın", "karnım", "karnıma", "karında"])
             and has_any(t, ["şiddetli", "çok fena", "çok kötü", "dayanılm",
                             "müthiş", "çok şiddetli", "feci", "yoğun"])
             and has_any(t, ["ağrı", "sancı"]))
            # Karın sertliği / dokunma intoleransı
            or (has_any(t, ["karın", "karnım", "karnıma"])
                and has_any(t, ["sert", "tahta gibi", "dokunduram",
                                "dokundurmuyor", "elletmiyor", "elletme"]))
            # Sağ alt karın (apandisit klasiği)
            or ("sağ alt" in t and "karın" in t)
            or ("apandisit" in t)
        ),
    ),

    # 8) Yüksek ateş + komplikasyon
    (
        "yuksek_ates_komplikasyon",
        "Yüksek ateş ile birlikte komplikasyon belirtisi",
        lambda t: (
            # Çok yüksek ateş ifadesi
            has_any(t, ["40 derece", "40°", "41 derece", "41°"])
            or (re.search(r"\b40[\.,]\d", t) is not None
                and "derece" in t)
            # Ateş + bilinç değişikliği / kasılma
            or (has_any(t, ["ateş", "yüksek ateş"])
                and has_any(t, ["bilinç", "baygın", "kasılma", "havale", "titreme şiddetli"]))
        ),
    ),

    # 9) Şiddetli dehidratasyon
    (
        "agir_dehidratasyon",
        "Ağır dehidratasyon belirtileri",
        lambda t: (
            has_any(t, [
                "idrar yapamı",
                "idrar yapmı",
                "gözleri çukur",
                "göz çukur",
                "ağzım çok kuru",
                "su tutam",
                "sıvı tutam",
                "her şeyi kus",
            ])
            and has_any(t, ["günlerdir", "gündür", "saatlerdir", "kusma", "ishal"])
        ),
    ),
]


# ---------------------------------------------------------------------------
# Kullanıcı bağlamı (risk grubu düzeltmesi için)
# ---------------------------------------------------------------------------
@dataclass
class UserContext:
    """Risk grubunu belirleyen kullanıcı bilgisi (opsiyonel)."""
    age: Optional[int] = None
    is_pregnant: bool = False
    has_chronic_disease: bool = False
    is_immunocompromised: bool = False
    is_infant: bool = False  # 3 ay altı bebek

    def is_risk_group(self) -> tuple[bool, str]:
        reasons = []
        if self.is_infant or (self.age is not None and self.age < 1):
            reasons.append("bebek (1 yaş altı)")
        elif self.age is not None and self.age < 5:
            reasons.append("küçük çocuk (5 yaş altı)")
        elif self.age is not None and self.age >= 65:
            reasons.append("65 yaş üstü")

        if self.is_pregnant:
            reasons.append("gebelik")
        if self.has_chronic_disease:
            reasons.append("kronik hastalık")
        if self.is_immunocompromised:
            reasons.append("bağışıklık baskılanması")

        return (len(reasons) > 0, ", ".join(reasons))


# ---------------------------------------------------------------------------
# Skorlama sonucu
# ---------------------------------------------------------------------------
@dataclass
class RiskAssessment:
    level: RiskLevel
    reasoning: str
    red_flags: List[str] = field(default_factory=list)
    llm_assessment: Optional[str] = None
    risk_group_uplift: bool = False
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "emoji": self.level.emoji,
            "reasoning": self.reasoning,
            "red_flags": self.red_flags,
            "risk_group_uplift": self.risk_group_uplift,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# LLM skorlama prompt'u
# ---------------------------------------------------------------------------
LLM_SCORING_PROMPT = """Sen bir tıbbi triaj asistanısın. Aşağıdaki kullanıcı \
metnini ve ilgili tıbbi referans bağlamını değerlendirip, riski şu üç \
seviyeden BİRİNE sınıflandır:

- DUSUK: Hafif, kendi kendine geçebilecek belirtiler. Evde takip yeterli.
- ORTA: Aile hekimi/poliklinik değerlendirmesi gereken (24-48 saat içinde).
- ACIL: Derhal acil servise / 112'ye yönlendirme gereken durumlar.

## REFERANS BAĞLAM
{rag_context}

## KULLANICI METNİ
{user_text}

## GÖREV
Sadece aşağıdaki JSON formatında cevap ver, başka HİÇBİR ŞEY yazma:

{{"level": "DUSUK", "reasoning": "Kararın gerekçesi (1-2 cümle, Türkçe)", \
"confidence": 0.85}}

Olası level değerleri sadece: "DUSUK", "ORTA" veya "ACIL"

JSON:"""


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------
class RiskScorer:
    """Hibrit risk skorlama motoru."""

    def __init__(self, llm=None):
        """
        Args:
            llm: LangChain chat modeli (örn: ChatAnthropic). Yoksa sadece
                 kural tabanlı tarama çalışır.
        """
        self.llm = llm

    def scan_red_flags(self, text: str) -> List[str]:
        """Metinde kırmızı bayrak ara, eşleşen açıklamaların listesini döndür."""
        normalized = normalize_text(text)
        found = []
        for name, description, detector in RED_FLAG_RULES:
            try:
                if detector(normalized):
                    found.append(description)
            except Exception:
                # Bir kural patladıysa diğerlerini etkilemesin
                continue
        return found

    def llm_score(
        self, user_text: str, rag_context: str = ""
    ) -> tuple[RiskLevel, str, float]:
        """LLM'e sor, (seviye, gerekçe, güven) döndür."""
        if self.llm is None:
            # LLM yoksa varsayılan olarak DÜŞÜK döner; risk grubu uplift
            # gerekirse onu sonra ORTA'ya yükseltir.
            return RiskLevel.LOW, "LLM mevcut değil; varsayılan DÜŞÜK.", 0.3

        prompt = LLM_SCORING_PROMPT.format(
            rag_context=rag_context if rag_context else "(referans yok)",
            user_text=user_text,
        )

        try:
            response = self.llm.invoke(prompt)
            raw = response.content.strip()

            # JSON'u parse et (LLM bazen ekstra metin koyabilir)
            json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if not json_match:
                return (
                    RiskLevel.MEDIUM,
                    f"LLM çıktısı parse edilemedi.",
                    0.3,
                )

            data = json.loads(json_match.group())
            level_str = str(data.get("level", "ORTA")).upper()
            level_map = {
                "DUSUK": RiskLevel.LOW,
                "DÜŞÜK": RiskLevel.LOW,
                "DUŞUK": RiskLevel.LOW,
                "ORTA": RiskLevel.MEDIUM,
                "ACIL": RiskLevel.URGENT,
                "ACİL": RiskLevel.URGENT,
            }
            level = level_map.get(level_str, RiskLevel.MEDIUM)
            reasoning = data.get("reasoning", "(gerekçe yok)")
            confidence = float(data.get("confidence", 0.5))
            return level, reasoning, max(0.0, min(1.0, confidence))

        except Exception as exc:
            return RiskLevel.MEDIUM, f"LLM hatası: {exc}", 0.2

    def apply_risk_group_uplift(
        self, level: RiskLevel, user_context: UserContext
    ) -> tuple[RiskLevel, bool, str]:
        """Risk grubuna göre seviyeyi bir üst kademeye çıkarabilir."""
        is_risk, reason = user_context.is_risk_group()
        if not is_risk:
            return level, False, ""

        if level == RiskLevel.LOW:
            return RiskLevel.MEDIUM, True, f"Risk grubu ({reason}) → ORTA'ya yükseltildi"
        if level == RiskLevel.MEDIUM:
            return RiskLevel.URGENT, True, f"Risk grubu ({reason}) → ACİL'e yükseltildi"
        return level, False, ""

    def score(
        self,
        user_text: str,
        rag_context: str = "",
        user_context: Optional[UserContext] = None,
    ) -> RiskAssessment:
        """
        Üç aşamalı skorlama yap, RiskAssessment döndür.
        """
        if user_context is None:
            user_context = UserContext()

        # 1) Kural tabanlı kırmızı bayrak
        red_flags = self.scan_red_flags(user_text)
        if red_flags:
            return RiskAssessment(
                level=RiskLevel.URGENT,
                reasoning="Kural tabanlı tarama kırmızı bayrak tespit etti.",
                red_flags=red_flags,
                confidence=1.0,
            )

        # 2) LLM
        llm_level, llm_reasoning, confidence = self.llm_score(user_text, rag_context)

        # 3) Risk grubu düzeltmesi
        final_level, uplifted, uplift_reason = self.apply_risk_group_uplift(
            llm_level, user_context
        )

        reasoning_parts = [f"LLM: {llm_reasoning}"]
        if uplifted:
            reasoning_parts.append(uplift_reason)
        reasoning = " | ".join(reasoning_parts)

        return RiskAssessment(
            level=final_level,
            reasoning=reasoning,
            red_flags=[],
            llm_assessment=llm_reasoning,
            risk_group_uplift=uplifted,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Bağımsız test (LLM gerektirmez)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 64)
    print("  RISK SKORLAMA — KURAL TABANLI TEST (LLM'siz)")
    print("=" * 64)

    scorer = RiskScorer(llm=None)

    test_cases = [
        # (metin, beklenen_seviye, açıklama)
        ("Burnum akıyor, hafif öksürüğüm var",
         RiskLevel.LOW, "Hafif belirti — LLM'siz default DÜŞÜK"),

        ("Göğsümde basınç hissediyorum, sol koluma vuruyor",
         RiskLevel.URGENT, "Klasik kalp belirtisi — kural tabanlı"),

        ("Göğsüm ağrıyor, çenem de ağrıyor",
         RiskLevel.URGENT, "Çeneye yayılan göğüs ağrısı"),

        ("Nefes alamıyorum, dudaklarım moraryor",
         RiskLevel.URGENT, "Solunum yetmezliği"),

        ("Karnım çok fena ağrıyor, hiç dokundurmuyorum",
         RiskLevel.URGENT, "Akut karın — defans"),

        ("Sağ alt karnımda şiddetli ağrı var",
         RiskLevel.URGENT, "Apandisit klasiği"),

        ("3 gündür hafif baş ağrım var",
         RiskLevel.LOW, "Belirsiz — LLM'siz default DÜŞÜK"),

        ("Hayatımın en şiddetli baş ağrısı, ani başladı",
         RiskLevel.URGENT, "Tunderclap baş ağrısı — nörolojik"),

        ("İntihar etmek istiyorum",
         RiskLevel.URGENT, "Ruhsal acil"),

        ("Yüzüm ve dudaklarım şişti, nefes alamıyorum",
         RiskLevel.URGENT, "Anafilaksi"),

        ("İki gündür kusuyorum, kanlı kustum",
         RiskLevel.URGENT, "İç kanama belirtisi"),

        ("Mide bulantım var, hafif",
         RiskLevel.LOW, "Hafif sindirim — default DÜŞÜK"),
    ]

    pass_count = 0
    for text, expected, note in test_cases:
        result = scorer.score(text)
        ok = result.level == expected
        if ok:
            pass_count += 1
        symbol = "✓" if ok else "✗"
        print(f"\n{symbol} {note}")
        print(f"  📝 \"{text}\"")
        print(f"  🎯 Beklenen: {expected.emoji} {expected.value}  →  "
              f"Sonuç: {result.level.emoji} {result.level.value}")
        if result.red_flags:
            for rf in result.red_flags:
                print(f"     🚨 {rf}")

    print("\n" + "=" * 64)
    print(f"  SONUÇ: {pass_count}/{len(test_cases)} test geçti")
    print("=" * 64)

    # Risk grubu düzeltmesi
    print("\n" + "=" * 64)
    print("  RİSK GRUBU DÜZELTMESİ TESTİ")
    print("=" * 64)

    elderly = UserContext(age=72, has_chronic_disease=True)
    result = scorer.score("Halsizim ve iştahsızım", user_context=elderly)
    print(f"\n  72 yaş + kronik hastalık + hafif şikayet:")
    print(f"     {result.level.emoji} {result.level.value}")
    print(f"     Yükseltildi: {result.risk_group_uplift}")
    print(f"     Gerekçe: {result.reasoning}")

    pregnant = UserContext(age=30, is_pregnant=True)
    result = scorer.score("Hafif baş ağrım var", user_context=pregnant)
    print(f"\n  30 yaş + gebelik + hafif baş ağrısı:")
    print(f"     {result.level.emoji} {result.level.value}")
    print(f"     Yükseltildi: {result.risk_group_uplift}")
