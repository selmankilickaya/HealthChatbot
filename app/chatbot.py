"""
chatbot.py
RAG-zenginleştirilmiş Türkçe sağlık chatbotu — LangChain + Claude API.

Akış:
    1. Kullanıcı mesajı al
    2. RAG motorundan en alakalı 4 belge parçası getir
    3. Sistem prompt'u + bağlam + geçmiş ile Claude'a gönder
    4. Cevabı kullanıcıya döndür (kaynaklarla birlikte)
"""
from typing import List, Dict, Tuple, Optional
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

from app.rag_engine import RAGEngine


# ---------------------------------------------------------------------------
# Sistem Prompt'u — Claude'un kişiliği ve davranış kuralları
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Sen, Türkiye'de yaşayan kullanıcılar için tasarlanmış bir sağlık \
karar destek asistanısın. Adın "SağlıkAsistanı". Görevin TANIYICILIK DEĞİL; \
kullanıcının semptomlarını anlayıp, sağlanan tıbbi referans belgelerine \
dayanarak onu DOĞRU SAĞLIK KARARINA YÖNLENDİRMEKTİR.

## TEMEL KURALLAR (Asla ihlal etme)

1. **TANI KOYMA.** "Şu hastalığa sahipsin" gibi bir ifade ASLA kullanma. \
Yerine "şu durum olabilir, ama mutlaka hekime başvurmalısınız" gibi yönlendirici \
ifadeler kullan.

2. **KAYNAK BAĞLAMI DIŞINA ÇIKMA.** Sana sağlanan REFERANS BELGELERİ dışındaki \
bilgiyi kullanma. Cevap belgelerde yoksa "Bu konuda elimde yeterli bilgi yok, \
lütfen bir sağlık kuruluşuna başvurun" de.

3. **İLAÇ ÖNERME.** Hiçbir koşulda spesifik ilaç adı, dozaj veya marka önerme. \
"Hekiminizin uygun göreceği ilaçlar" gibi genel ifadeler kullan.

4. **ACİL DURUMLARI ATLAMAZ.** Eğer kullanıcının anlattıkları kırmızı bayrak \
(göğüs ağrısı, nefes darlığı, bilinç kaybı, kanama, inme belirtileri vs.) \
içeriyorsa, en başta "🚨 BU DURUM ACİL OLABİLİR. Lütfen DERHAL 112'yi arayın \
veya en yakın acil servise başvurun." uyarısı yap.

## CEVAP FORMATI

Cevabın şu yapıda olmalı:

- **Empati**: Kısa bir empati cümlesi ile başla.
- **Ek bilgi soruları**: Eğer semptomlar yetersizse, 1-3 net soru sor (yaş, \
süre, eşlik eden belirtiler, kronik hastalık vb.). Tüm soruları aynı anda \
yığma — en kritik 1-2 tanesini sor.
- **Bilgilendirme**: Belgelerden alınan ilgili bilgiyi sade Türkçe ile özetle.
- **Yönlendirme**: Risk düzeyine göre net öneri ver:
   - 🟢 Düşük: "Evde takip edebilirsiniz, ancak X gün içinde geçmezse aile \
hekimine başvurun."
   - 🟡 Orta: "En kısa sürede aile hekimi veya polikliniğe başvurmanız \
önerilir."
   - 🔴 Acil: "🚨 Lütfen DERHAL 112'yi arayın veya acil servise başvurun."

## TARZ

- Sade, anlaşılır Türkçe kullan. Tıbbi jargondan kaçın; gerekirse parantez \
içinde açıkla (örn: "siyanoz" yerine "dudaklarda morarma").
- Empatik ama abartısız. Korkutma; ama küçümseme de.
- Maksimum 200 kelime. Madde işareti kullanmaktan çekinme.
- Her cevabın sonuna küçük bir not ekle: \
"⚠️ Bu bir tanı değildir, yalnızca bilgilendirme amaçlıdır."

## ŞİMDİ KULLANICIYI DİNLE

Aşağıda sana, kullanıcının sorusuna cevap vermek için kullanabileceğin \
REFERANS BELGELERİ ve KONUŞMA GEÇMİŞİ verilecek. Sadece bu bilgilerle \
kullanıcıya yardım et."""


def format_context(documents: List[Document]) -> str:
    """RAG'dan gelen belgeleri prompt'a uygun formata çevirir."""
    if not documents:
        return "(Konuyla ilgili referans belge bulunamadı.)"

    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", "?")
        content = doc.page_content.strip()
        parts.append(f"[Belge {i} — Kaynak: {source}]\n{content}")
    return "\n\n".join(parts)


class HealthChatbot:
    """RAG ile zenginleştirilmiş Türkçe sağlık chatbotu."""

    def __init__(
        self,
        rag_engine: RAGEngine,
        model: str = "claude-haiku-4-5-20251001",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 800,
        retrieve_k: int = 4,
    ):
        # API anahtarı: .env'den veya parametre olarak gelebilir
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key.startswith("sk-ant-..."):
            raise ValueError(
                "ANTHROPIC_API_KEY tanımlı değil. .env dosyasına gerçek "
                "anahtarınızı girin."
            )

        self.rag_engine = rag_engine
        self.retrieve_k = retrieve_k
        self.model_name = model

        # Düşük sıcaklık (0.3) → tutarlı, az hayali cevaplar
        self.llm = ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def respond(
        self,
        user_message: str,
        history: Optional[List[Dict]] = None,
    ) -> Tuple[str, List[Document]]:
        """
        Kullanıcı mesajına RAG-zenginleştirilmiş cevap üretir.

        Args:
            user_message: Kullanıcının yeni mesajı.
            history: Önceki mesajlar listesi
                    [{"role": "user|assistant", "content": "..."}]

        Returns:
            (cevap_metni, kullanılan_belgeler) — UI'da kaynak gösterimi için
        """
        # 1. RAG: en alakalı belgeleri getir
        retrieved_docs = self.rag_engine.retrieve(
            user_message, k=self.retrieve_k
        )
        context = format_context(retrieved_docs)

        # 2. Mesaj zincirini hazırla
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Konuşma geçmişini ekle (varsa)
        if history:
            for msg in history:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        # Bağlam + yeni soru
        contextualized_message = (
            f"### REFERANS BELGELER\n{context}\n\n"
            f"### KULLANICI SORUSU\n{user_message}"
        )
        messages.append(HumanMessage(content=contextualized_message))

        # 3. Claude API çağrısı
        response = self.llm.invoke(messages)
        answer = response.content

        return answer, retrieved_docs


# ---------------------------------------------------------------------------
# Bağımsız test — `python -m app.chatbot`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 64)
    print("  CHATBOT TEST")
    print("=" * 64)

    print("\nRAG motoru başlatılıyor...")
    rag = RAGEngine(verbose=False)

    if not rag.is_indexed():
        print("\n⚠️  ChromaDB boş. Önce indeksleyin:")
        print("    python scripts/build_index.py")
        exit(1)

    print(f"  ✓ {rag.document_count()} belge indekste hazır.")

    try:
        bot = HealthChatbot(rag_engine=rag)
        print(f"  ✓ Claude bağlantısı kuruldu (model: {bot.model_name})")
    except ValueError as e:
        print(f"\n❌ {e}")
        exit(1)

    # Test soruları
    test_questions = [
        "3 gündür öksürüğüm ve hafif ateşim var, ne yapmalıyım?",
        "Karnım çok fena ağrıyor, sağ tarafımın altında, kusma da var.",
        "Göğsümde basınç hissediyorum ve sol koluma vuruyor.",
    ]

    history: List[Dict] = []
    for q in test_questions:
        print("\n" + "─" * 64)
        print(f"👤 Kullanıcı: {q}")
        print("─" * 64)

        answer, sources = bot.respond(q, history=history)

        print(f"🤖 Asistan:\n{answer}")
        print(f"\n📎 Kullanılan kaynaklar:")
        for s in sources:
            print(f"   • {s.metadata.get('source_file')}")

        # Geçmişe ekle (multi-turn için)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})

    print("\n" + "=" * 64)
    print("  ✅ Test tamamlandı.")
    print("=" * 64)
