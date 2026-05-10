"""
streamlit_app.py
Türkçe Semptom Değerlendirici — Ana Streamlit arayüzü.

Çalıştırma:
    streamlit run streamlit_app.py
"""
from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv

from app.rag_engine import RAGEngine
from app.chatbot import HealthChatbot
from app.risk_scorer import RiskScorer, RiskLevel, UserContext


# ---------------------------------------------------------------------------
# Sayfa konfigürasyonu
# ---------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Türkçe Semptom Değerlendirici",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)


DISCLAIMER = """
⚠️ **Önemli Uyarı:** Bu uygulama bir tanı aracı **değildir**.
Yalnızca bilgilendirme ve yönlendirme amaçlıdır. Acil durumlarda **112**'yi arayın.
Şikayetleriniz için bir sağlık kuruluşuna başvurmanız önerilir.
"""

WELCOME_MESSAGE = (
    "Merhaba 👋 Ben SağlıkAsistanı. Şikayetlerinizi anlatın, size yardımcı "
    "olmaya çalışayım. Yan panelden yaş ve sağlık durumu bilgilerinizi "
    "girerseniz daha doğru yönlendirme yapabilirim.\n\n"
    "_Unutmayın: Ben bir tanı aracı değilim, sadece doğru sağlık kararına "
    "yönlendiren bir asistanım._"
)


# ---------------------------------------------------------------------------
# Önbelleğe alınmış sistem başlatıcılar (Streamlit cache)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Bilgi tabanı yükleniyor...")
def get_rag_engine() -> RAGEngine:
    """RAG motorunu bir kez başlat, oturumlar arası paylaş."""
    engine = RAGEngine(verbose=False)
    return engine


@st.cache_resource(show_spinner="Asistan hazırlanıyor...")
def get_chatbot(_rag_engine: RAGEngine) -> HealthChatbot:
    """Chatbot'u bir kez başlat (API anahtarı ortamdan alınır)."""
    return HealthChatbot(rag_engine=_rag_engine)


@st.cache_resource
def get_risk_scorer(_chatbot: HealthChatbot) -> RiskScorer:
    """Risk skorlayıcı, chatbot'un LLM'ini paylaşır (ekstra API yükü yok)."""
    return RiskScorer(llm=_chatbot.llm)


# ---------------------------------------------------------------------------
# Sistem hazırlık kontrolü
# ---------------------------------------------------------------------------
def check_system_ready() -> tuple[bool, str]:
    """ChromaDB ve API key durumunu kontrol eder."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key.startswith("sk-ant-..."):
        return False, (
            "❌ **ANTHROPIC_API_KEY** tanımlı değil.\n\n"
            "1. Proje kökünde `.env` dosyasını aç\n"
            "2. `ANTHROPIC_API_KEY=sk-ant-...` satırına gerçek anahtarınızı yapıştır\n"
            "3. Streamlit'i yeniden başlat (`Ctrl+C` → tekrar `streamlit run`)"
        )

    try:
        rag = get_rag_engine()
        if not rag.is_indexed():
            return False, (
                "❌ **Bilgi tabanı boş.**\n\n"
                "Aşağıdaki komutu terminalde çalıştırın:\n\n"
                "```\npython scripts/build_index.py\n```"
            )
        return True, ""
    except Exception as e:
        return False, f"❌ Sistem başlatılamadı: {e}"


# ---------------------------------------------------------------------------
# Yan panel — kullanıcı profili
# ---------------------------------------------------------------------------
def render_sidebar() -> UserContext:
    """Yan panelde kullanıcı profilini topla, UserContext döndür."""
    with st.sidebar:
        st.header("👤 Profil")
        st.caption("Bilgileriniz daha doğru yönlendirme için kullanılır.")

        age_choice = st.selectbox(
            "Yaş aralığı",
            options=["Belirtmek istemiyorum", "0-1 yaş (bebek)", "1-5 yaş",
                     "6-17 yaş", "18-39 yaş", "40-64 yaş", "65 yaş ve üzeri"],
            index=0,
        )

        # Yaş eşlemesi
        age_map = {
            "0-1 yaş (bebek)": 0,
            "1-5 yaş": 3,
            "6-17 yaş": 12,
            "18-39 yaş": 28,
            "40-64 yaş": 52,
            "65 yaş ve üzeri": 70,
        }
        age = age_map.get(age_choice)
        is_infant = age_choice == "0-1 yaş (bebek)"

        is_pregnant = st.checkbox("Gebelik")
        has_chronic = st.checkbox(
            "Kronik hastalığım var",
            help="KOAH, astım, kalp hastalığı, diyabet, böbrek hastalığı vb.",
        )
        is_immuno = st.checkbox(
            "Bağışıklığım baskılanmış",
            help="Kanser tedavisi, organ nakli, immünsupresif ilaç kullanımı.",
        )

        st.divider()

        st.header("📊 Risk Seviyesi")
        risk_placeholder = st.empty()
        # İlk render — boş durum
        if "last_risk" not in st.session_state:
            risk_placeholder.info("Henüz değerlendirme yapılmadı.")
        else:
            render_risk_card(st.session_state.last_risk, container=risk_placeholder)

        st.divider()

        st.header("ℹ️ Hedef Kategoriler")
        st.markdown(
            "- Solunum yolu hastalıkları\n"
            "- Sindirim sistemi şikayetleri\n"
            "- Genel ağrı sendromları"
        )

        st.divider()

        if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("last_risk", None)
            st.session_state.pop("last_sources", None)
            st.rerun()

        st.caption(
            "Geliştirici: [Selman Kılıçkaya](https://github.com/selmankilickaya/HealthChatbot)"
        )

    return UserContext(
        age=age,
        is_pregnant=is_pregnant,
        has_chronic_disease=has_chronic,
        is_immunocompromised=is_immuno,
        is_infant=is_infant,
    )


# ---------------------------------------------------------------------------
# Risk kartı çizici
# ---------------------------------------------------------------------------
def render_risk_card(assessment, container=None):
    """Risk seviyesine göre renkli kart göster."""
    target = container if container is not None else st
    level = assessment.level

    if level == RiskLevel.URGENT:
        target.error(
            f"### {level.emoji} {level.value.upper()}\n\n"
            f"**Öneri:** Derhal **112** veya en yakın acil servise başvurun."
        )
    elif level == RiskLevel.MEDIUM:
        target.warning(
            f"### {level.emoji} {level.value.upper()}\n\n"
            f"**Öneri:** En kısa sürede aile hekiminize başvurun."
        )
    else:
        target.success(
            f"### {level.emoji} {level.value.upper()}\n\n"
            f"**Öneri:** Evde takip edebilirsiniz, geçmezse hekime başvurun."
        )

    if assessment.red_flags:
        target.markdown("**🚨 Tespit edilen kritik belirtiler:**")
        for rf in assessment.red_flags:
            target.markdown(f"- {rf}")

    if assessment.risk_group_uplift:
        target.caption(
            "ℹ️ Risk grubunda olduğunuz için risk seviyesi yükseltildi."
        )


# ---------------------------------------------------------------------------
# Ana uygulama
# ---------------------------------------------------------------------------
def main():
    st.title("🩺 Türkçe Semptom Değerlendirici")
    st.caption("LLM + RAG tabanlı sağlık karar destek sistemi")
    st.warning(DISCLAIMER)

    # Sistem hazır mı?
    ready, error_msg = check_system_ready()
    if not ready:
        st.error(error_msg)
        st.stop()

    # Yan panel + kullanıcı profili
    user_context = render_sidebar()

    # Bileşenleri başlat (cache'li)
    rag = get_rag_engine()
    chatbot = get_chatbot(rag)
    scorer = get_risk_scorer(chatbot)

    # Sohbet geçmişi
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Karşılama mesajı
        st.session_state.messages.append({
            "role": "assistant",
            "content": WELCOME_MESSAGE,
        })

    # Geçmişi render et
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Asistan mesajının altında kaynaklar (varsa)
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Kullanılan kaynaklar"):
                    for src in msg["sources"]:
                        st.markdown(f"- `{src}`")

    # Yeni mesaj girişi
    if prompt := st.chat_input("Şikayetinizi yazınız..."):
        # Kullanıcı mesajını göster ve kaydet
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Asistan cevabı
        with st.chat_message("assistant"):
            try:
                with st.spinner("Düşünüyor..."):
                    # Geçmişi chatbot için hazırla (sadece user/assistant rolleri)
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]  # son mesaj hariç
                        if m["role"] in ("user", "assistant")
                    ]

                    # 1) Cevap üret
                    answer, sources = chatbot.respond(prompt, history=history)

                    # 2) Risk skoru hesapla — son birkaç kullanıcı mesajını birleştir
                    user_messages_text = "\n".join(
                        m["content"]
                        for m in st.session_state.messages
                        if m["role"] == "user"
                    )
                    rag_context_text = "\n\n".join(
                        s.page_content[:500] for s in sources
                    )
                    risk = scorer.score(
                        user_text=user_messages_text,
                        rag_context=rag_context_text,
                        user_context=user_context,
                    )

                # Cevabı göster
                st.markdown(answer)

                # Kaynaklar
                source_names = list({s.metadata.get("source_file", "?") for s in sources})
                if source_names:
                    with st.expander("📎 Kullanılan kaynaklar"):
                        for src in source_names:
                            st.markdown(f"- `{src}`")

                # Mesajı geçmişe kaydet
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_names,
                })

                # Risk durumunu sakla — yan panel rerun'da görür
                st.session_state.last_risk = risk
                st.session_state.last_sources = source_names

                # Yan paneli güncellemek için rerun
                st.rerun()

            except Exception as e:
                error_text = (
                    f"Bir hata oluştu: `{type(e).__name__}: {e}`\n\n"
                    "Lütfen tekrar deneyin. Sorun devam ederse API anahtarınızı "
                    "ve internet bağlantınızı kontrol edin."
                )
                st.error(error_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_text,
                })


if __name__ == "__main__":
    main()
