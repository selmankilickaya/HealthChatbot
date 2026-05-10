"""
Türkçe Semptom Değerlendirici — Streamlit Arayüzü
Ana giriş noktası. `streamlit run streamlit_app.py` ile çalıştırılır.
"""
import streamlit as st

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Türkçe Semptom Değerlendirici",
    page_icon="🩺",
    layout="centered",
)

# Sorumluluk reddi (her oturumda gösterilir)
DISCLAIMER = """
⚠️ **Önemli Uyarı:** Bu uygulama bir tanı aracı **değildir**.
Sadece bilgilendirme ve yönlendirme amaçlıdır. Acil durumlarda **112**'yi arayın.
Şikayetleriniz için bir sağlık kuruluşuna başvurmanız önerilir.
"""


def main():
    st.title("🩺 Türkçe Semptom Değerlendirici")
    st.caption("LLM + RAG tabanlı sağlık karar destek sistemi (Prototip)")

    st.warning(DISCLAIMER)

    # Sohbet geçmişi
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Geçmişi göster
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Kullanıcı girişi
    if prompt := st.chat_input("Şikayetinizi yazınız..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # TODO: Buraya RAG + LLM cevabı gelecek (sonraki adımlarda)
        with st.chat_message("assistant"):
            placeholder_response = (
                "🚧 Sistem henüz geliştirme aşamasında.\n\n"
                f"Mesajınızı aldım: *\"{prompt}\"*\n\n"
                "RAG + LLM motoru bağlandığında burada gerçek cevap olacak."
            )
            st.markdown(placeholder_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": placeholder_response}
            )

    # Yan panel — risk göstergesi (placeholder)
    with st.sidebar:
        st.header("Risk Seviyesi")
        st.info("Henüz değerlendirme yapılmadı.")

        st.divider()
        st.caption("**Hedef Kategoriler**")
        st.markdown(
            "- Solunum yolu hastalıkları\n"
            "- Sindirim sistemi şikayetleri\n"
            "- Genel ağrı sendromları"
        )

        st.divider()
        if st.button("🗑️ Sohbeti Temizle"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
