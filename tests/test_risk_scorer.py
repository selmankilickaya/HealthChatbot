"""
test_risk_scorer.py
RiskScorer modülü için birim testleri.

Çalıştırma:
    pytest tests/test_risk_scorer.py -v
"""
import pytest
from app.risk_scorer import (
    RiskScorer, RiskLevel, UserContext, RiskAssessment, normalize_text,
)


# ---------------------------------------------------------------------------
# Yardımcı fonksiyon testleri
# ---------------------------------------------------------------------------
class TestNormalizeText:
    """Türkçe-aware küçük harf dönüşümü."""

    def test_buyuk_i_donusumu(self):
        assert normalize_text("İNTİHAR") == "intihar"

    def test_dotted_i(self):
        assert normalize_text("İlk yardım") == "ilk yardım"

    def test_zaten_kucuk(self):
        assert normalize_text("öksürük") == "öksürük"

    def test_karisik_metin(self):
        # 'I' (dotless) → 'ı' kuralı
        result = normalize_text("ATEŞIM 38 DERECE")
        assert "ateş" in result
        assert "derece" in result


# ---------------------------------------------------------------------------
# RiskLevel enum testleri
# ---------------------------------------------------------------------------
class TestRiskLevel:
    def test_emoji_eslesmesi(self):
        assert RiskLevel.LOW.emoji == "🟢"
        assert RiskLevel.MEDIUM.emoji == "🟡"
        assert RiskLevel.URGENT.emoji == "🔴"

    def test_numeric_siralama(self):
        assert RiskLevel.LOW.numeric < RiskLevel.MEDIUM.numeric
        assert RiskLevel.MEDIUM.numeric < RiskLevel.URGENT.numeric

    def test_numeric_donusum(self):
        assert RiskLevel.from_numeric(1) == RiskLevel.LOW
        assert RiskLevel.from_numeric(2) == RiskLevel.MEDIUM
        assert RiskLevel.from_numeric(3) == RiskLevel.URGENT

    def test_numeric_sinir_disi(self):
        assert RiskLevel.from_numeric(0) == RiskLevel.LOW
        assert RiskLevel.from_numeric(99) == RiskLevel.URGENT


# ---------------------------------------------------------------------------
# UserContext testleri
# ---------------------------------------------------------------------------
class TestUserContext:
    def test_bos_bagilam_risk_grubu_degil(self):
        ctx = UserContext()
        is_risk, _ = ctx.is_risk_group()
        assert is_risk is False

    def test_yasli_risk_grubu(self):
        ctx = UserContext(age=70)
        is_risk, reason = ctx.is_risk_group()
        assert is_risk is True
        assert "65 yaş" in reason

    def test_bebek_risk_grubu(self):
        ctx = UserContext(is_infant=True)
        is_risk, reason = ctx.is_risk_group()
        assert is_risk is True
        assert "bebek" in reason

    def test_gebelik_risk_grubu(self):
        ctx = UserContext(age=30, is_pregnant=True)
        is_risk, reason = ctx.is_risk_group()
        assert is_risk is True
        assert "gebelik" in reason

    def test_coklu_risk_faktoru(self):
        ctx = UserContext(age=70, has_chronic_disease=True)
        is_risk, reason = ctx.is_risk_group()
        assert is_risk is True
        assert "65 yaş" in reason
        assert "kronik" in reason


# ---------------------------------------------------------------------------
# Kural tabanlı kırmızı bayrak taraması
# ---------------------------------------------------------------------------
class TestRedFlagDetection:
    """LLM gerektirmez - tamamen deterministik."""

    @pytest.fixture
    def scorer(self):
        return RiskScorer(llm=None)

    def test_kalp_belirtileri(self, scorer):
        flags = scorer.scan_red_flags(
            "Göğsümde basınç var, sol koluma vuruyor"
        )
        assert len(flags) > 0
        assert any("kalp" in f.lower() for f in flags)

    def test_solunum_yetmezligi(self, scorer):
        flags = scorer.scan_red_flags("Nefes alamıyorum, dudaklarım moraryor")
        assert len(flags) > 0
        assert any("solunum" in f.lower() for f in flags)

    def test_norolojik_acil(self, scorer):
        flags = scorer.scan_red_flags(
            "Hayatımın en şiddetli baş ağrısı, ani başladı"
        )
        assert len(flags) > 0
        assert any("nörolojik" in f.lower() for f in flags)

    def test_ic_kanama(self, scorer):
        flags = scorer.scan_red_flags("Kanlı kustum, midem çok kötü")
        assert len(flags) > 0
        assert any("kanama" in f.lower() for f in flags)

    def test_akut_karin(self, scorer):
        flags = scorer.scan_red_flags(
            "Karnım çok şiddetli ağrıyor, dokundurmuyorum"
        )
        assert len(flags) > 0

    def test_anafilaksi(self, scorer):
        flags = scorer.scan_red_flags("Yüzüm ve dudaklarım şişti")
        assert len(flags) > 0
        assert any("anafilak" in f.lower() for f in flags)

    def test_intihar(self, scorer):
        flags = scorer.scan_red_flags("İntihar etmek istiyorum")
        assert len(flags) > 0

    def test_intihar_buyuk_harf_duyarsiz(self, scorer):
        """Türkçe büyük 'İ' → 'i' düzgün dönüşmeli."""
        flags = scorer.scan_red_flags("İNTİHAR ETMEK İSTİYORUM")
        assert len(flags) > 0

    def test_hafif_belirti_kirmizi_bayrak_yok(self, scorer):
        flags = scorer.scan_red_flags("Hafif öksürüğüm var")
        assert len(flags) == 0

    def test_bos_metin(self, scorer):
        flags = scorer.scan_red_flags("")
        assert flags == []

    def test_coklu_kategori_eslesmesi(self, scorer):
        """Anafilaksi + solunum yetmezliği aynı anda yakalanmalı."""
        flags = scorer.scan_red_flags(
            "Yüzüm şişti, nefes alamıyorum"
        )
        assert len(flags) >= 2


# ---------------------------------------------------------------------------
# Risk grubu yükseltmesi
# ---------------------------------------------------------------------------
class TestRiskGroupUplift:
    @pytest.fixture
    def scorer(self):
        return RiskScorer(llm=None)

    def test_dusuk_orta_yukseltme(self, scorer):
        elderly = UserContext(age=70)
        new_level, uplifted, _ = scorer.apply_risk_group_uplift(
            RiskLevel.LOW, elderly
        )
        assert new_level == RiskLevel.MEDIUM
        assert uplifted is True

    def test_orta_acil_yukseltme(self, scorer):
        ctx = UserContext(is_pregnant=True)
        new_level, uplifted, _ = scorer.apply_risk_group_uplift(
            RiskLevel.MEDIUM, ctx
        )
        assert new_level == RiskLevel.URGENT
        assert uplifted is True

    def test_acil_yukselemez(self, scorer):
        """Acil zaten en üst seviye, daha yukarı çıkmaz."""
        ctx = UserContext(age=70)
        new_level, uplifted, _ = scorer.apply_risk_group_uplift(
            RiskLevel.URGENT, ctx
        )
        assert new_level == RiskLevel.URGENT
        assert uplifted is False

    def test_risk_grubu_yoksa_yukseltme_yok(self, scorer):
        normal = UserContext(age=30)
        new_level, uplifted, _ = scorer.apply_risk_group_uplift(
            RiskLevel.LOW, normal
        )
        assert new_level == RiskLevel.LOW
        assert uplifted is False


# ---------------------------------------------------------------------------
# Tümleşik skorlama (LLM'siz)
# ---------------------------------------------------------------------------
class TestIntegratedScoring:
    @pytest.fixture
    def scorer(self):
        return RiskScorer(llm=None)

    def test_kirmizi_bayrak_anlik_acil(self, scorer):
        """Kırmızı bayrak varsa LLM beklenmeden ACİL döner."""
        result = scorer.score("Göğsüm sıkışıyor, sol koluma vuruyor")
        assert result.level == RiskLevel.URGENT
        assert len(result.red_flags) > 0
        assert result.confidence == 1.0  # deterministik

    def test_yasli_hafif_belirti_orta(self, scorer):
        """Yaşlı + hafif belirti → LLM'siz default DÜŞÜK, ORTA'ya yükselir."""
        elderly = UserContext(age=70, has_chronic_disease=True)
        result = scorer.score("Hafif öksürüğüm var", user_context=elderly)
        # LLM yok, default DÜŞÜK → risk grubu → ORTA
        assert result.level == RiskLevel.MEDIUM
        assert result.risk_group_uplift is True

    def test_assessment_to_dict(self, scorer):
        result = scorer.score("Burnum akıyor")
        d = result.to_dict()
        assert "level" in d
        assert "emoji" in d
        assert "reasoning" in d
        assert "red_flags" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
