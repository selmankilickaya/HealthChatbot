"""
run_evaluation.py
Sistemin altın standart veri seti üzerinde değerlendirilmesi.

İki konfigürasyon karşılaştırılır:
    A) RULE_ONLY: Sadece kural tabanlı + risk grubu (LLM yok, ücretsiz)
    B) HYBRID:    Kural + LLM + risk grubu (tam sistem)

Çıktılar:
    - Konsola özet rapor
    - evaluation/evaluation_report.md (Markdown rapor)
    - evaluation/results.csv (her örneğin sonucu)

Çalıştırma (proje kökünden):
    python evaluation/run_evaluation.py            # her ikisi
    python evaluation/run_evaluation.py --rule-only  # sadece kural
"""
from __future__ import annotations
from pathlib import Path
import argparse
import csv
import sys
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Optional

# Proje köküne erişim
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.risk_scorer import RiskScorer, RiskLevel, UserContext


EVAL_DIR = ROOT / "evaluation"
TEST_SET_PATH = EVAL_DIR / "symptom_test_set.csv"


# ---------------------------------------------------------------------------
# Veri yapıları
# ---------------------------------------------------------------------------
@dataclass
class TestCase:
    id: int
    category: str
    text: str
    expected: RiskLevel
    risk_group: str
    description: str

    def to_user_context(self) -> UserContext:
        """CSV'deki risk_grubu sütununa göre UserContext oluşturur."""
        rg = self.risk_group.strip().lower()
        if rg in ("hayır", "yok", "no", ""):
            return UserContext()
        if "yaşlı" in rg or "65" in rg:
            return UserContext(age=70)
        if "bebek" in rg:
            return UserContext(is_infant=True, age=0)
        if "çocuk" in rg:
            return UserContext(age=3)
        if "gebe" in rg:
            return UserContext(is_pregnant=True, age=28)
        if "bağışıklık" in rg or "immun" in rg:
            return UserContext(is_immunocompromised=True)
        if "kronik" in rg:
            return UserContext(has_chronic_disease=True)
        return UserContext()


@dataclass
class EvalResult:
    case: TestCase
    predicted: RiskLevel
    correct: bool
    red_flags: List[str]
    reasoning: str
    confidence: float
    uplifted: bool


# ---------------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------------
def load_test_set(path: Path = TEST_SET_PATH) -> List[TestCase]:
    """CSV'den altın standart veri setini oku."""
    cases = []
    risk_map = {
        "Düşük": RiskLevel.LOW,
        "Orta": RiskLevel.MEDIUM,
        "Acil": RiskLevel.URGENT,
    }
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expected = risk_map.get(row["beklenen_risk"].strip())
            if expected is None:
                print(f"⚠️  Geçersiz beklenen_risk: {row['beklenen_risk']}")
                continue
            cases.append(TestCase(
                id=int(row["id"]),
                category=row["kategori"].strip(),
                text=row["semptom_metni"].strip(),
                expected=expected,
                risk_group=row["risk_grubu"].strip(),
                description=row["aciklama"].strip(),
            ))
    return cases


# ---------------------------------------------------------------------------
# Değerlendirme
# ---------------------------------------------------------------------------
def evaluate(scorer: RiskScorer, cases: List[TestCase]) -> List[EvalResult]:
    """Tüm test vakalarını skorla, sonuç listesi döndür."""
    results = []
    for case in cases:
        ctx = case.to_user_context()
        assessment = scorer.score(case.text, user_context=ctx)
        results.append(EvalResult(
            case=case,
            predicted=assessment.level,
            correct=(assessment.level == case.expected),
            red_flags=assessment.red_flags,
            reasoning=assessment.reasoning,
            confidence=assessment.confidence,
            uplifted=assessment.risk_group_uplift,
        ))
    return results


# ---------------------------------------------------------------------------
# Metrikler
# ---------------------------------------------------------------------------
def compute_metrics(results: List[EvalResult]) -> dict:
    """Doğruluk, sınıf bazlı precision/recall, false negative rate hesapla."""
    n = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / n if n > 0 else 0.0

    # Karmaşıklık matrisi: confusion[gerçek][tahmin] = sayı
    levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.URGENT]
    confusion: dict = {a: {b: 0 for b in levels} for a in levels}
    for r in results:
        confusion[r.case.expected][r.predicted] += 1

    # Sınıf bazlı metrikler
    per_class = {}
    for level in levels:
        tp = confusion[level][level]
        fp = sum(confusion[other][level] for other in levels if other != level)
        fn = sum(confusion[level][other] for other in levels if other != level)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class[level] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "support": tp + fn,
        }

    # En kritik metrik: ACİL vakaların kaçırılma oranı
    urgent_total = per_class[RiskLevel.URGENT]["support"]
    urgent_caught = per_class[RiskLevel.URGENT]["tp"]
    urgent_missed = urgent_total - urgent_caught
    urgent_miss_rate = urgent_missed / urgent_total if urgent_total > 0 else 0.0

    return {
        "n": n,
        "correct": correct,
        "accuracy": accuracy,
        "confusion": confusion,
        "per_class": per_class,
        "urgent_total": urgent_total,
        "urgent_caught": urgent_caught,
        "urgent_missed": urgent_missed,
        "urgent_miss_rate": urgent_miss_rate,
    }


# ---------------------------------------------------------------------------
# Konsol raporu
# ---------------------------------------------------------------------------
def print_report(name: str, metrics: dict, results: List[EvalResult]):
    print("\n" + "=" * 68)
    print(f"  KONFİGÜRASYON: {name}")
    print("=" * 68)

    print(f"\n📊 Genel Doğruluk: {metrics['correct']}/{metrics['n']}  "
          f"({metrics['accuracy'] * 100:.1f}%)")

    # Karmaşıklık matrisi
    print("\n📋 Karmaşıklık Matrisi:")
    print(f"{'':16s}{'Tahmin →':<32s}")
    print(f"{'Gerçek ↓':16s}{'Düşük':>10s}{'Orta':>10s}{'Acil':>10s}")
    print(f"{'-' * 46}")
    for level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.URGENT]:
        row = metrics["confusion"][level]
        print(f"  {level.emoji} {level.value:12s}"
              f"{row[RiskLevel.LOW]:>10d}"
              f"{row[RiskLevel.MEDIUM]:>10d}"
              f"{row[RiskLevel.URGENT]:>10d}")

    # Sınıf bazlı performans
    print("\n📈 Sınıf Bazlı Performans:")
    for level, m in metrics["per_class"].items():
        if m["support"] == 0:
            continue
        print(f"  {level.emoji} {level.value:8s} "
              f"P={m['precision'] * 100:5.1f}%  "
              f"R={m['recall'] * 100:5.1f}%  "
              f"(n={m['support']})")

    # Kritik metrik
    print(f"\n🚨 ACİL Vakaları Kaçırma Oranı (False Negative Rate):")
    print(f"   {metrics['urgent_missed']}/{metrics['urgent_total']} vaka kaçırıldı  "
          f"({metrics['urgent_miss_rate'] * 100:.1f}%)")
    if metrics['urgent_missed'] == 0:
        print(f"   ✅ Mükemmel: Hiçbir acil vaka atlanmadı!")
    elif metrics['urgent_miss_rate'] < 0.1:
        print(f"   ✓ İyi: %10'un altında")
    else:
        print(f"   ⚠️  Yüksek: gözden geçirilmeli")

    # Yanlış sınıflandırılan örnekler
    wrong = [r for r in results if not r.correct]
    if wrong:
        print(f"\n❌ Yanlış Sınıflandırılan Örnekler ({len(wrong)}):")
        for r in wrong[:5]:  # ilk 5'i göster
            print(f"   #{r.case.id} [{r.case.category}] "
                  f"Beklenen: {r.case.expected.emoji} "
                  f"→ Tahmin: {r.predicted.emoji}")
            print(f"      \"{r.case.text[:80]}...\"")


# ---------------------------------------------------------------------------
# Markdown rapor
# ---------------------------------------------------------------------------
def write_markdown_report(
    output_path: Path,
    rule_metrics: dict,
    hybrid_metrics: Optional[dict],
    rule_results: List[EvalResult],
    hybrid_results: Optional[List[EvalResult]],
):
    """Akademik rapor formatında değerlendirme sonuçlarını yaz."""
    lines = []
    lines.append("# Değerlendirme Raporu")
    lines.append("")
    lines.append("Türkçe Semptom Değerlendirici sisteminin altın standart "
                 "veri seti üzerindeki performans değerlendirmesi.")
    lines.append("")
    lines.append("## Test Seti Özeti")
    lines.append("")
    lines.append(f"- **Toplam örnek:** {rule_metrics['n']}")
    lines.append(f"- **Düşük risk:** {rule_metrics['per_class'][RiskLevel.LOW]['support']} örnek")
    lines.append(f"- **Orta risk:** {rule_metrics['per_class'][RiskLevel.MEDIUM]['support']} örnek")
    lines.append(f"- **Acil risk:** {rule_metrics['per_class'][RiskLevel.URGENT]['support']} örnek")
    lines.append("")

    # Tablo başlığı
    lines.append("## Performans Karşılaştırması")
    lines.append("")
    if hybrid_metrics:
        lines.append("| Metrik | Sadece Kural | Hibrit (Kural + LLM) |")
        lines.append("|---|---|---|")
        lines.append(f"| Genel Doğruluk | "
                     f"{rule_metrics['accuracy']*100:.1f}% | "
                     f"{hybrid_metrics['accuracy']*100:.1f}% |")
        for lvl in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.URGENT]:
            r_p = rule_metrics['per_class'][lvl]['precision']
            r_r = rule_metrics['per_class'][lvl]['recall']
            h_p = hybrid_metrics['per_class'][lvl]['precision']
            h_r = hybrid_metrics['per_class'][lvl]['recall']
            lines.append(f"| {lvl.value} P/R | "
                         f"{r_p*100:.0f}% / {r_r*100:.0f}% | "
                         f"{h_p*100:.0f}% / {h_r*100:.0f}% |")
        lines.append(f"| Acil Kaçırma | "
                     f"{rule_metrics['urgent_missed']}/{rule_metrics['urgent_total']} | "
                     f"{hybrid_metrics['urgent_missed']}/{hybrid_metrics['urgent_total']} |")
    else:
        lines.append("| Metrik | Sadece Kural |")
        lines.append("|---|---|")
        lines.append(f"| Genel Doğruluk | {rule_metrics['accuracy']*100:.1f}% |")
        lines.append(f"| Acil Kaçırma | "
                     f"{rule_metrics['urgent_missed']}/{rule_metrics['urgent_total']} "
                     f"({rule_metrics['urgent_miss_rate']*100:.1f}%) |")
    lines.append("")

    lines.append("## Sonuç ve Tartışma")
    lines.append("")
    if hybrid_metrics:
        improvement = (hybrid_metrics['accuracy'] - rule_metrics['accuracy']) * 100
        lines.append(f"Hibrit sistem, sadece kural tabanlı yaklaşıma göre "
                     f"**{improvement:+.1f} puanlık** doğruluk artışı sağlamıştır. ")
        if hybrid_metrics['urgent_missed'] == 0:
            lines.append("Hibrit konfigürasyonda **hiçbir acil vaka atlanmamış**, "
                         "bu da kırmızı bayrak protokolünün etkinliğini doğrulamaktadır.")
    else:
        lines.append("Sadece kural tabanlı sistem, herhangi bir LLM çağrısı "
                     "olmaksızın deterministik kırmızı bayrak taraması ve risk "
                     "grubu yükseltmesi ile çalışmaktadır.")
    lines.append("")
    lines.append("> Not: Bu değerlendirme örnek bir altın standart üzerinde "
                 "yapılmıştır. Saha testlerinde hekim onaylı geniş bir veri "
                 "seti ile yeniden ölçülmesi planlanmaktadır.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📝 Markdown rapor yazıldı: {output_path}")


# ---------------------------------------------------------------------------
# CSV detay raporu
# ---------------------------------------------------------------------------
def write_results_csv(output_path: Path, results: List[EvalResult], config_name: str):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "kategori", "metin", "beklenen", "tahmin",
            "dogru", "kirmizi_bayraklar", "guven", "yukseltildi", "konfigurasyon",
        ])
        for r in results:
            writer.writerow([
                r.case.id,
                r.case.category,
                r.case.text,
                r.case.expected.value,
                r.predicted.value,
                "Evet" if r.correct else "Hayır",
                "; ".join(r.red_flags),
                f"{r.confidence:.2f}",
                "Evet" if r.uplifted else "Hayır",
                config_name,
            ])
    print(f"📝 Detay CSV yazıldı: {output_path}")


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Risk skorlayıcı değerlendirme")
    parser.add_argument(
        "--rule-only", action="store_true",
        help="Sadece kural tabanlı modu çalıştır (LLM kullanma)",
    )
    args = parser.parse_args()

    # Test setini yükle
    cases = load_test_set()
    print(f"📂 Test seti yüklendi: {len(cases)} örnek")

    # A) Sadece kural tabanlı
    print("\n→ A) RULE_ONLY konfigürasyonu çalıştırılıyor...")
    rule_scorer = RiskScorer(llm=None)
    rule_results = evaluate(rule_scorer, cases)
    rule_metrics = compute_metrics(rule_results)
    print_report("RULE_ONLY (Sadece Kural Tabanlı)", rule_metrics, rule_results)
    write_results_csv(EVAL_DIR / "results_rule_only.csv", rule_results, "RULE_ONLY")

    hybrid_metrics = None
    hybrid_results = None

    # B) Hibrit (LLM dahil) — opsiyonel
    if not args.rule_only:
        print("\n→ B) HYBRID konfigürasyonu çalıştırılıyor (Claude API kullanır)...")
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from app.chatbot import HealthChatbot
            from app.rag_engine import RAGEngine

            rag = RAGEngine(verbose=False)
            if not rag.is_indexed():
                print("⚠️  ChromaDB boş — önce: python scripts/build_index.py")
                print("   Hibrit değerlendirme atlanıyor.")
            else:
                bot = HealthChatbot(rag_engine=rag)
                hybrid_scorer = RiskScorer(llm=bot.llm)
                hybrid_results = evaluate(hybrid_scorer, cases)
                hybrid_metrics = compute_metrics(hybrid_results)
                print_report("HYBRID (Kural + LLM)", hybrid_metrics, hybrid_results)
                write_results_csv(
                    EVAL_DIR / "results_hybrid.csv", hybrid_results, "HYBRID"
                )
        except Exception as e:
            print(f"\n⚠️  Hibrit değerlendirme hatası: {e}")
            print("   Sadece kural tabanlı sonuç kaydedildi.")

    # Markdown rapor
    write_markdown_report(
        EVAL_DIR / "evaluation_report.md",
        rule_metrics, hybrid_metrics,
        rule_results, hybrid_results,
    )

    print("\n" + "=" * 68)
    print("  ✅ Değerlendirme tamamlandı.")
    print("=" * 68)


if __name__ == "__main__":
    main()
