"""
A3. Performans Qiymətləndirməsi

WER (Word Error Rate) və CER (Character Error Rate) hesablayır.
Ən yaxşı və ən pis 5 nümunəni göstərir.
Nəticələri CSV və JSON formatında saxlayır.
"""

import os
import json
import csv
from jiwer import wer, cer
import numpy as np


def normalize_text(text):
    """
    Azərbaycan dili üçün mətn normallaşdırması.
    WER/CER hesablamasından əvvəl tətbiq olunur.

    Args:
        text: Xam mətn

    Returns:
        str: Normallaşdırılmış mətn
    """
    if text is None:
        return ""

    text = text.strip().lower()

    # Əlavə boşluqları sil
    text = " ".join(text.split())

    # Azərbaycan dilinə xas olmayan xüsusi simvolları sil
    # (durğu işarələri saxlanılır, çünki WER-ə təsir edə bilər)
    chars_to_remove = ["«", "»", "„", """, """, "–", "—", "…"]
    for char in chars_to_remove:
        text = text.replace(char, "")

    # Durğu işarələrini sil (standart ASR qiymətləndirmə)
    import re
    text = re.sub(r'[^\w\s]', '', text)

    # Əlavə boşluqları yenidən sil
    text = " ".join(text.split())

    return text


def calculate_metrics(results):
    """
    Bütün nümunələr üçün WER və CER hesablayır.

    Args:
        results: list[dict] - hər dict-də "reference" və "prediction" olmalıdır

    Returns:
        dict: Metrikalar və detallı nəticələr
    """
    print("\n[*] WER və CER hesablanır...")

    detailed_results = []

    for r in results:
        ref = normalize_text(r["reference"])
        pred = normalize_text(r["prediction"])

        if len(ref) == 0:
            continue

        # Fərdi WER/CER
        try:
            sample_wer = wer(ref, pred)
        except Exception:
            sample_wer = 1.0

        try:
            sample_cer = cer(ref, pred)
        except Exception:
            sample_cer = 1.0

        detailed_results.append({
            "index": r["index"],
            "reference": r["reference"],
            "prediction": r["prediction"],
            "normalized_reference": ref,
            "normalized_prediction": pred,
            "wer": sample_wer,
            "cer": sample_cer,
            "audio_length": r.get("audio_length", 0),
        })

    # Ortalama metrikalar
    wer_scores = [d["wer"] for d in detailed_results]
    cer_scores = [d["cer"] for d in detailed_results]

    avg_wer = np.mean(wer_scores) * 100
    avg_cer = np.mean(cer_scores) * 100
    median_wer = np.median(wer_scores) * 100
    median_cer = np.median(cer_scores) * 100

    # WER-ə görə sıralama
    sorted_by_wer = sorted(detailed_results, key=lambda x: x["wer"])
    best_5 = sorted_by_wer[:5]
    worst_5 = sorted_by_wer[-5:][::-1]

    metrics = {
        "avg_wer": round(avg_wer, 2),
        "avg_cer": round(avg_cer, 2),
        "median_wer": round(median_wer, 2),
        "median_cer": round(median_cer, 2),
        "total_samples": len(detailed_results),
        "best_5": best_5,
        "worst_5": worst_5,
        "all_results": detailed_results,
    }

    return metrics


def print_metrics(metrics):
    """
    Metrikaları formatlanmış şəkildə çap edir.

    Args:
        metrics: calculate_metrics()-dən qaytarılan dict
    """
    print("\n" + "=" * 70)
    print("                    ASR PERFORMANS NƏTİCƏLƏRİ")
    print("=" * 70)

    print(f"\n📊 Ümumi Statistika ({metrics['total_samples']} nümunə):")
    print(f"   ├── Ortalama WER:  {metrics['avg_wer']:.2f}%")
    print(f"   ├── Ortalama CER:  {metrics['avg_cer']:.2f}%")
    print(f"   ├── Median WER:    {metrics['median_wer']:.2f}%")
    print(f"   └── Median CER:    {metrics['median_cer']:.2f}%")

    print(f"\n🏆 Ən Yaxşı 5 Nümunə (Ən aşağı WER):")
    print("-" * 70)
    for i, sample in enumerate(metrics["best_5"], 1):
        print(f"  {i}. WER={sample['wer']*100:.1f}% | CER={sample['cer']*100:.1f}%")
        print(f"     Ref:  {sample['reference']}")
        print(f"     Pred: {sample['prediction']}")
        print()

    print(f"\n❌ Ən Pis 5 Nümunə (Ən yüksək WER):")
    print("-" * 70)
    for i, sample in enumerate(metrics["worst_5"], 1):
        print(f"  {i}. WER={sample['wer']*100:.1f}% | CER={sample['cer']*100:.1f}%")
        print(f"     Ref:  {sample['reference']}")
        print(f"     Pred: {sample['prediction']}")
        print()


def save_results(metrics, output_dir="../results"):
    """
    Nəticələri CSV və JSON formatında saxlayır.

    Args:
        metrics: calculate_metrics()-dən qaytarılan dict
        output_dir: Nəticələrin saxlanacağı qovluq
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- JSON ---
    json_path = os.path.join(output_dir, "base_model_results.json")
    save_data = {
        "model": "openai/whisper-small",
        "avg_wer": metrics["avg_wer"],
        "avg_cer": metrics["avg_cer"],
        "median_wer": metrics["median_wer"],
        "median_cer": metrics["median_cer"],
        "total_samples": metrics["total_samples"],
        "best_5": [
            {
                "index": s["index"],
                "reference": s["reference"],
                "prediction": s["prediction"],
                "wer": round(s["wer"] * 100, 2),
                "cer": round(s["cer"] * 100, 2),
            }
            for s in metrics["best_5"]
        ],
        "worst_5": [
            {
                "index": s["index"],
                "reference": s["reference"],
                "prediction": s["prediction"],
                "wer": round(s["wer"] * 100, 2),
                "cer": round(s["cer"] * 100, 2),
            }
            for s in metrics["worst_5"]
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"[✓] JSON nəticələr saxlandı: {json_path}")

    # --- CSV (bütün nümunələr) ---
    csv_path = os.path.join(output_dir, "base_model_all_samples.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "reference", "prediction", "wer", "cer", "audio_length"],
        )
        writer.writeheader()
        for row in metrics["all_results"]:
            writer.writerow({
                "index": row["index"],
                "reference": row["reference"],
                "prediction": row["prediction"],
                "wer": round(row["wer"] * 100, 2),
                "cer": round(row["cer"] * 100, 2),
                "audio_length": round(row["audio_length"], 2),
            })
    print(f"[✓] CSV nəticələr saxlandı: {csv_path}")

    # --- Cədvəl (Markdown) ---
    md_path = os.path.join(output_dir, "base_model_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ASR Baza Model Nəticələri\n\n")
        f.write(f"**Model:** openai/whisper-small\n\n")
        f.write("## Ümumi Metrikalar\n\n")
        f.write("| Metrika | Dəyər |\n")
        f.write("|---------|-------|\n")
        f.write(f"| Ortalama WER | {metrics['avg_wer']:.2f}% |\n")
        f.write(f"| Ortalama CER | {metrics['avg_cer']:.2f}% |\n")
        f.write(f"| Median WER | {metrics['median_wer']:.2f}% |\n")
        f.write(f"| Median CER | {metrics['median_cer']:.2f}% |\n")
        f.write(f"| Nümunə sayı | {metrics['total_samples']} |\n\n")

        f.write("## Ən Yaxşı 5 Nümunə\n\n")
        f.write("| # | WER | CER | Reference | Prediction |\n")
        f.write("|---|-----|-----|-----------|------------|\n")
        for i, s in enumerate(metrics["best_5"], 1):
            f.write(
                f"| {i} | {s['wer']*100:.1f}% | {s['cer']*100:.1f}% | "
                f"{s['reference'][:50]} | {s['prediction'][:50]} |\n"
            )

        f.write("\n## Ən Pis 5 Nümunə\n\n")
        f.write("| # | WER | CER | Reference | Prediction |\n")
        f.write("|---|-----|-----|-----------|------------|\n")
        for i, s in enumerate(metrics["worst_5"], 1):
            f.write(
                f"| {i} | {s['wer']*100:.1f}% | {s['cer']*100:.1f}% | "
                f"{s['reference'][:50]} | {s['prediction'][:50]} |\n"
            )

    print(f"[✓] Markdown cədvəl saxlandı: {md_path}")


if __name__ == "__main__":
    # Test üçün dummy data
    test_results = [
        {"index": 0, "reference": "salam dünya", "prediction": "salam dünya", "audio_length": 2.0},
        {"index": 1, "reference": "bu bir testdir", "prediction": "bu bir təstdir", "audio_length": 1.5},
    ]
    metrics = calculate_metrics(test_results)
    print_metrics(metrics)
