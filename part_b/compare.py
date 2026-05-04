"""
B3. Müqayisə — Fine-tuned vs Baza Model

Bu skript fine-tuned modeli baza modeli ilə eyni test nümunələrində
müqayisə edir və nəticələri cədvəl/qrafik şəklində göstərir.

İstifadə:
    python compare.py [--test_samples N] [--fine_tuned_path PATH]
"""

import os
import sys
import json
import argparse
import numpy as np

import torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer
from tqdm import tqdm

# Proyekt kökünü path-a əlavə et
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part_a.evaluate import normalize_text


def load_test_data(max_samples=50):
    """Test datasetini yükləyir."""
    print("[*] Test dataseti yüklənir...")
    dataset = load_dataset(
        "google/fleurs", "az_az",
        split="test", trust_remote_code=True
    )
    if "raw_transcription" in dataset.column_names:
        dataset = dataset.rename_column("raw_transcription", "sentence")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.filter(
        lambda x: x["sentence"] is not None and len(x["sentence"].strip()) > 0
    )
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"[✓] {len(dataset)} test nümunəsi yükləndi")
    return dataset


def transcribe_with_model(model, processor, dataset, language="azerbaijani"):
    """
    Verilmiş model və processor ilə dataseti transkripsiya edir.

    Returns:
        list[dict]: Nəticələr
    """
    device = next(model.parameters()).device
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )

    results = []
    for i in tqdm(range(len(dataset)), desc="Transkripsiya"):
        sample = dataset[i]
        audio = sample["audio"]

        input_features = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225,
            )

        prediction = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        ref = sample["sentence"].strip()
        ref_norm = normalize_text(ref)
        pred_norm = normalize_text(prediction)

        try:
            sample_wer = wer(ref_norm, pred_norm) if ref_norm else 1.0
        except Exception:
            sample_wer = 1.0

        try:
            sample_cer = cer(ref_norm, pred_norm) if ref_norm else 1.0
        except Exception:
            sample_cer = 1.0

        results.append({
            "index": i,
            "reference": ref,
            "prediction": prediction,
            "wer": sample_wer,
            "cer": sample_cer,
        })

    return results


def compare_models(
    base_model_name="openai/whisper-small",
    fine_tuned_path=None,
    test_samples=50,
    results_dir=None,
):
    """
    Baza və fine-tuned modelləri müqayisə edir.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if fine_tuned_path is None:
        fine_tuned_path = os.path.join(project_root, "part_b", "checkpoints", "best_model")
    if results_dir is None:
        results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          MODEL MÜQAYİSƏSİ — Base vs Fine-Tuned            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ─── Test Data ───
    test_ds = load_test_data(max_samples=test_samples)

    # ─── Baza Model ───
    print("\n[*] Baza model yüklənir...")
    base_processor = WhisperProcessor.from_pretrained(base_model_name)
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    base_model.to(device)
    base_model.eval()

    print("[*] Baza model ilə transkripsiya...")
    base_results = transcribe_with_model(base_model, base_processor, test_ds)

    # Yaddaşı boşalt
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ─── Fine-Tuned Model ───
    print(f"\n[*] Fine-tuned model yüklənir: {fine_tuned_path}")
    ft_processor = WhisperProcessor.from_pretrained(fine_tuned_path)
    ft_model = WhisperForConditionalGeneration.from_pretrained(fine_tuned_path)
    ft_model.to(device)
    ft_model.eval()

    print("[*] Fine-tuned model ilə transkripsiya...")
    ft_results = transcribe_with_model(ft_model, ft_processor, test_ds)

    # ─── Müqayisə ───
    base_avg_wer = np.mean([r["wer"] for r in base_results]) * 100
    base_avg_cer = np.mean([r["cer"] for r in base_results]) * 100
    ft_avg_wer = np.mean([r["wer"] for r in ft_results]) * 100
    ft_avg_cer = np.mean([r["cer"] for r in ft_results]) * 100

    wer_diff = ft_avg_wer - base_avg_wer
    cer_diff = ft_avg_cer - base_avg_cer

    print("\n" + "=" * 70)
    print("                      MÜQAYİSƏ NƏTİCƏLƏRİ")
    print("=" * 70)
    print(f"\n{'Metrika':<20} {'Baza Model':>15} {'Fine-Tuned':>15} {'Fərq':>15}")
    print("-" * 65)
    print(f"{'WER (%)':<20} {base_avg_wer:>14.2f}% {ft_avg_wer:>14.2f}% {wer_diff:>+14.2f}%")
    print(f"{'CER (%)':<20} {base_avg_cer:>14.2f}% {ft_avg_cer:>14.2f}% {cer_diff:>+14.2f}%")
    print("-" * 65)

    improvement = "Yaxşılaşma ✓" if wer_diff < 0 else "Pisləşmə ✗"
    print(f"\nNəticə: {improvement} (WER: {abs(wer_diff):.2f}% {'azalma' if wer_diff < 0 else 'artım'})")

    # ─── Nəticələri Saxla ───
    comparison = {
        "base_model": base_model_name,
        "fine_tuned_model": fine_tuned_path,
        "test_samples": test_samples,
        "base_wer": round(base_avg_wer, 2),
        "base_cer": round(base_avg_cer, 2),
        "fine_tuned_wer": round(ft_avg_wer, 2),
        "fine_tuned_cer": round(ft_avg_cer, 2),
        "wer_diff": round(wer_diff, 2),
        "cer_diff": round(cer_diff, 2),
        "per_sample": [],
    }

    for b, f in zip(base_results, ft_results):
        comparison["per_sample"].append({
            "index": b["index"],
            "reference": b["reference"],
            "base_prediction": b["prediction"],
            "ft_prediction": f["prediction"],
            "base_wer": round(b["wer"] * 100, 2),
            "ft_wer": round(f["wer"] * 100, 2),
            "base_cer": round(b["cer"] * 100, 2),
            "ft_cer": round(f["cer"] * 100, 2),
        })

    comp_path = os.path.join(results_dir, "comparison_results.json")
    with open(comp_path, "w", encoding="utf-8") as fp:
        json.dump(comparison, fp, ensure_ascii=False, indent=2)
    print(f"\n[✓] Müqayisə nəticələri saxlandı: {comp_path}")

    # ─── Markdown cədvəl ───
    md_path = os.path.join(results_dir, "comparison_table.md")
    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write("# Model Müqayisəsi: Base vs Fine-Tuned\n\n")
        fp.write("| Metrika | Baza Model | Fine-Tuned | Fərq |\n")
        fp.write("|---------|------------|------------|------|\n")
        fp.write(f"| WER (%) | {base_avg_wer:.2f}% | {ft_avg_wer:.2f}% | {wer_diff:+.2f}% |\n")
        fp.write(f"| CER (%) | {base_avg_cer:.2f}% | {ft_avg_cer:.2f}% | {cer_diff:+.2f}% |\n")
        fp.write(f"\n**Test nümunələri:** {test_samples}\n")
        fp.write(f"\n**Nəticə:** {improvement}\n")

        fp.write("\n## Nümunə-nümunə Müqayisə (ilk 10)\n\n")
        fp.write("| # | Reference | Base Pred | FT Pred | Base WER | FT WER |\n")
        fp.write("|---|-----------|-----------|---------|----------|--------|\n")
        for s in comparison["per_sample"][:10]:
            fp.write(
                f"| {s['index']} | {s['reference'][:30]}... | "
                f"{s['base_prediction'][:30]}... | {s['ft_prediction'][:30]}... | "
                f"{s['base_wer']:.1f}% | {s['ft_wer']:.1f}% |\n"
            )

    print(f"[✓] Markdown cədvəl saxlandı: {md_path}")

    return comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Model Müqayisəsi - Base vs Fine-Tuned")
    parser.add_argument("--base_model", type=str, default="openai/whisper-small")
    parser.add_argument("--fine_tuned_path", type=str, default=None)
    parser.add_argument("--test_samples", type=int, default=50)
    parser.add_argument("--results_dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_models(
        base_model_name=args.base_model,
        fine_tuned_path=args.fine_tuned_path,
        test_samples=args.test_samples,
        results_dir=args.results_dir,
    )
