"""
Part A — Tam ASR Baza Tətbiqi Pipeline

Bu skript aşağıdakıları ardıcıl olaraq icra edir:
1. Common Voice Azerbaijani datasetini yükləyir
2. Whisper modelini yükləyir
3. Test nümunələri üzərində inferens aparır
4. WER/CER metrikalarını hesablayır
5. Nəticələri saxlayır

İstifadə:
    python run_part_a.py [--max_samples N] [--model MODEL_NAME] [--device DEVICE]

Nümunələr:
    python run_part_a.py                          # Default: bütün test nümunələri
    python run_part_a.py --max_samples 50         # İlk 50 nümunə
    python run_part_a.py --model openai/whisper-medium  # Fərqli model
"""

import sys
import os
import argparse
import time

# Proyekt kökünü path-a əlavə et
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part_a.dataset_loader import load_common_voice_az, inspect_dataset, prepare_for_evaluation
from part_a.inference import load_model
from part_a.evaluate import calculate_metrics, print_metrics, save_results


def parse_args():
    parser = argparse.ArgumentParser(description="ASR Baza Tətbiqi - Azərbaycan dili")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maksimum test nümunəsi sayı (default: hamısı)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-small",
        help="HuggingFace model adı (default: openai/whisper-small)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Hesablama cihazı: cuda, cpu (default: avtomatik)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Nəticələrin saxlanacağı qovluq",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Nəticə qovluğu
    if args.output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        )
    else:
        output_dir = args.output_dir

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       ASR BAZA TƏTBİQİ — Azərbaycan Dili (Part A)        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    start_time = time.time()

    # ─── A1: Dataset Hazırlığı ───
    print("━" * 60)
    print("  A1. DATASET HAZIRLIĞI")
    print("━" * 60)
    dataset = load_common_voice_az(split="test", max_samples=args.max_samples)
    inspect_dataset(dataset, n_samples=3)
    dataset = prepare_for_evaluation(dataset)

    # ─── A2: Model Seçimi və İnferens ───
    print("\n" + "━" * 60)
    print("  A2. MODEL SEÇİMİ VƏ İNFERENS")
    print("━" * 60)
    model = load_model(model_name=args.model, device=args.device)

    inference_start = time.time()
    results = model.transcribe_batch(dataset)
    inference_time = time.time() - inference_start

    print(f"\n[ℹ] İnferens vaxtı: {inference_time:.1f} saniyə")
    print(f"[ℹ] Ortalama inferens: {inference_time/len(results):.2f} san/nümunə")

    # ─── A3: Performans Qiymətləndirməsi ───
    print("\n" + "━" * 60)
    print("  A3. PERFORMANS QİYMƏTLƏNDİRMƏSİ")
    print("━" * 60)
    metrics = calculate_metrics(results)
    print_metrics(metrics)

    # Nəticələri saxla
    save_results(metrics, output_dir=output_dir)

    total_time = time.time() - start_time
    print(f"\n{'━' * 60}")
    print(f"[✓] Part A tamamlandı! Ümumi vaxt: {total_time:.1f} saniyə")
    print(f"[✓] Nəticələr saxlandı: {output_dir}")


if __name__ == "__main__":
    main()
