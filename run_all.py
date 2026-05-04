"""
Tam Pipeline — Bütün hissələri ardıcıl icra edir

Bu skript Part A, Part B və hesabat generasiyasını
bir komanda ilə icra etməyə imkan verir.

İstifadə:
    python run_all.py                     # Tam pipeline
    python run_all.py --part a            # Yalnız Part A
    python run_all.py --part b            # Yalnız Part B
    python run_all.py --part c            # Yalnız Hesabat
    python run_all.py --part ab           # Part A + B
    python run_all.py --max_samples 30    # Sürətli test (30 nümunə)
"""

import os
import sys
import argparse
import time

# Proyekt kökünü path-a əlavə et
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_part_a(args):
    """Part A — ASR Baza Tətbiqi"""
    print("\n" + "█" * 70)
    print("█  HİSSƏ A — ASR BAZA TƏTBİQİ")
    print("█" * 70 + "\n")

    from part_a.dataset_loader import load_common_voice_az, inspect_dataset, prepare_for_evaluation
    from part_a.inference import load_model
    from part_a.evaluate import calculate_metrics, print_metrics, save_results

    results_dir = os.path.join(PROJECT_ROOT, "results")

    # A1: Dataset
    dataset = load_common_voice_az(split="test", max_samples=args.max_samples)
    inspect_dataset(dataset, n_samples=3)
    dataset = prepare_for_evaluation(dataset)

    # A2: Model + İnferens
    model = load_model(model_name=args.model, device=args.device)
    results = model.transcribe_batch(dataset)

    # A3: Qiymətləndirmə
    metrics = calculate_metrics(results)
    print_metrics(metrics)
    save_results(metrics, output_dir=results_dir)

    return metrics


def run_part_b(args):
    """Part B — Fine-Tuning"""
    print("\n" + "█" * 70)
    print("█  HİSSƏ B — FİNE-TUNİNG")
    print("█" * 70 + "\n")

    from part_b.fine_tune import run_fine_tuning
    from part_b.compare import compare_models
    from part_b.visualize import generate_all_plots

    results_dir = os.path.join(PROJECT_ROOT, "results")

    # B1 & B2: Fine-tuning
    trainer, history = run_fine_tuning(
        model_name=args.model,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # B3: Müqayisə
    test_for_compare = min(args.max_samples or 50, 50)
    comparison = compare_models(
        base_model_name=args.model,
        test_samples=test_for_compare,
    )

    # B3: Vizualizasiya
    generate_all_plots(results_dir=results_dir)

    return comparison


def run_part_c(args):
    """Part C — Analitik Hesabat"""
    print("\n" + "█" * 70)
    print("█  HİSSƏ C — ANALİTİK HESABAT")
    print("█" * 70 + "\n")

    from generate_report import generate_report

    generate_report(
        results_dir=os.path.join(PROJECT_ROOT, "results"),
        output_path=os.path.join(PROJECT_ROOT, "report.pdf"),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="ASR Pipeline — Azərbaycan dili",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Nümunələr:
  python run_all.py                        # Tam pipeline (A + B + C)
  python run_all.py --part a               # Yalnız Part A
  python run_all.py --part b               # Yalnız Part B
  python run_all.py --part c               # Yalnız hesabat
  python run_all.py --max_samples 20       # Sürətli test
  python run_all.py --epochs 3 --part b    # Fine-tuning (3 epoch)
        """,
    )
    parser.add_argument("--part", type=str, default="abc",
                        help="İcra ediləcək hissələr: a, b, c, ab, ac, bc, abc (default: abc)")
    parser.add_argument("--model", type=str, default="openai/whisper-small",
                        help="HuggingFace model adı")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda, cpu (default: avtomatik)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Test nümunə sayı (Part A)")
    parser.add_argument("--train_samples", type=int, default=150,
                        help="Training nümunə sayı (Part B)")
    parser.add_argument("--val_samples", type=int, default=50,
                        help="Validation nümunə sayı (Part B)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Epoch sayı (Part B)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch ölçüsü (Part B)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Öyrənmə sürəti (Part B)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                                                                ║")
    print("║       🎙️  ASR PİPELİNE — Azərbaycan Dili                      ║")
    print("║       Mozilla Common Voice + OpenAI Whisper                    ║")
    print("║                                                                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"\n  Model:          {args.model}")
    print(f"  Hissələr:       {args.part.upper()}")
    print(f"  Test nümunə:    {args.max_samples or 'Hamısı'}")
    if "b" in args.part.lower():
        print(f"  Train nümunə:   {args.train_samples}")
        print(f"  Val nümunə:     {args.val_samples}")
        print(f"  Epochs:         {args.epochs}")

    start = time.time()
    parts = args.part.lower()

    if "a" in parts:
        run_part_a(args)

    if "b" in parts:
        run_part_b(args)

    if "c" in parts:
        run_part_c(args)

    elapsed = time.time() - start
    print(f"\n{'═' * 70}")
    print(f"  ✅ Pipeline tamamlandı! Ümumi vaxt: {elapsed/60:.1f} dəqiqə")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
