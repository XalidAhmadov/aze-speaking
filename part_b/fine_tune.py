"""
B1 & B2. Fine-Tuning Pipeline

Whisper modelini Azərbaycan dili üçün fine-tune edir.
- Data hazırlığı (train/validation split)
- Fine-tuning (Seq2Seq Trainer ilə)
- Hər epoch-da validation WER izlənir
- Ən yaxşı checkpoint saxlanılır
- Overfitting-in qarşısı alınır

İstifadə:
    python fine_tune.py [--train_samples N] [--val_samples N] [--epochs N]

Nümunələr:
    python fine_tune.py                                  # Default (150 train, 50 val)
    python fine_tune.py --train_samples 200 --epochs 10  # Daha çox data/epoch
"""

import os
import sys
import json
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
import evaluate

# Proyekt kökünü path-a əlavə et
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════
# B1. DATA HAZIRLIĞI
# ═══════════════════════════════════════════════════════════════

def prepare_dataset(processor, train_samples=150, val_samples=50):
    """
    Common Voice datasetini fine-tuning üçün hazırlayır.

    Args:
        processor: WhisperProcessor
        train_samples: Training nümunə sayı
        val_samples: Validation nümunə sayı

    Returns:
        train_dataset, val_dataset
    """
    print("[*] Dataset yüklənir...")

    # Training üçün train split, validation üçün validation split
    train_ds = load_dataset(
        "google/fleurs", "az_az",
        split="train", trust_remote_code=True
    )
    val_ds = load_dataset(
        "google/fleurs", "az_az",
        split="validation", trust_remote_code=True
    )

    if "raw_transcription" in train_ds.column_names:
        train_ds = train_ds.rename_column("raw_transcription", "sentence")
    if "raw_transcription" in val_ds.column_names:
        val_ds = val_ds.rename_column("raw_transcription", "sentence")

    # Nümunə sayını məhdudlaşdır
    train_ds = train_ds.select(range(min(train_samples, len(train_ds))))
    val_ds = val_ds.select(range(min(val_samples, len(val_ds))))

    print(f"[✓] Train: {len(train_ds)} nümunə, Validation: {len(val_ds)} nümunə")

    # 16kHz-ə resample
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
    val_ds = val_ds.cast_column("audio", Audio(sampling_rate=16000))

    # Boş cümlələri sil
    train_ds = train_ds.filter(
        lambda x: x["sentence"] is not None and len(x["sentence"].strip()) > 0
    )
    val_ds = val_ds.filter(
        lambda x: x["sentence"] is not None and len(x["sentence"].strip()) > 0
    )

    def prepare_batch(batch):
        """Batch-ı model input formatına çevirir."""
        audio = batch["audio"]
        # Input features
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        # Labels
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    print("[*] Data preprocessing başladı...")
    train_ds = train_ds.map(
        prepare_batch,
        remove_columns=train_ds.column_names,
        num_proc=1,
    )
    val_ds = val_ds.map(
        prepare_batch,
        remove_columns=val_ds.column_names,
        num_proc=1,
    )

    print(f"[✓] Data hazırlandı!")
    return train_ds, val_ds


# ═══════════════════════════════════════════════════════════════
# DATA COLLATOR
# ═══════════════════════════════════════════════════════════════

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Whisper fine-tuning üçün xüsusi data collator.
    Input features və labels-ı düzgün pad edir.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Input features-ı batch-a yığ
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Labels-ı pad et
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Pad token-ləri -100 ilə əvəz et (loss hesablamasından çıxar)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # BOS token-i çıxar (əgər varsa)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ═══════════════════════════════════════════════════════════════
# TRAINİNG METRİKALARI İZLƏYİCİ CALLBACK
# ═══════════════════════════════════════════════════════════════

class MetricsLoggerCallback(TrainerCallback):
    """
    Hər epoch sonunda training və validation metrikalarını
    log edir (qrafik üçün).
    """

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.eval_wers = []
        self.epochs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.train_losses.append({
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "loss": logs["loss"],
                })

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            self.eval_losses.append({
                "step": state.global_step,
                "epoch": state.epoch,
                "eval_loss": metrics.get("eval_loss", None),
            })
            if "eval_wer" in metrics:
                self.eval_wers.append({
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "eval_wer": metrics["eval_wer"],
                })
            self.epochs.append(state.epoch)

    def get_history(self):
        return {
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "eval_wers": self.eval_wers,
            "epochs": self.epochs,
        }


# ═══════════════════════════════════════════════════════════════
# B2. FİNE-TUNİNG
# ═══════════════════════════════════════════════════════════════

def run_fine_tuning(
    model_name="openai/whisper-small",
    train_samples=150,
    val_samples=50,
    num_epochs=5,
    batch_size=8,
    learning_rate=1e-5,
    output_dir=None,
    results_dir=None,
):
    """
    Whisper modelini Azərbaycan dili üçün fine-tune edir.

    Args:
        model_name: Baza model adı
        train_samples: Training nümunə sayı
        val_samples: Validation nümunə sayı
        num_epochs: Epoch sayı
        batch_size: Batch ölçüsü
        learning_rate: Öyrənmə sürəti
        output_dir: Checkpoint-lərin saxlanacağı qovluq
        results_dir: Nəticələrin saxlanacağı qovluq
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if output_dir is None:
        output_dir = os.path.join(project_root, "part_b", "checkpoints")
    if results_dir is None:
        results_dir = os.path.join(project_root, "results")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          WHISPER FİNE-TUNİNG — Azərbaycan Dili            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Model:          {model_name}")
    print(f"  Device:         {device}")
    print(f"  Train nümunə:   {train_samples}")
    print(f"  Val nümunə:     {val_samples}")
    print(f"  Epochs:         {num_epochs}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Learning rate:  {learning_rate}")
    print()

    # ─── Model və Processor Yüklə ───
    print("[*] Model yüklənir...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Azərbaycan dili konfiqurasiyası
    model.generation_config.language = "azerbaijani"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    print(f"[✓] Model yükləndi: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parametr")

    # ─── Data Hazırlığı ───
    train_ds, val_ds = prepare_dataset(processor, train_samples, val_samples)

    # ─── Data Collator ───
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ─── Metrika Funksiyası ───
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # -100 ilə əvəz edilmiş token-ləri pad token-ə çevir
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer_val = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer_val}

    # ─── Callback ───
    metrics_logger = MetricsLoggerCallback()

    # ─── Training Arguments ───
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        warmup_steps=50,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,
    )

    # ─── Trainer ───
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=[metrics_logger],
    )

    # ─── Training Başlat ───
    print("\n" + "━" * 60)
    print("  TRAINING BAŞLADI")
    print("━" * 60)

    train_result = trainer.train()

    # ─── Nəticələri Saxla ───
    print("\n[*] Ən yaxşı model saxlanır...")
    trainer.save_model(os.path.join(output_dir, "best_model"))
    processor.save_pretrained(os.path.join(output_dir, "best_model"))

    # Training tarixçəsini saxla
    history = metrics_logger.get_history()

    # Trainer-dən əlavə log-lar
    train_log = train_result.metrics
    history["final_train_metrics"] = train_log

    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)
    print(f"[✓] Training tarixçəsi saxlandı: {history_path}")

    # Final evaluation
    print("\n[*] Final evaluation...")
    eval_results = trainer.evaluate()
    print(f"[✓] Final Eval WER: {eval_results.get('eval_wer', 'N/A'):.2f}%")
    print(f"[✓] Final Eval Loss: {eval_results.get('eval_loss', 'N/A'):.4f}")

    eval_path = os.path.join(results_dir, "fine_tuned_eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[✓] Fine-tuning tamamlandı!")
    print(f"[✓] Ən yaxşı model: {os.path.join(output_dir, 'best_model')}")

    return trainer, history


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper Fine-Tuning - Azərbaycan dili")
    parser.add_argument("--model", type=str, default="openai/whisper-small")
    parser.add_argument("--train_samples", type=int, default=150)
    parser.add_argument("--val_samples", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer, history = run_fine_tuning(
        model_name=args.model,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
    )
