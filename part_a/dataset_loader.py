"""
A1. Dataset Hazırlığı - Mozilla Common Voice Azerbaijani Dataset Yükləyicisi

Bu modul Mozilla Common Voice 17.0 Azərbaycan dili datasetini
Hugging Face-dən yükləyir və ASR inferensi üçün hazırlayır.
"""

import os
import sys
from datasets import load_dataset, Audio


def load_common_voice_az(split="test", streaming=False, max_samples=None):
    """
    Mozilla Common Voice 17.0 Azerbaijani datasetini yükləyir.

    Args:
        split: Dataset split-i ("train", "validation", "test")
        streaming: Streaming rejimində yükləmə (böyük datasetlər üçün)
        max_samples: Maksimum nümunə sayı (None = hamısı)

    Returns:
        dataset: Hazırlanmış dataset
    """
    print(f"[*] Common Voice 17.0 Azerbaijani ({split}) dataseti yüklənir...")

    dataset = load_dataset(
        "google/fleurs",
        "az_az",
        split=split,
        trust_remote_code=True,
        streaming=streaming,
    )

    # Convert raw_transcription to sentence for compatibility
    if "raw_transcription" in dataset.column_names:
        dataset = dataset.rename_column("raw_transcription", "sentence")

    # Audio-nu 16kHz-ə resample et (Whisper tələbi)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if max_samples is not None and not streaming:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"[✓] {len(dataset)} nümunə seçildi ({split} split-dən)")
    elif not streaming:
        print(f"[✓] {len(dataset)} nümunə yükləndi ({split} split)")

    return dataset


def inspect_dataset(dataset, n_samples=5):
    """
    Dataset strukturunu araşdırır və ilk n nümunəni göstərir.

    Args:
        dataset: Yüklənmiş dataset
        n_samples: Göstəriləcək nümunə sayı
    """
    print("\n" + "=" * 60)
    print("DATASET STRUKTURU")
    print("=" * 60)
    print(f"Sütunlar: {dataset.column_names}")
    print(f"Nümunə sayı: {len(dataset)}")
    print(f"Features: {dataset.features}")

    print(f"\n--- İlk {n_samples} nümunə ---")
    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        print(f"\nNümunə {i + 1}:")
        print(f"  Cümlə (sentence): {sample['sentence']}")
        print(f"  Audio uzunluğu: {len(sample['audio']['array']) / sample['audio']['sampling_rate']:.2f} saniyə")
        print(f"  Sampling rate: {sample['audio']['sampling_rate']} Hz")
        if "age" in sample:
            print(f"  Yaş: {sample.get('age', 'N/A')}")
        if "gender" in sample:
            print(f"  Cins: {sample.get('gender', 'N/A')}")


def prepare_for_evaluation(dataset):
    """
    Dataseti qiymətləndirmə üçün hazırlayır.
    Boş və ya çox qısa nümunələri çıxarır.

    Args:
        dataset: Xam dataset

    Returns:
        dataset: Təmizlənmiş dataset
    """
    print("[*] Dataset qiymətləndirmə üçün hazırlanır...")

    # Boş cümlələri çıxar
    dataset = dataset.filter(
        lambda x: x["sentence"] is not None and len(x["sentence"].strip()) > 0
    )

    print(f"[✓] Təmizlənmiş dataset: {len(dataset)} nümunə")
    return dataset


if __name__ == "__main__":
    # Test yükləmə
    ds = load_common_voice_az(split="test", max_samples=10)
    inspect_dataset(ds, n_samples=3)
    ds = prepare_for_evaluation(ds)
