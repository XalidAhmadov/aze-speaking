"""
A2. Model Seçimi və İnferens

Azərbaycan dili üçün OpenAI Whisper modelini istifadə edərək
ASR inferensi həyata keçirir.

Model: openai/whisper-small
- Multilingual dəstək (99+ dil, o cümlədən Azərbaycan)
- 244M parametr
- Yaxşı keyfiyyət/sürət balansı
"""

import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm


class WhisperASR:
    """
    Whisper ASR modeli üçün wrapper sinfi.
    """

    def __init__(self, model_name="openai/whisper-small", device=None):
        """
        Whisper modelini yükləyir.

        Args:
            model_name: HuggingFace model adı
            device: Hesablama cihazı ("cuda", "cpu", və ya None - avtomatik)
        """
        self.model_name = model_name

        # Cihaz seçimi
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[*] Model yüklənir: {model_name}")
        print(f"[*] Cihaz: {self.device}")

        # Processor və model yüklə
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Azərbaycan dili üçün forced decoder IDs
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="azerbaijani", task="transcribe"
        )

        print(f"[✓] Model uğurla yükləndi ({self._count_params():.1f}M parametr)")

    def _count_params(self):
        """Model parametr sayını hesablayır (milyonlarla)."""
        return sum(p.numel() for p in self.model.parameters()) / 1e6

    def transcribe(self, audio_array, sampling_rate=16000):
        """
        Tək bir audio nümunəsini transkripsiya edir.

        Args:
            audio_array: Audio məlumatı (numpy array)
            sampling_rate: Audio sampling rate

        Returns:
            str: Transkripsiya nəticəsi
        """
        # Audio-nu input features-ə çevir
        input_features = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_features.to(self.device)

        # İnferens
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=self.forced_decoder_ids,
                max_new_tokens=225,
            )

        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        return transcription.strip()

    def transcribe_batch(self, dataset, max_samples=None):
        """
        Datasetin bütün nümunələrini transkripsiya edir.

        Args:
            dataset: HuggingFace dataset (audio sütunu ilə)
            max_samples: Maksimum nümunə sayı

        Returns:
            list[dict]: Hər nümunə üçün nəticələr
        """
        results = []
        n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

        print(f"\n[*] {n} nümunə transkripsiya olunur...")

        for i in tqdm(range(n), desc="Transkripsiya"):
            sample = dataset[i]
            audio = sample["audio"]
            reference = sample["sentence"].strip()

            # Transkripsiya
            prediction = self.transcribe(
                audio["array"],
                sampling_rate=audio["sampling_rate"]
            )

            results.append({
                "index": i,
                "reference": reference,
                "prediction": prediction,
                "audio_length": len(audio["array"]) / audio["sampling_rate"],
            })

        print(f"[✓] {len(results)} nümunə transkripsiya olundu")
        return results


def load_model(model_name="openai/whisper-small", device=None):
    """
    Model yaradıb qaytarır.

    Args:
        model_name: HuggingFace model adı
        device: Hesablama cihazı

    Returns:
        WhisperASR: Model instansı
    """
    return WhisperASR(model_name=model_name, device=device)


if __name__ == "__main__":
    # Test
    model = load_model()
    print(f"\nModel: {model.model_name}")
    print(f"Device: {model.device}")
    print("Model uğurla yükləndi və hazırdır.")
