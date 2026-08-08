# 🎙️ Azərbaycan Dili üçün Avtomatik Nitq Tanıma (ASR)


## Layihənin Qısa İzahatı

Bu layihə Google FLEURS datasetindən istifadə edərək Azərbaycan dili üçün avtomatik nitq tanıma (ASR) sistemini tətbiq edir. Layihə üç hissədən ibarətdir:

- **Hissə A**: Baza model inferensi və performans qiymətləndirməsi
- **Hissə B**: Fine-tuning cəhdi və müqayisə
- **Hissə C**: Analitik hesabat

## İstifadə Olunan Model və Parametrlər

| Parametr | Dəyər |
|----------|-------|
| **Model** | `openai/whisper-small` |
| **Parametr sayı** | 244M |
| **Dil** | Azərbaycan (az) |
| **Dataset** | Google FLEURS (az_az) |


## WER/CER Nəticələri

### Baza Model

| Metrika | Dəyər |
|---------|-------|
| Ortalama WER | *Skript icra edildikdən sonra doldurulacaq* |
| Ortalama CER | *Skript icra edildikdən sonra doldurulacaq* |

### Fine-Tuning Müqayisəsi

| Metrika | Baza Model | Fine-Tuned | Fərq |
|---------|------------|------------|------|
| WER (%) | — | — | — |
| CER (%) | — | — | — |

> **Qeyd**: Cədvəllər `python run_all.py` icra edildikdən sonra `results/` qovluğunda avtomatik yaradılır.

## Kodu İşə Salmaq Üçün Addımlar

### 1. Mühit Quraşdırması

```bash
python -m venv venv

venv\Scripts\activate


# Asılılıqları quraşdır
pip install -r requirements.txt
```

### 2. Tam Pipeline (A + B + C)

```bash
python run_all.py
```

### 3. Hissələri Ayrı-Ayrı İcra Et

```bash
# Yalnız Part A (Baza model qiymətləndirməsi)
python run_all.py --part a

# Yalnız Part A (sürətli test — 30 nümunə)
python run_all.py --part a --max_samples 30

# Yalnız Part B (Fine-tuning + Müqayisə + Qrafiklər)
python run_all.py --part b

# Yalnız Part C (PDF hesabat yaradılması)
python run_all.py --part c
```

### 4. Xüsusi Parametrlərlə

```bash
# Fərqli model istifadə et
python run_all.py --model openai/whisper-medium

# Fine-tuning parametrlərini dəyiş
python run_all.py --part b --epochs 10 --train_samples 200 --learning_rate 5e-6

# GPU istifadə et
python run_all.py --device cuda
```

### 5. Ayrı Skriptlərlə

```bash
# Part A ayrıca
cd part_a
python run_part_a.py --max_samples 50

# Fine-tuning ayrıca
cd part_b
python fine_tune.py --epochs 5 --train_samples 150

# Müqayisə ayrıca
cd part_b
python compare.py --test_samples 50

# Qrafiklər ayrıca
cd part_b
python visualize.py

# PDF hesabat ayrıca
python generate_report.py
```

## Layihə Strukturu

```
az-stt-intern/
├── README.md                    ← Bu fayl
├── requirements.txt             ← Python asılılıqları
├── run_all.py                   ← Tam pipeline runner
├── generate_report.py           ← PDF hesabat generatoru
│
├── part_a/                      ← Hissə A — ASR Baza Tətbiqi
│   ├── __init__.py
│   ├── dataset_loader.py        ← A1: Dataset hazırlığı
│   ├── inference.py             ← A2: Whisper model inferensi
│   ├── evaluate.py              ← A3: WER/CER hesablama
│   └── run_part_a.py            ← Part A runner
│
├── part_b/                      ← Hissə B — Fine-Tuning
│   ├── __init__.py
│   ├── fine_tune.py             ← B1-B2: Data hazırlığı + Fine-tuning
│   ├── compare.py               ← B3: Model müqayisəsi
│   ├── visualize.py             ← B3: Qrafik vizualizasiya
│   └── checkpoints/             ← Saxlanmış modellər
│
├── results/                     ← Nəticələr
│   ├── base_model_results.json  ← Baza model metrikaları
│   ├── base_model_all_samples.csv
│   ├── base_model_summary.md
│   ├── training_history.json    ← Fine-tuning tarixçəsi
│   ├── comparison_results.json  ← Müqayisə nəticələri
│   ├── comparison_table.md
│   ├── training_loss.png        ← Loss qrafiki
│   ├── wer_per_epoch.png        ← WER qrafiki
│   ├── comparison_bar.png       ← Müqayisə bar chart
│   └── comparison_scatter.png   ← Scatter plot
│
└── report.pdf                   ← Hissə C — Analitik hesabat
```
