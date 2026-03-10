# Projekt Magisterski: Wpływ kompresji obrazu na skuteczność modeli głębokiego uczenia w diagnostyce kardiologicznej

## 📋 Aktualny Stan Projektu

**Data ostatniej aktualizacji:** 2026-03-10
**Status:** W realizacji
**Ocena jakości kodu:** 8.5/10

---

## ✅ Zakończone Elementy

### Implementacja
- ✅ Kompresja obrazów (JPEG, JPEG2000, AVIF)
- ✅ Pipeline treningowy z AMP (mixed precision)
- ✅ Eksperyment A: Train na skompresowanych, test na oryginałach
- ✅ Eksperyment B: Train na oryginałach, test na skompresowanych
- ✅ Analiza statystyczna (poprawione testy: paired dla Exp B)
- ✅ Feature maps analysis (spectral entropy, Shannon entropy)
- ✅ Pomiar jakości (PSNR, SSIM, compression ratio)

### Modele
- ✅ ResNet-50
- ✅ EfficientNet-B0

### Zbiory danych
- ✅ ARCADE (angiografia wieńcowa) - task: syntax (26 klas)
- ✅ ISIC 2019 (dermoskopia) - 8 klas (benchmark)

---

## 🚀 Uproszczona Struktura Projektu

```
c:/Uczelnia/Magisterka/
├── config.py                    # Konfiguracja
├── DECISIONS.md                 # Ustalenia z promotora
├── STATUS.md                    # Ten plik
│
└── src/                         # Kod źródłowy (12 plików)
    │
    ├── core/                    # Trening i datasety
    │   ├── dataset.py           # ARCADE dataset
    │   ├── isic_dataset.py      # ISIC 2019 dataset
    │   └── train.py             # Trening modeli
    │
    ├── experiments/             # Eksperymenty
    │   ├── experiment_a.py      # Train na skompresowanych
    │   └── experiment_b.py      # Test na skompresowanych
    │
    ├── analysis/                # Analiza wyników
    │   ├── statistical_analysis_corrected.py  # Analiza statystyczna
    │   ├── feature_analysis.py   # Feature maps analysis
    │   └── generate_tables_plots.py  # Tabele i wykresy
    │
    └── processing/              # Przetwarzanie danych
        ├── compress_images.py   # Kompresja ARCADE
        ├── compress_isic.py     # Kompresja ISIC
        └── measure_quality.py   # PSNR, SSIM
```

**Redukcja:** 20→12 plików Python (~40% mniej chaosu)

---

## 📊 Wymagania Promotora (OBOWIĄZKOWE)

✅ **1. EfficientNet-B0** - Drugi model do artykułu
```bash
python src/experiments/experiment_a.py --model efficientnet_b0 --task syntax --format jpeg --mvp
python src/experiments/experiment_b.py --model efficientnet_b0 --task syntax --format jpeg --mvp
```

✅ **2. Analiza statystyczna** - p-value dla porównania formatów
```bash
python src/analysis/statistical_analysis_corrected.py --experiment experiment_a --model resnet50 --task syntax
python src/analysis/statistical_analysis_corrected.py --experiment experiment_b --model resnet50 --task syntax
```

✅ **3. Tabele z wynikami** - Podsumowanie liczebne
```bash
python src/analysis/generate_tables_plots.py --model resnet50 --task syntax
```

✅ **4. Benchmark dataset** - ISIC 2019 (inny zbiór medyczny)
```bash
python src/experiments/experiment_a.py --model resnet50 --dataset isic --format jpeg --mvp
python src/experiments/experiment_b.py --model resnet50 --dataset isic --format jpeg --mvp
```

✅ **5. Feature maps analysis** - Spectral entropy, Shannon entropy
```bash
python src/analysis/feature_analysis.py --model resnet50 --experiment-id <ID> --dataset arcade
```

---

## 🎯 Priorytety na Najbliższy Czas

### WYSOKI priorytet (wymagane do artykułu)
1. **Uruchomić eksperymenty dla ResNet-50** (ARCADE + ISIC)
2. **Uruchomić eksperymenty dla EfficientNet-B0** (ARCADE + ISIC)
3. **Wygenerować poprawione analizy statystyczne**
4. **Stworzyć tabele i wykresy do artykułu**
5. **Przeprowadzić feature analysis dla kluczowych modeli**

---

## 🔧 Kluczowe Polecenia

### Trening i eksperymenty
```bash
# Standard trening
python src/core/train.py --model resnet50 --task syntax --epochs 50

# Eksperyment A: Train na skompresowanych, test na oryginałach
python src/experiments/experiment_a.py --model resnet50 --task syntax --format jpeg --mvp

# Eksperyment B: Train na oryginałach, test na skompresowanych
python src/experiments/experiment_b.py --model resnet50 --task syntax --format jpeg --mvp

# EfficientNet-B0
python src/experiments/experiment_a.py --model efficientnet_b0 --task syntax --format jpeg --mvp

# ISIC 2019
python src/experiments/experiment_a.py --model resnet50 --dataset isic --format jpeg --mvp
```

### Analiza
```bash
# POPRAWIONA analiza statystyczna (zawsze używaj tej!)
python src/analysis/statistical_analysis_corrected.py --experiment experiment_a --model resnet50 --task syntax

# Tabele i wykresy do artykułu
python src/analysis/generate_tables_plots.py --model resnet50 --task syntax

# Feature analysis
python src/analysis/feature_analysis.py --model resnet50 --experiment-id <ID> --dataset arcade
```

### Kompresja
```bash
# Kompresja ARCADE
python src/processing/compress_images.py --format all --task syntax --split all --mvp

# Kompresja ISIC 2019
python src/processing/compress_isic.py --format all --mvp
```

---

## ⚙️ Konfiguracja

### Formaty kompresji
- **JPEG**: Standard internetowy, subsampling=0 (4:4:4)
- **JPEG2000**: Standard DICOM, kompresja przez CR
- **AVIF**: Nowoczesny format (2019), wysoka kompresja

### Poziomy jakości
- **Full**: [100, 95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10]
- **MVP**: [100, 85, 70, 50, 30, 10] (używaj `--mvp` flag)

### Hiperparametry
- Learning rate: 1e-4
- Batch size: 16
- Weight decay: 1e-4
- Optimizer: Adam
- Scheduler: CosineAnnealingLR
- Early stopping patience: 10 epochs
- Random seed: 42

---

## 📈 Postęp

**Zakończone:** ~75%
**Pozostało:** ~25% (głównie uruchomienie eksperymentów i generowanie wyników)

**Szacowany czas do ukończenia:** 2-3 tygodnie intensywnej pracy nad eksperymentami

---

## 📞 Wsparcie

### Dokumentacja
- Decyzje promotora: `DECISIONS.md`
- Ten plik: `STATUS.md`

### Struktura wyników
- JSON z pełnymi metadanymi w `results/*/training_results.json`
- CSV z wynikami w `results/*_results.csv`
- Checkpointy w `models/checkpoints/*/best_model.pth`

---

## 🚨 Ważne Ostrzeżenia

### ZAWSZE używaj poprawionej analizy statystycznej:
- ✅ `src/analysis/statistical_analysis_corrected.py` - POPRAWIONA
- ❌ Stara wersja została usunięta

### Feature analysis jest memory-intensive:
- Zmniejsz `--max-batches` jak potrzebba
- Może wymagać dużej ilości RAM GPU

---

## 🎯 Cel Projektu

**Celem pracy jest wyznaczenie dopuszczalnego poziomu kompresji stratnej dla diagnostyki kardiologicznej wspomaganej sztuczną inteligencją oraz sformułowanie rekomendacji praktycznych dla systemów PACS i platform telemedycznych.**
