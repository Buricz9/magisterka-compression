# Projekt Magisterski: Wpływ kompresji obrazu na skuteczność modeli głębokiego uczenia w diagnostyce kardiologicznej

**Aktualna wersja:** 1.0.0 (2026-03-10)

---

## 📋 Dokumentacja

- **[DECISIONS.md](DECISIONS.md)** - Ustalenia z promotorem (requirementy, decisions)
- **[STATUS.md](STATUS.md)** - Aktualny stan projektu (postęp, wyniki, priorytety)

---

## 🚀 Szybki Start

### 1. Struktura Projektu

```
├── config.py                          # Konfiguracja
├── DECISIONS.md                       # Ustalenia promotora
├── STATUS.md                          # Stan projektu
│
└── src/                               # Kod źródłowy
    ├── core/                          # Trening i datasety
    │   ├── dataset.py                  # ARCADE dataset
    │   ├── isic_dataset.py             # ISIC 2019 dataset
    │   ├── train.py                    # Standard trening
    │   └── train_cv.py                 # K-fold cross-validation
    │
    ├── experiments/                   # Eksperymenty
    │   ├── experiment_a.py              # Train na skompresowanych
    │   ├── experiment_b.py              # Test na skompresowanych
    │   ├── experiment_isic.py           # Eksperymenty ISIC
    │   └── run_efficientnet_experiments.py  # EfficientNet-B0
    │
    ├── analysis/                      # Analiza wyników
    │   ├── statistical_analysis_corrected.py  # POPRAWIONA analiza statystyczna
    │   ├── feature_analysis.py         # Feature maps analysis
    │   └── generate_tables_plots.py    # Tabele i wykresy do artykułu
    │
    └── processing/                    # Przetwarzanie danych
        ├── compress_images.py          # Kompresja ARCADE
        ├── compress_isic.py            # Kompresja ISIC
        ├── measure_quality.py          # PSNR, SSIM
        └── preprocess_isic.py          # Preprocessing ISIC
```

### 2. Kluczowe Polecenia

```bash
# === TRENING ===

# Standard trening (pojedynczy run)
python -m src.core.train --model resnet50 --task syntax --epochs 50

# K-fold cross-validation (zalecane do artykułu)
python -m src.core.train_cv --model resnet50 --task syntax --k-folds 5

# === EKSPERYMENTY ===

# Eksperyment A: Train na skompresowanych, test na oryginałach
python -m src.experiments.experiment_a --model resnet50 --task syntax --format jpeg --mvp

# Eksperyment B: Train na oryginałach, test na skompresowanych
python -m src.experiments.experiment_b --model resnet50 --task syntax --format jpeg --mvp

# EfficientNet-B0 wszystkie eksperymenty (ARCADE + ISIC)
python -m src.experiments.run_efficientnet_experiments --experiment both --dataset both --mvp

# === ANALIZA ===

# POPRAWIONA analiza statystyczna (zawsze używaj tej!)
python -m src.analysis.statistical_analysis_corrected --experiment experiment_a --model resnet50 --task syntax
python -m src.analysis.statistical_analysis_corrected --experiment experiment_b --model resnet50 --task syntax

# Tabele i wykresy do artykułu
python -m src.analysis.generate_tables_plots --model resnet50 --task syntax

# Feature analysis
python -m src.analysis.feature_analysis --model resnet50 --experiment-id <ID> --dataset arcade

# === PRZETWARZANIE ===

# Kompresja ARCADE
python -m src.processing.compress_images --format all --task syntax --split all --mvp

# Kompresja ISIC 2019
python -m src.processing.compress_isic --format all --mvp

# Pomiar jakości
python -m src.processing.measure_quality --format all --task syntax --split all --mvp
```

---

## ⚠️ Ważne Ostrzeżenia

### NIE UŻYWAJ tych plików (przestarzałe):
- ❌ `src/analysis/statistical_analysis.py` - **Użyj `statistical_analysis_corrected.py`**
- ❌ Starej wersji `generate_plots.py` - **Użyj `generate_tables_plots.py`**

### ZAWSZE używaj:
- ✅ `src/analysis/statistical_analysis_corrected.py` - Poprawione testy statystyczne
- ✅ `src/core/train_cv.py` - K-fold cross-validation dla wiarygodnych wyników

---

## 📊 Formaty i Poziomy Kompresji

### Formaty:
- **JPEG**: Standard internetowy
- **JPEG2000**: Standard DICOM (medycyna)
- **AVIF**: Nowoczesny format (2019)

### Poziomy jakości:
- **Full**: [100, 95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10]
- **MVP**: [100, 85, 70, 50, 30, 10] (używaj `--mvp` flag)

---

## 🎯 Priorytety

Do ukończenia artykułu potrzebujesz:

1. **K-fold cross-validation** wyniki (kluczowe dla statystyki)
2. **Poprawione analizy statystyczne** (paired tests dla Exp B)
3. **EfficientNet-B0** wyniki (wymóg promotora)
4. **Tabele i wykresy** do artykułu

---

## 📞 Wsparcie

- Szczegółowy status projektu: [STATUS.md](STATUS.md)
- Ustalenia z promotorem: [DECISIONS.md](DECISIONS.md)

---

## 📈 Postęp

**Zakończone:** ~75%
**Pozostało:** ~25% (głównie uruchomienie eksperymentów i generowanie wyników)
