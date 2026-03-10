# Projekt Magisterski: Wpływ kompresji obrazu na skuteczność modeli głębokiego uczenia w diagnostyce kardiologicznej

## 📋 Aktualny Stan Projektu

**Data ostatniej aktualizacji:** 2026-03-10
**Status:** W realizacji
**Ocena jakości kodu:** 8.5/10 (po wprowadzonych poprawkach)

---

## ✅ Zakończone Elementy

### Implementacja
- ✅ Kompresja obrazów (JPEG, JPEG2000, AVIF) z matched compression ratio
- ✅ Pomiar jakości (PSNR, SSIM)
- ✅ Pipeline treningowy z AMP (mixed precision)
- ✅ Eksperyment A: Train na skompresowanych, test na oryginałach
- ✅ Eksperyment B: Train na oryginałach, test na skompresowanych
- ✅ Analiza statystyczna (z poprawionymi testami: paired dla Exp B)
- ✅ Feature maps analysis (spectral entropy, effective rank)
- ✅ ISIC 2019 dataset (benchmark) - pełna implementacja

### Modele
- ✅ ResNet-50
- ✅ EfficientNet-B0

### Zbiory danych
- ✅ ARCADE (angiografia wieńcowa) - task: syntax (26 klas)
- ✅ ISIC 2019 (dermoskopia) - 8 klas (benchmark)

### Poprawki kodu (2026-03-10)
- ✅ K-fold cross-validation (K=5)
- ✅ Poprawione testy statystyczne (paired t-test dla Exp B)
- ✅ Pełne metadane reprodukcyjności
- ✅ Generowanie tabel LaTeX i wykresów
- ✅ Feature analysis dla ISIC 2019

---

## 🚀 Aktualna Struktura Projektu

```
c:/Uczelnia/Magisterka/
├── config.py                          # Konfiguracja projektu
├── DECISIONS.md                       # Ustalenia z promotorem
│
├── src/                               # Główny kod źródłowy
│   ├── dataset.py                     # ARCADE dataset
│   ├── isic_dataset.py                # ISIC 2019 dataset
│   ├── train.py                       # Trening (standard)
│   ├── train_cv.py                    # Trening z K-fold CV
│   ├── experiment_a.py                # Eksperyment A
│   ├── experiment_b.py                # Eksperyment B
│   ├── experiment_isic.py             # Eksperymenty ISIC
│   ├── run_efficientnet_experiments.py # EfficientNet-B0 runner
│   │
│   ├── statistical_analysis.py        # Analiza statystyczna (oryginał)
│   ├── statistical_analysis_corrected.py # POPRAWIONA analiza statystyczna
│   │
│   ├── feature_analysis.py             # Feature maps analysis
│   ├── generate_tables_plots.py       # Generowanie tabel/wykresów
│   │
│   ├── compress_images.py             # Kompresja ARCADE
│   ├── compress_isic.py               # Kompresja ISIC
│   ├── measure_quality.py             # PSNR, SSIM
│   ├── preprocess_isic.py             # Preprocessing ISIC
│   └── generate_plots.py              (przestarzałe - użyj generate_tables_plots.py)
│
├── dataset/                           # Dane
│   ├── arcade/                        # ARCADE dataset
│   ├── compressed/                    # ARCADE skompresowane
│   ├── isic_2019/                     # ISIC 2019 dataset
│   └── compressed_isic/               # ISIC 2019 skompresowane
│
├── results/                           # Wyniki eksperymentów
│   ├── experiment_a/                  # Wyniki Exp A
│   ├── experiment_b/                  # Wyniki Exp B
│   ├── isic_experiment_a/             # ISIC Exp A
│   ├── isic_experiment_b/             # ISIC Exp B
│   ├── efficientnet_b0/               # EfficientNet-B0 wyniki
│   ├── statistical_analysis/          # Analizy statystyczne
│   ├── feature_analysis/              # Feature maps
│   └── metrics/                       # Metryki jakości
│
├── models/                            # Modele i checkpointy
│   └── checkpoints/                   # Zapisane modele
│
├── plots/                             # Wykresy
└── venv/                              # Virtual environment
```

---

## 📊 Wyniki

### Dostępne wyniki
- ARCADE (ResNet-50): Eksperymenty A i B dla wszystkich formatów
- ISIC 2019 (ResNet-50): Częściowo
- Feature analysis: Przykładowe analizy
- Analizy statystyczne: Z poprawionymi testami

### Brakujące wyniki (do uzyskania)
- EfficientNet-B0 pełne eksperymenty
- K-fold cross-validation wyniki
- Kompletna analiza statystyczna z poprawionymi testami
- Tabele i wykresy do artykułu

---

## 🎯 Priorytety na Najbliższy Czas

### WYSOKI priorytet (wymagane do artykułu)
1. **Uruchomić K-fold cross-validation** dla kluczowych konfiguracji
2. **Wygenerować poprawione analizy statystyczne** (paired tests)
3. **Uruchomić EfficientNet-B0** dla ARCADE i ISIC
4. **Stworzyć tabele i wykresy** do artykułu

### Średni priorytet
5. Feature analysis dla EfficientNet-B0
6. Kompletna analiza porównawcza ResNet-50 vs EfficientNet-B0

---

## 🔧 Kluczowe Pliki i Ich Zastosowanie

### Trening i Eksperymenty
```bash
# Standard trening (pojedynczy run)
python src/train.py --model resnet50 --task syntax --epochs 50

# K-fold cross-validation (zalecane do artykułu)
python src/train_cv.py --model resnet50 --task syntax --k-folds 5

# Eksperyment A (train na skompresowanych)
python src/experiment_a.py --model resnet50 --task syntax --format jpeg --mvp

# Eksperyment B (test na skompresowanych)
python src/experiment_b.py --model resnet50 --task syntax --format jpeg --mvp

# EfficientNet-B0 wszystkie eksperymenty
python src/run_efficientnet_experiments.py --experiment both --dataset both --mvp
```

### Analiza
```bash
# POPRAWIONA analiza statystyczna (używaj tej!)
python src/statistical_analysis_corrected.py --experiment experiment_a --model resnet50 --task syntax
python src/statistical_analysis_corrected.py --experiment experiment_b --model resnet50 --task syntax

# Tabele i wykresy do artykułu
python src/generate_tables_plots.py --model resnet50 --task syntax

# Feature analysis
python src/feature_analysis.py --model resnet50 --experiment-id <ID> --dataset arcade
python src/feature_analysis.py --model resnet50 --experiment-id <ID> --dataset isic
```

### Kompresja
```bash
# Kompresja ARCADE
python src/compress_images.py --format all --task syntax --split all --mvp

# Kompresja ISIC 2019
python src/compress_isic.py --format all --mvp
```

---

## ⚙️ Konfiguracja

### Formaty kompresji
- **JPEG**: Standard internetowy, subsampling=0 (4:4:4)
- **JPEG2000**: Standard DICOM, kompresja przez CR (compression ratio)
- **AVIF**: Nowoczesny format (2019), wysoka kompresja

### Poziomy jakości
- **Full**: [100, 95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10]
- **MVP**: [100, 85, 70, 50, 30, 10]

### Hiperparametry
- Learning rate: 1e-4
- Batch size: 16
- Weight decay: 1e-4
- Optimizer: Adam
- Scheduler: CosineAnnealingLR
- Early stopping patience: 10 epochs
- Random seed: 42

---

## 📝 Najważniejsze Ulepszenia (2026-03-10)

### Krytyczne poprawki metodologiczne
1. **K-fold cross-validation** - zamiast pojedynczego podziału
2. **Paired tests** dla Eksperymentu B - poprawna analiza statystyczna
3. **Metadane reprodukcyjności** - pełne śledzenie wersji bibliotek

### Nowe funkcjonalności
4. **EfficientNet-B0** - drugi model wymagany przez promotora
5. **Generowanie tabel/wykresów** - LaTeX + matplotlib
6. **ISIC 2019 feature analysis** - pełne wsparcie dla benchmarku

---

## 🚨 Ostrzeżenia

### Przestarzałe pliki (nie używaj)
- `src/statistical_analysis.py` - ❌ Użyj `statistical_analysis_corrected.py`
- `src/generate_plots.py` - ❌ Użyj `generate_tables_plots.py`

### Ważne uwagi
- ZAWSZE używaj `statistical_analysis_corrected.py` - stara wersja ma błędne testy!
- K-fold CV wymaga 5× więcej czasu - rozważ K=3 do testów
- Feature analysis jest memory-intensive - zmniejsz `--max-batches` jak potrzeba

---

## 📞 Wsparcie

### Dokumentacja techniczna
- Decyzje promotora: `DECISIONS.md`
- Artykuły naukowe o datasetach: `ArtykułyWykorzystująceDataset.md`

### Struktura wyników
- JSON z pełnymi metadanymi w `results/*/training_results.json`
- CSV z wynikami w `results/*_results.csv`
- Checkpointy w `models/checkpoints/*/best_model.pth`

---

## 📈 Postęp

**Zakończone:** ~75%
**Pozostało:** ~25% (głównie uruchomienie eksperymentów i generowanie wyników)

**Szacowany czas do ukończenia:** 2-3 tygodnie intensywnej pracy nad eksperymentami
