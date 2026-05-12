# Wpływ kompresji obrazu na klasyfikację segmentów tętnic wieńcowych

Praca magisterska. Porównanie wpływu kompresji **JPEG / JPEG2000 / AVIF** na skuteczność modeli głębokiego uczenia (**ResNet-50**, **EfficientNet-B0**) w zadaniu **multi-label classification** segmentów tętnic wieńcowych (26 klas SYNTAX Score) na zbiorze **ARCADE** (1500 obrazów angiografii rentgenowskiej 512×512 PNG).

## Struktura projektu

```
C:\Uczelnia\Magisterka\
├── config.py                          # ścieżki, hiperparametry, get_data_path
├── artykul.tex / artykul.pdf           # praca magisterska
├── dataset/
│   ├── arcade/                         # oryginalne PNG (1000 train, 200 val, 300 test)
│   ├── compressed/{format}/q{Q}/...    # generowane przez compress_images.py
│   └── cr_maps/                        # JSON z compression ratios per obraz
├── src/
│   ├── core/
│   │   ├── dataset.py                  # ARCADE DataLoader (multi-label, 26 klas)
│   │   ├── isic_dataset.py             # ISIC 2019 (opcjonalny benchmark)
│   │   └── train.py                    # train_model, evaluate_model, create_model
│   ├── processing/
│   │   ├── compress_images.py          # kompresja JPEG/JP2/AVIF z matched-CR
│   │   ├── measure_quality.py          # PSNR/SSIM/CR per format
│   │   └── compress_isic.py            # analog dla ISIC (poza scope tezy)
│   ├── experiments/
│   │   ├── experiment_a.py             # trening na compressed, test na PNG
│   │   └── run_baseline.py             # trening i test na PNG
│   └── analysis/
│       ├── generate_tables_plots.py    # LaTeX tables, comparison plots
│       ├── generate_quality_plots.py   # PSNR/SSIM/CR plots
│       ├── statistical_analysis_corrected.py  # t-test, ANOVA, Kruskal
│       └── feature_analysis.py         # spectral entropy, effective rank
├── models/checkpoints/                 # ResNet/EfficientNet best.pth per (model, format, Q)
├── results/
│   ├── experiment_a/*.csv              # wyniki klasyfikacji
│   └── metrics/quality_*.csv           # PSNR/SSIM/CR
└── plots/                              # PDF wykresów
```

## Metodologia kluczowa

### Compression Ratio (RAW-based)

```
CR = (H × W × C × bits/8) / file_size_bytes
```

Dla 512×512 RGB 8-bit: `CR = 786432 / file_size`. Konwencja zgodna z ISO 15444 JPEG2000, ACR/DICOM medical imaging guidelines, AVIF/AV1 benchmarks. **Wyższe CR → mocniejsza kompresja → niższa jakość.**

### Q dla każdego formatu

- **JPEG**: natywny `quality` ∈ [1, 100] przekazywany do libjpeg. `subsampling=0` (4:4:4, pełna chrominancja), `optimize=True`.
- **JPEG2000**: Q to *etykieta zastępcza*. Procedura: kompresuj JPEG z Q → zmierz `S_JPEG` → binary search nad `quality_layers[rate]` (w `quality_mode="rates"`) aż plik JP2 ma rozmiar ≈ `S_JPEG` (tolerancja 5%). Parametry: `irreversible=True` (9/7 wavelet lossy), `mct=1` (Multi-Component Transform).
- **AVIF**: analogicznie, binary search nad `quality` ∈ [1, 99] (cap=99, bo quality=100 w pillow-avif-plugin aktywuje wewnętrzny tryb lossless). Parametry: `subsampling="4:4:4"`, `speed=6`, `range="full"`.

### Trening (Eksperyment A)

- Backbone: ResNet-50 / EfficientNet-B0, ImageNet pretrained, head wymieniony na 26-output (sigmoid w loss)
- Loss: `BCEWithLogitsLoss` z `pos_weight` per-class (radzenie sobie z 18× imbalance)
- Optimizer: Adam, `lr=1e-4`, `weight_decay=1e-4`
- Early stopping na `val_f1_macro`, `patience=10`
- Train/val w domenie compressed (`train_quality == val_quality`), test na oryginalnych PNG

## Komendy

### Setup
```powershell
cd C:\Uczelnia\Magisterka
.\venv\Scripts\Activate.ps1
```

### 1) Kompresja wszystkich formatów (~5h)
```powershell
python -m src.processing.compress_images --format all --task syntax --split all
```
JPEG kompresowane pierwsze (buduje CR map), potem JP2 i AVIF z matched-CR.

### 2) Pomiar jakości obrazu (~10 min, opcjonalne)
```powershell
python -m src.processing.measure_quality --task syntax --format jpeg     --split all
python -m src.processing.measure_quality --task syntax --format jpeg2000 --split all
python -m src.processing.measure_quality --task syntax --format avif     --split all
```

### 3) Baseline na PNG
```powershell
python -m src.experiments.run_baseline --model resnet50        --task syntax
python -m src.experiments.run_baseline --model efficientnet_b0 --task syntax
```

### 4) Eksperyment A — trening per format
```powershell
python -m src.experiments.experiment_a --model resnet50        --format jpeg
python -m src.experiments.experiment_a --model resnet50        --format jpeg2000
python -m src.experiments.experiment_a --model resnet50        --format avif
python -m src.experiments.experiment_a --model efficientnet_b0 --format jpeg
python -m src.experiments.experiment_a --model efficientnet_b0 --format jpeg2000
python -m src.experiments.experiment_a --model efficientnet_b0 --format avif
```

Wszystkie poziomy Q z `config.QUALITY_LEVELS = [100, 95, …, 10]`. Można dodać `--quality 100` żeby przetrenować tylko jeden poziom (merge CSV).

### 5) Tabele i wykresy
```powershell
python -m src.analysis.generate_tables_plots --model resnet50        --task syntax
python -m src.analysis.generate_tables_plots --model efficientnet_b0 --task syntax
python -m src.analysis.generate_quality_plots
```

### 6) Analiza statystyczna (po retreningu)
```powershell
python -m src.analysis.statistical_analysis_corrected --experiment experiment_a --model resnet50        --task syntax
python -m src.analysis.statistical_analysis_corrected --experiment experiment_a --model efficientnet_b0 --task syntax
```

## Decyzje promotorskie

| Data | Decyzja | Uzasadnienie |
|---|---|---|
| **25.11.2025** | Inicjalny kontakt, link do GitHub | — |
| **16.02.2026** | Dodać EfficientNet-B0 jako drugi model; analiza statystyczna (p-value); benchmark dataset (ISIC); feature maps analysis (spectral/Shannon entropy) | Za mało materiału do publikacji |
| **26.03.2026** | Przejście z single-label na multi-label (sigmoid + BCE + pos_weight) | 99.3% obrazów ma >1 segment różnej klasy — single-label arbitralny |
| **2026-05-12** | Q=100 dla JP2/AVIF musi być stratny (nie lossless); zdefiniować jak Q jest liczony dla każdego formatu | Promotor wykryła PSNR=∞ dla JP2 Q=100 (fizycznie niemożliwe). Diagnoza: gałąź ratunkowa `if CR<=1` zapisywała JP2/AVIF bezstratnie dla 99.7% obrazów testowych. Fix: matched-CR przez binary search, cap AVIF quality≤99, RAW-based CR. |

## Known issues / metodologiczne caveats

- **Klasy 11 i 25 nie występują w zbiorze treningowym** (out of 26 SYNTAX classes). Wpływ na mAP — raportujemy `n_classes_present`.
- **Imbalance 18×** między klasą najczęstszą a najrzadszą — rozwiązane przez `pos_weight` w `BCEWithLogitsLoss`.
- **`n=1` per (format, Q)** — testy statystyczne (t-test, ANOVA) na 13 punktach Q traktowanych jako próba IID są dyskusyjne. Lepiej raportować `mean ± std` po Q jako opisówkę.
- **Subsampling JPEG vs AVIF** — oba 4:4:4, zgodnie z metodologią. JP2 ma analogicznie `mct=1` (RGB → YCbCr w domenie falkowej).
- **Edge case Q=100**: JPEG Q=100 z 4:4:4 może być większy niż źródłowy PNG (PNG to bezstratny deflate, świetny dla angiografii z dużym czarnym tłem). Binary search dla JP2/AVIF dopasowuje rozmiar do `S_JPEG` — może wypisać warning gdy target nie do osiągnięcia, plik wychodzi mniejszy, ale wciąż stratny.

## Datasety

- **ARCADE/Syntax** (główny): 1500 obrazów 512×512 angiografii rentgenowskiej. 1000 train / 200 val / 300 test. 26 klas SYNTAX Score. Multi-label.
- **ISIC 2019** (opcjonalny benchmark): w `dataset/isic_2019/`. Kod w `src/core/isic_dataset.py` i `src/processing/compress_isic.py`. Poza scope obecnej tezy.
