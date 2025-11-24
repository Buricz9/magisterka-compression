# Wpływ kompresji na modele deep learning w obrazach medycznych

Projekt magisterski testujący wpływ kompresji (JPEG, JPEG2000, AVIF) na accuracy modeli AI dla obrazów kardiologicznych (dataset ARCADE).

## Motywacja

**Problem**: Obrazy medyczne zajmują ogromną ilość miejsca. W systemach PACS (Picture Archiving) i telemedicynie kompresja jest konieczna, ale nie wiadomo jak bardzo wpływa na modele AI.

**Pytanie badawcze**: Czy można skompresować obrazy medyczne bez znaczącej utraty dokładności modeli diagnostycznych AI?

**Nowość**: Pierwsze kompleksowe porównanie JPEG (baseline), JPEG2000 (standard DICOM), i AVIF (2019, cutting-edge) dla AI w kardiologii.

## Co zrobiliśmy

1. **Uproszczenie projektu**: Redukcja kodu z ~4000 do ~900 linii, usunięcie zbędnych skryptów
2. **Obsługa 3 formatów**: JPEG (baseline), JPEG2000 (standard medyczny), AVIF (cutting-edge)
3. **Reorganizacja danych**: `dataset/arcade/` (oryginał) + `dataset/compressed/{jpeg,jpeg2000,avif}/`
4. **Kompresja**: 3000 obrazów × 3 formaty × 6 poziomów jakości = ~54k obrazów
5. **Pomiar jakości**: PSNR, SSIM, compression ratio dla wszystkich formatów

## Struktura

```
dataset/
├── arcade/                    ← Oryginalne PNG
└── compressed/
    ├── jpeg/q{100,85,70,50,30,10}/
    ├── jpeg2000/q{100,85,70,50,30,10}/
    └── avif/q{100,85,70,50,30,10}/

results/
└── metrics/                   ← CSV z PSNR/SSIM
    ├── quality_syntax_train_jpeg.csv
    ├── quality_syntax_train_jpeg2000.csv
    ├── quality_syntax_train_avif.csv
    └── ...
```

## Metodologia eksperymentów

### Eksperyment A: "Czy model nauczy się na złej jakości?"
**Setup**: Trening na skompresowanych obrazach (Q=100,85,70,50,30,10), test na baseline PNG
**Cel**: Sprawdzić czy model wytrenowany na skompresowanych danych radzi sobie z oryginałami
**Zastosowanie**: Szybszy trening (mniejsze pliki) w chmurze
**Pytanie**: Czy strata jakości danych treningowych = strata accuracy?

### Eksperyment B: "Czy model wytrenowany na oryginałach działa na złej jakości?"
**Setup**: Trening na baseline PNG, test na skompresowanych (Q=100,85,70,50,30,10)
**Cel**: Sprawdzić jak model reaguje na kompresję w inference
**Zastosowanie**: Telemedicyna (przesyłanie skompresowanych obrazów)
**Pytanie**: Przy jakiej kompresji model przestaje działać poprawnie?

### Metryki
- **PSNR/SSIM**: Jakość obrazu (wyższe = lepsze)
- **Compression ratio**: Ile miejsca zaoszczędzono
- **Accuracy/F1**: Dokładność modelu (26 klas naczyń krwionośnych)
- **Trade-off**: Który format daje najlepszy stosunek kompresji do accuracy?

## Następne kroki

### Eksperyment A - Trening na skompresowanych
```powershell
.\venv\Scripts\python.exe src\experiment_a.py --model resnet50 --task syntax --format jpeg --epochs 50 --device cuda
.\venv\Scripts\python.exe src\experiment_a.py --model resnet50 --task syntax --format jpeg2000 --epochs 50 --device cuda
.\venv\Scripts\python.exe src\experiment_a.py --model resnet50 --task syntax --format avif --epochs 50 --device cuda
```

### Eksperyment B - Test na skompresowanych
```powershell
.\venv\Scripts\python.exe src\experiment_b.py --model resnet50 --task syntax --format jpeg --epochs 50 --device cuda
.\venv\Scripts\python.exe src\experiment_b.py --model resnet50 --task syntax --format jpeg2000 --epochs 50 --device cuda
.\venv\Scripts\python.exe src\experiment_b.py --model resnet50 --task syntax --format avif --epochs 50 --device cuda
```

## Wyniki

- **Metryki kompresji**: `results/metrics/quality_*.csv`
- **Eksperyment A**: `results/experiment_a/`
- **Eksperyment B**: `results/experiment_b/`
- **Modele**: `models/checkpoints/`

## Oczekiwane wnioski

### 1. JPEG2000 vs JPEG
**Hipoteza**: JPEG2000 lepszy (to standard DICOM w medycynie)
**Sprawdzamy**: Czy standardy medyczne mają rację?

### 2. AVIF vs reszta
**Hipoteza**: AVIF (2019) daje lepszą kompresję przy tej samej jakości
**Sprawdzamy**: Czy warto adoptować nowy format?

### 3. Optymalny punkt kompresji
**Cel**: Znaleźć "sweet spot" - maksymalna kompresja bez utraty accuracy
**Praktyka**: Rekomendacje dla systemów PACS i telemedicyny

### 4. Różnica eksperymentów
- **Eksperyment A** → Wskaże na odporność modelu na kompresję w treningu
- **Eksperyment B** → Wskaże granice kompresji dla inference
- **Porównanie** → Który scenariusz jest bezpieczniejszy?

## Jak interpretować wyniki?

### PSNR/SSIM (results/metrics/)
```
Q=100: PSNR ~45dB, SSIM ~0.99  ← prawie identyczne
Q=50:  PSNR ~35dB, SSIM ~0.95  ← zauważalne artefakty
Q=10:  PSNR ~28dB, SSIM ~0.85  ← mocna degradacja
```

### Accuracy (results/experiment_a/, experiment_b/)
```
Baseline: ~85% accuracy
Q=100: ~84% accuracy  ← dopuszczalne (~1% drop)
Q=50:  ~80% accuracy  ← ostrzeżenie (~5% drop)
Q=10:  ~70% accuracy  ← niedopuszczalne (>15% drop)
```

### Trade-off analysis
**Dobry format**: Wysoka kompresja (3-5x) + niska strata accuracy (<2%)
**Zły format**: Niska kompresja (2x) LUB duża strata accuracy (>5%)

---

**Status**: Kompresja ✅ | Pomiar jakości ✅ | Eksperymenty → do zrobienia

**Czas realizacji**: ~2-3 dni treningów (6 eksperymentów × 2-4h każdy)
