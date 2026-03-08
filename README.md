# Wpływ kompresji na modele deep learning w obrazach medycznych

Wpływ kompresji obrazów (JPEG, JPEG2000, AVIF) na skuteczność modeli AI w diagnostyce kardiologicznej.

## Zbiory danych

**ARCADE** - 3000 obrazów angiografii wieńcowej (26 klas klasyfikacji)
- Source: https://zenodo.org/records/10390295

**ISIC 2019** (benchmark) - 25,331 obrazów dermoskopii (8 klas)
- Source: https://challenge.isic-archive.com/landing/2019/
- Klasy: MEL, NV, BCC, AK, BKL, DF, VASC, SCC

## Formaty kompresji

| Format | Opis | Zastosowanie |
|-------|------|--------------|
| **JPEG** | Baseline | Standard internetowy |
| **JPEG2000** | Standard DICOM | Medycyna, PACS |
| **AVIF** | Nowoczesny (2019) | Cutting-edge, wysoka kompresja |

## Modele

- **ResNet-50** - klasyczny CNN
- **EfficientNet-B0** - wydajna architektura

## Eksperymenty

### A: Trening na skompresowanych, test na oryginałach
### B: Trening na oryginałach, test na skompresowanych

## Szybkie start

```bash
# Kompresja (wszystkie formaty)
python src/compress_images.py --format all --task syntax --split all --mvp

# Pomiar jakości
python src/measure_quality.py --format all --task syntax --split all --mvp

# Eksperyment A
python src/experiment_a.py --model resnet50 --task syntax --format jpeg --mvp --device cuda
python src/experiment_a.py --model resnet50 --task syntax --format jpeg2000 --mvp --device cuda
python src/experiment_a.py --model resnet50 --task syntax --format avif --mvp --device cuda

# Eksperyment B
python src/experiment_b.py --model resnet50 --task syntax --format jpeg --mvp --device cuda
python src/experiment_b.py --model resnet50 --task syntax --format jpeg2000 --mvp --device cuda
python src/experiment_b.py --model resnet50 --task syntax --format avif --mvp --device cuda

# ISIC 2019 (benchmark)
# 1. Pobierz dataset z https://challenge.isic-archive.com/landing/2019/
# 2. Preprocessing:
python src/preprocess_isic.py --input-root dataset/isic_2019_raw --output-root dataset/isic_2019

# 3. Kompresja:
python src/compress_isic.py --format all --mvp

# 4. Eksperymenty:
python src/experiment_isic.py --experiment both --model resnet50 --mvp --device cuda
```

## Pełna dokumentacja
- **DECISIONS.md** - decyzje promotora i status projektu
- **ArtykułyWykorzystująceDataset.md** - literatura
