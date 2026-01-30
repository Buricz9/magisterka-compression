# TODO - Projekt Magisterski

## 📋 Wpływ kompresji na modele deep learning w obrazach medycznych

**Obsługiwane formaty:** JPEG, JPEG2000, AVIF

---
test
## ✅ GOTOWE (Przygotowanie)

- [x] Pobranie datasetu ARCADE (3000 obrazów)
- [x] Kod kompresji dla 3 formatów (JPEG, JPEG2000, AVIF)
- [x] Kod pomiaru metryk jakości (PSNR, SSIM)
- [x] Implementacja DataLoadera z obsługą wszystkich formatów
- [x] Pipeline treningowy z obsługą formatów

---

## 🔴 DO ZROBIENIA

### 1. Kompresja danych

**Dla każdego formatu osobno lub wszystkich naraz:**

```bash
# JPEG (już może być zrobione)
python src/compress_images.py --format jpeg --task all --split all --mvp

# JPEG2000 (NOWY!)
python src/compress_images.py --format jpeg2000 --task all --split all --mvp

# AVIF (NOWY!)
python src/compress_images.py --format avif --task all --split all --mvp

# LUB wszystko naraz:
python src/compress_images.py --format all --task all --split all --mvp
```

---

### 2. Pomiar jakości kompresji

```bash
# Dla każdego formatu
python src/measure_quality.py --format jpeg --task all --split all --mvp
python src/measure_quality.py --format jpeg2000 --task all --split all --mvp
python src/measure_quality.py --format avif --task all --split all --mvp

# LUB wszystko naraz:
python src/measure_quality.py --format all --task all --split all --mvp
```

---

### 3. Eksperyment A - Trening na skompresowanych danych

**Dla każdego formatu:**

- [ ] JPEG - ResNet-50, syntax
- [ ] JPEG - EfficientNet-B0, syntax
- [ ] JPEG2000 - ResNet-50, syntax
- [ ] JPEG2000 - EfficientNet-B0, syntax
- [ ] AVIF - ResNet-50, syntax
- [ ] AVIF - EfficientNet-B0, syntax

**Komendy:**
```bash
# JPEG
python src/experiment_a.py --model resnet50 --task syntax --format jpeg --epochs 50 --device cuda

# JPEG2000
python src/experiment_a.py --model resnet50 --task syntax --format jpeg2000 --epochs 50 --device cuda

# AVIF
python src/experiment_a.py --model resnet50 --task syntax --format avif --epochs 50 --device cuda
```

---

### 4. Eksperyment B - Test na skompresowanych danych

**Dla każdego formatu:**

- [ ] JPEG - ResNet-50, syntax
- [ ] JPEG2000 - ResNet-50, syntax
- [ ] AVIF - ResNet-50, syntax

**Komendy:**
```bash
# JPEG
python src/experiment_b.py --model resnet50 --task syntax --format jpeg --epochs 50 --device cuda

# JPEG2000
python src/experiment_b.py --model resnet50 --task syntax --format jpeg2000 --epochs 50 --device cuda

# AVIF
python src/experiment_b.py --model resnet50 --task syntax --format avif --epochs 50 --device cuda
```

---

## 📊 ANALIZA WYNIKÓW

### 5. Porównanie formatów

- [ ] Wykresy: JPEG vs JPEG2000 vs AVIF - accuracy vs compression ratio
- [ ] Wykresy: PSNR/SSIM dla każdego formatu
- [ ] Tabele porównawcze
- [ ] Analiza: który format najlepszy dla AI medycznego?

**Pytania badawcze:**
1. Czy JPEG2000 (standard medyczny) rzeczywiście lepszy niż JPEG?
2. Czy AVIF (najnowszy) przewyższa oba poprzednie?
3. Jaki format daje najlepszy trade-off: rozmiar vs accuracy?

---

## 📝 PISANIE PRACY

### 6. Rozdziały

- [ ] Wprowadzenie i motywacja
- [ ] State-of-the-art
- [ ] Metodologia
  - [ ] Dataset ARCADE
  - [ ] **3 formaty kompresji:** JPEG, JPEG2000, AVIF
  - [ ] Metryki (PSNR, SSIM, accuracy, F1)
- [ ] Wyniki eksperymentów
  - [ ] **Porównanie JPEG vs JPEG2000 vs AVIF**
  - [ ] Eksperyment A i B dla każdego formatu
- [ ] Dyskusja
  - [ ] **Rekomendacje dla systemów medycznych**
  - [ ] Który format wybrać?
- [ ] Wnioski

---

## 🎯 WARTOŚĆ NAUKOWA

**Unikalne aspekty Twojej pracy:**

1. **3 formaty kompresji:**
   - JPEG (baseline, wszechobecny)
   - JPEG2000 (standard medyczny DICOM)
   - AVIF (cutting-edge, 2019)

2. **Pierwsze kompleksowe porównanie** tych formatów dla AI w obrazach kardiologicznych

3. **Praktyczne rekomendacje:**
   - Dla PACS (Picture Archiving)
   - Dla telemedicyny (transmisja obrazów)
   - Dla systemów AI diagnostycznych

---

## ⚙️ SZYBKIE KOMENDY

### Tryb MVP (szybkie testy):
```bash
# Kompresja wszystkich formatów
python src/compress_images.py --format all --task all --split all --mvp

# Pomiar jakości wszystkich formatów
python src/measure_quality.py --format all --task all --split all --mvp

# Eksperyment A - JPEG2000
python src/experiment_a.py --model resnet50 --task syntax --format jpeg2000 --epochs 5 --mvp

# Eksperyment B - AVIF
python src/experiment_b.py --model resnet50 --task syntax --format avif --epochs 5 --mvp
```

### Produkcja (pełne eksperymenty):
```bash
# Wszystko dla jednego formatu
python src/compress_images.py --format jpeg2000 --task all --split all
python src/measure_quality.py --format jpeg2000 --task all --split all
python src/experiment_a.py --model resnet50 --task syntax --format jpeg2000 --epochs 50
python src/experiment_b.py --model resnet50 --task syntax --format jpeg2000 --epochs 50
```

---

## 📁 STRUKTURA DANYCH

```
dataset/
├── arcade/                    ← Oryginał (PNG)
└── compressed/
    ├── jpeg/                  ← JPEG (Q100, Q85, Q70, Q50, Q30, Q10)
    ├── jpeg2000/              ← JPEG2000 (te same jakości)
    └── avif/                  ← AVIF (te same jakości)
```

---

## 💡 INSTALACJA BIBLIOTEK

**Dla JPEG2000:**
```bash
pip install pillow  # JPEG2000 jest wspierany natywnie
```

**Dla AVIF:**
```bash
pip install pillow-avif-plugin
```

---

## 📅 TIMELINE (szacunkowy)

| Tydzień | Zadanie |
|---------|---------|
| 1 | Kompresja wszystkich formatów + pomiar jakości |
| 2-3 | Eksperyment A - JPEG, JPEG2000, AVIF |
| 4-5 | Eksperyment B - wszystkie formaty |
| 6 | Analiza porównawcza, wykresy, statystyki |
| 7-9 | Pisanie pracy magisterskiej |
| 10 | Korekty, prezentacja |

**Czas do ukończenia:** 8-10 tygodni

---

## 🏆 OCZEKIWANE WNIOSKI

1. **JPEG2000 vs JPEG:** Czy standard medyczny rzeczywiście lepszy?
2. **AVIF:** Czy najnowszy format przewyższa starsze?
3. **Rekomendacje:** Który format dla różnych zastosowań medycznych?
4. **Trade-off:** Rozmiar vs accuracy - optymalny punkt?

---

**Status:** Infrastruktura gotowa, obsługa 3 formatów zaimplementowana ✅
