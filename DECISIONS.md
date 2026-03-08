# Decyzje promotorskie i Status Projektu

## Decyzje z promotora

### 25.11.2025 - Inicjalny kontakt

- Wysłanie linku do repozytorium GitHub

### 16.02.2026 - Spotkanie z promotorem

**Decyzje:**

1. **EfficientNet-B0** - dodać do artykułu jako drugi model
2. **Analiza statystyczna** - p-value dla porównania formatów
3. **Podsumowanie liczebne** - tabela z wynikami
4. **Benchmark dataset** - dodać inny zbiór danych (medyczny lub nie)
5. **Feature maps analysis** - spectral entropy, Shannon entropy

**Uzasadnienie:**
> Za mało rzeczy do publikacji - potrzeba rozszerzenia eksperymentów

---

## Status Implementacji

### ✅ Zrobione

| Komponent | Status | Data |
|-----------|--------|------|
| Kompresja obrazów | ✅ Gotowe | JPEG, JPEG2000, AVIF |
| Pomiar jakości | ✅ Gotowe | PSNR, SSIM |
| Eksperyment A (syntax) | ✅ Zrobione | ResNet-50, JPEG/JPEG2000/AVIF |
| Eksperyment B (syntax) | ✅ Zrobione | ResNet-50, JPEG/JPEG2000/AVIF |

| Stenosis | ❌ Pominięty | Tylko 1 klas (nie nadaje się do klasyfikacji) |

### 🔴 Do Zrobienia

| Zadanie | Plik | Priorytet |
|---------|------|-----------|
| EfficientNet-B0 eksperymenty | `src/experiment_a.py`, `experiment_b.py` | Wysoki |
| Analiza statystyczna (p-value) | `src/statistical_analysis.py` | Wysoki |
| Feature maps (spectral entropy) | `src/feature_analysis.py` | Średni |
| Benchmark dataset | `src/benchmark_datasets.py` | Średni |
| Tabele do artykuł | - | Wysoki |
| Wykresy do artyku | - | Wysoki |

### 📝 Do omówienia z promotorem

1. **Segmentacja** - skupić się na segmentacji (nie tylko klasyfikacja)
2. **Poziomy kompresji** - sprawdzić akceptowane w stowarzyszeniach radiologii

:

- JPEG: Q=? (standard internetowy)
- JPEG2000: Q=? (standard medyczny DICOM)

3 **AVIF vs JPEG2000** - czy nowoczesny format przewyższa standard medyczny?

1. **Zbiór danych** - który benchmark? ISIC 2019? ChestX-ray14?

 CIFAR-10 (baseline non-medical)?

---

## Timeline

| Data | Kamień |
|------|-------|
| 25.11.2025 | Inicjalny kontakt z promotorem |
| 16.02.2026 | Spotkanie z decyzjami |
| TBD | Eksperymenty EfficientNet-B0 |
| TBD | Analiza statystyczna |
| TBD | Feature maps analysis |
| TBD | Benchmark dataset |
| TBD | Finalizacja artykułu |
