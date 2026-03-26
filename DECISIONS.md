# Decyzje promotorskie i Status Projektu

## Decyzje z promotora

### 25.11.2025 - Inicjalny kontakt

- Wysłanie linku do repozytorium GitHub

### 16.02.2026 - Spotkanie z promotorem

1. **EfficientNet-B0** - dodać do artykułu jako drugi model
2. **Analiza statystyczna** - p-value dla porównania formatów
3. **Podsumowanie liczebne** - tabela z wynikami
4. **Benchmark dataset** - dodać inny zbiór danych (medyczny lub nie) - TODO
5. **Feature maps analysis** - spectral entropy, Shannon entropy

> **Uzasadnienie:** Za mało rzeczy do publikacji - potrzeba rozszerzenia eksperymentów

---

## Status Projektu

| Zadanie | Status | Kod | Uwagi |
|---------|--------|-----|-------|
| Kompresja JPEG | ✅ Zrobione | ✅ | Wszystkie poziomy Q |
| Kompresja JPEG2000 | ✅ Zrobione | ✅ | Naprawiono błąd CR (pliki ~33KB) |
| Kompresja AVIF | ✅ Zrobione | ✅ | Wszystkie poziomy Q |
| Metryki jakości (PSNR/SSIM/CR) | ✅ Zrobione | ✅ | Wykresy PDF gotowe |
| Eksperyment A ResNet-50 - JPEG | ✅ Zrobione | ✅ | Wyniki w CSV |
| Eksperyment A ResNet-50 - AVIF | ✅ Zrobione | ✅ | Wyniki w CSV |
| Eksperyment A ResNet-50 - JPEG2000 | ⏳ W trakcie | ✅ | Uruchomiony na Colab |
| Eksperyment A EfficientNet-B0 - JPEG | ❌ Do odpalenia | ✅ | Kod gotowy, brak wyników |
| Eksperyment A EfficientNet-B0 - JPEG2000 | ❌ Do odpalenia | ✅ | Kod gotowy, brak wyników |
| Eksperyment A EfficientNet-B0 - AVIF | ❌ Do odpalenia | ✅ | Kod gotowy, brak wyników |
| Analiza statystyczna (p-value) | ❌ Do odpalenia | ✅ | Kod kompletny (888 linii), brak wyników EfficientNet |
| Feature maps analysis | ❌ Do odpalenia | ✅ | Kod kompletny (680 linii), są checkpointy |
| ARCADE dataset | ✅ Zrobione | ✅ | 3000 obrazów, 26 klas |
| ISIC 2019 (benchmark) | ❌ Opcjonalny | ✅ | Promoter wymaga |

**Legenda statusu:**

- ✅ Zrobione
- ❌ Do zrobienia / Do odpalenia
- 📊 Do analizy
- ⏳ W trakcie

**Legenda kodu:**

- ✅ Jest
- ❌ Nie ma
- ⚠️ Częściowo

---

## Komendy do odpalenia (Colab)

### EfficientNet-B0 - Eksperyment A (3 formaty, ~3-4h na GPU)

```bash
# JPEG
!python src/experiments/experiment_a.py --model efficientnet_b0 --task syntax --format jpeg --epochs 20 --batch-size 16

# JPEG2000
!python src/experiments/experiment_a.py --model efficientnet_b0 --task syntax --format jpeg2000 --epochs 20 --batch-size 16

# AVIF
!python src/experiments/experiment_a.py --model efficientnet_b0 --task syntax --format avif --epochs 20 --batch-size 16
```

### Analiza statystyczna (po EfficientNet)

```bash
# ResNet-50
!python src/analysis/statistical_analysis_corrected.py --experiment experiment_a --model resnet50 --task syntax

# EfficientNet-B0 (po uzyskaniu wyników)
!python src/analysis/statistical_analysis_corrected.py --experiment experiment_a --model efficientnet_b0 --task syntax
```

### Feature Maps Analysis (Spectral/Shannon Entropy)

```bash
# Przykłady dla ResNet-50 - analiza pojedynczych checkpointów
# Kod: 680 linii, zawiera spectral entropy, effective rank, stable rank, Shannon entropy

# JPEG Q=10
!python src/analysis/feature_analysis.py --model resnet50 --task syntax --experiment-id resnet50_syntax_q10_20260312_172310 --dataset arcade --max-batches 10

# JPEG Q=100 (baseline)
!python src/analysis/feature_analysis.py --model resnet50 --task syntax --experiment-id resnet50_syntax_q100_20260312_165431 --dataset arcade --max-batches 10

# AVIF Q=50
!python src/analysis/feature_analysis.py --model resnet50 --task syntax --experiment-id resnet50_syntax_q50_avif_20260312_204030 --dataset arcade --max-batches 10

# JPEG2000 Q=50
!python src/analysis/feature_analysis.py --model resnet50 --task syntax --experiment-id resnet50_syntax_q50_jpeg2000_20260312_190507 --dataset arcade --max-batches 10
```

---
