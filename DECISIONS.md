# Decyzje promotorskie i Status Projektu

## Decyzje z promotora

### 25.11.2025 - Inicjalny kontakt
- Wysłanie linku do repozytorium GitHub

### 16.02.2026 - Spotkanie z promotorem
1. **EfficientNet-B0** - dodać do artykułu jako drugi model
2. **Analiza statystyczna** - p-value dla porównania formatów
3. **Podsumowanie liczebne** - tabela z wynikami
4. **Benchmark dataset** - dodać inny zbiór danych (medyczny lub nie)
5. **Feature maps analysis** - spectral entropy, Shannon entropy

> **Uzasadnienie:** Za mało rzeczy do publikacji - potrzeba rozszerzenia eksperymentów

---

## Status Implementacji

### ✅ Zrobione
| Komponent | Status |
|-----------|--------|
| Kompresja obrazów (JPEG, JPEG2000, AVIF) | ✅ |
| Pomiar jakości (PSNR, SSIM) | ✅ |
| Eksperyment A/B - ResNet-50, syntax | ✅ |
| **ISIC 2019 - kod** | ✅ Zaimplementowany |
| Stenosis | ❌ Pominięty (tylko 1 klasa) |

### 🔴 Do Zrobienia
| Zadanie | Priorytet |
|---------|-----------|
| EfficientNet-B0 eksperymenty | Wysoki |
| Analiza statystyczna (p-value) | Wysoki |
| Tabele i wykresy do artykułu | Wysoki |
| **ISIC 2019 - pobranie i uruchomienie** | Wysoki |
| Feature maps (spectral entropy) | Średni |

---

## Workflow Agentów

| Faza | Główny agent | Agent wspierają |------|---------|-------------------|
| **Pisanie kodu** | `ml-code-reviewer` | `config-manager`, `performance-optimizer` |
| **Odpalenie kodu** | `experiment-runner` | `data-quality-monitor` |
| **Analiza wyników** | `results-analyzer` | `anomaly-detector` |
| **Pisanie artykuł** | `thesis-writer` | `visualization-generator`, `literature-reviewer` |

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
