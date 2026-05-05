---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 23px;
    padding: 50px 70px;
    background: linear-gradient(135deg, #f4f1ea 0%, #e8e2d4 100%);
    color: #2b2b2b;
    text-align: justify;
  }
  section.lead {
    text-align: center;
    background: linear-gradient(135deg, #14213d 0%, #1f3a68 50%, #2d5aa0 100%);
    color: #fff;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
  }
  section.lead h1 {
    color: #fff;
    font-size: 44px;
    line-height: 1.25;
    max-width: 90%;
    margin: 0 auto 0.5em auto;
    border-bottom: 3px solid #d4a857;
    padding-bottom: 18px;
    text-shadow: 0 2px 6px rgba(0,0,0,0.35);
  }
  section.lead .subtitle {
    color: #d4a857;
    font-size: 22px;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 1em;
    font-weight: 600;
  }
  section.lead .author {
    color: #fff;
    font-size: 30px;
    font-weight: 700;
    margin: 0.4em 0 0.2em 0;
  }
  section.lead .meta {
    color: #c9d4e6;
    font-size: 20px;
    line-height: 1.7;
    margin-top: 1.2em;
  }
  section.lead .meta strong { color: #fff; }
  h1 { color: #1f3a68; border-bottom: 3px solid #b08a4a; padding-bottom: 8px; }
  h2 { color: #1f3a68; }
  h3 { color: #2d5aa0; margin-top: 0.4em; }
  table {
    font-size: 19px;
    margin: 0 auto;
    border-collapse: collapse;
    width: 95%;
    background: #fffdf8;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
  }
  th {
    background: #1f3a68;
    color: #fff;
    padding: 8px 12px;
  }
  td {
    padding: 6px 12px;
    border-bottom: 1px solid #d8d2c2;
    text-align: center;
  }
  td:first-child { text-align: left; font-weight: 600; }
  code {
    font-size: 17px;
    background: #fffdf8;
    padding: 1px 5px;
    border-radius: 3px;
    border: 1px solid #d8d2c2;
  }
  blockquote {
    border-left: 4px solid #b08a4a;
    background: #fffdf8;
    padding: 10px 18px;
    margin: 12px 0;
    color: #1f3a68;
    font-style: italic;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  }
  ul, ol { line-height: 1.55; }
  strong { color: #1f3a68; }
  section::after {
    color: #6b7785;
    font-size: 14px;
  }
  .cols { display: flex; gap: 30px; }
  .cols > div { flex: 1; }
  section.center {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: stretch !important;
  }
  section.center > * { width: 100%; }
  section.center h2 { text-align: center; }
  section.center table { margin: 24px auto; }
  section.center blockquote { max-width: 90%; margin-left: auto; margin-right: auto; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Wpływ kompresji obrazu na skuteczność modeli głębokiego uczenia w diagnostyce kardiologicznej

<div class="subtitle">Praca magisterska</div>

<div class="author">Jakub Burek</div>

<div class="meta">
Promotor: <strong>dr Ilona Urbaniak</strong><br/>
Kwiecień 2026
</div>

---

## Plan prezentacji

1. **Motywacja i cel pracy** — dlaczego kompresja w obrazowaniu medycznym to nietrywialny problem.
2. **Zagadnienia teoretyczne** — formaty kompresji (JPEG / JPEG2000 / AVIF) oraz wybrane architektury sieci CNN.
3. **Metodyka** — projekt dwóch komplementarnych eksperymentów (A i B), zbiory danych, modele, metryki.
4. **Implementacja** — pipeline i organizacja kodu.
5. **Wyniki** — co pokazują liczby z Eksperymentu A.
6. **Diagnoza problemu z datasetem** — co się stało i dlaczego accuracy jest niska.
7. **Status, wnioski i plany dalszej pracy.**

> Cel prezentacji: pokazać, że **infrastruktura badawcza jest gotowa**, a kolejne kroki są precyzyjnie zaplanowane.

---

## 1. Motywacja

Współczesne obrazowanie medyczne — angiografia wieńcowa, CT, MRI, dermoskopia — generuje **terabajty danych dziennie**. Systemy szpitalne (PACS) oraz platformy telemedyczne muszą te dane przechowywać i przesyłać, co bez kompresji jest praktycznie niemożliwe.

Kompresja stratna pozwala zmniejszyć rozmiar pliku nawet **10–20×**, ale **usuwa drobne szczegóły** — a właśnie te szczegóły mogą decydować o:

- **diagnozie lekarza** (np. mikrozmiany w naczyniach wieńcowych),
- **predykcji modelu AI** wspomagającego specjalistę.

> W literaturze medycznej brakuje twardej odpowiedzi na pytanie: *do jakiego poziomu jakości Q można skompresować obraz, zanim model głębokiego uczenia zacznie się systematycznie mylić?*

---

## Cel pracy

> **Wyznaczenie dopuszczalnego poziomu kompresji stratnej dla diagnostyki kardiologicznej wspomaganej AI oraz sformułowanie rekomendacji praktycznych dla systemów PACS i platform telemedycznych.**

### Cele szczegółowe

- **Porównanie trzech formatów kompresji:** JPEG (standard internetowy), JPEG2000 (standard DICOM), AVIF (nowoczesny format z 2019 r.).
- **Ocena dwóch architektur CNN:** ResNet-50 (klasyk) i EfficientNet-B0 (efektywny obliczeniowo, realistyczny dla edge / PACS).
- **Dwa scenariusze użycia (Eksperyment A i B):** odpowiadające realnym sytuacjom — trening na archiwalnych skompresowanych danych vs. inferencja na obrazach przesłanych przez wąskie łącze.
- **Feature maps analysis** — sprawdzenie *co dzieje się w środku* sieci przy wysokiej kompresji (spectral entropy, Shannon entropy, effective rank).
- **Walidacja na drugim zbiorze medycznym** (ISIC 2019, dermoskopia) dla pokazania generalizacji wniosków.

---

## 2. Formaty kompresji — porównanie

| Format | Rok | Algorytm | Charakterystyka | Rola w pracy |
|---|---|---|---|---|
| **JPEG** | 1992 | DCT 8×8 + kwantyzacja | Powszechny, artefakty blokowe | Baseline, punkt odniesienia |
| **JPEG2000** | 2000 | Wavelet (DWT) | Standard **DICOM** w medycynie | Punkt odniesienia kliniczny |
| **AVIF** | 2019 | AV1 intra-frame | Najlepszy stosunek jakość/rozmiar | Format „nowej generacji" |

### Zakres testowanych jakości

- **Pełen zakres badawczy:** Q ∈ {100, 95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10}
- **Wersja MVP do artykułu:** Q ∈ {100, 85, 70, 50, 30, 10} — sześć poziomów wystarczy do uchwycenia trendu, a redukuje koszt obliczeniowy ~2×.

---

## Architektury sieci CNN

<div class="cols">
<div>

### ResNet-50

- Klasyk z 2015 r., **residual connections** rozwiązujące problem zanikającego gradientu.
- ~**25 mln parametrów**.
- Powszechny benchmark w *medical imaging*.
- W tej pracy: **referencja** dla porównań.

</div>
<div>

### EfficientNet-B0

- *Compound scaling* (Tan & Le, 2019) — równoczesne skalowanie głębokości, szerokości i rozdzielczości.
- ~**5 mln parametrów** (5× mniej niż ResNet-50).
- Najlepszy stosunek **dokładność / koszt obliczeniowy**.
- W tej pracy: **realistyczny model** dla wdrożeń edge / PACS.

</div>
</div>

> Oba modele wstępnie wytrenowane na ImageNet, fine-tuning na zbiorze ARCADE.

---

## 3. Metodyka — dwa komplementarne eksperymenty

### Eksperyment A — *„Czy mogę trenować na skompresowanych danych?”*
- **Train:** obrazy skompresowane różnymi formatami i poziomami Q.
- **Test:** obrazy oryginalne (Q = 100).
- Pytanie badawcze: czy można oszczędzić miejsce w archiwum treningowym **bez utraty jakości modelu**?

### Eksperyment B — *„Czy mogę przesyłać skompresowane dane do gotowego modelu?”*
- **Train:** obrazy oryginalne (najwyższa jakość).
- **Test:** obrazy skompresowane różnymi formatami i poziomami Q.
- Pytanie badawcze: czy model wytrenowany w idealnych warunkach pozostaje **odporny** na kompresję podczas inferencji? Scenariusz typowy dla **telemedycyny** i mobilnych platform diagnostycznych.

---

## Zbiory danych

### ARCADE — angiografia wieńcowa (główny zbiór)
- Task **syntax** = klasyfikacja **26 segmentów tętnic** wg klinicznej skali SYNTAX.
- 3 000 obrazów (split: train / val / test).
- **Wybrany jako case study kardiologiczny** — bezpośrednio związany z tematem pracy.

### ISIC 2019 — dermoskopia (benchmark, wymóg promotora)
- 8 klas zmian skórnych (czerniak, znamiona, raki podstawnokomórkowe…).
- Zupełnie inna domena medyczna → pokazuje, czy wnioski **generalizują** poza kardiologię.

> Dla obu zbiorów generujemy **trzy wersje skompresowane** (JPEG, JPEG2000, AVIF) na każdym poziomie Q. Łącznie kilkadziesiąt wariantów per zbiór.

---

<!-- _class: center -->

## Hiperparametry — wspólne dla wszystkich eksperymentów

| Parametr | Wartość | Uzasadnienie |
|---|---|---|
| Optymalizator | **Adam** | Standard dla fine-tuningu CNN |
| Learning rate | **1e-4** | Niski LR — fine-tuning, nie trening od zera |
| Weight decay | **1e-4** | Regularyzacja L2 |
| Batch size | **16** | Kompromis między VRAM a stabilnością |
| Scheduler | **CosineAnnealingLR** | Płynne wygaszanie LR |
| Early stopping | **10 epok** | Ochrona przed overfittingiem |
| Mixed precision | **AMP ✅** | ~2× szybszy trening na GPU |
| Random seed | **42** | Powtarzalność wyników |

> **36 niezależnych trenowań** już ukończonych: 2 modele × 3 formaty × 6 poziomów Q.

---

## 4. Pipeline implementacyjny

```
src/
├── core/                   # podstawa systemu
│   ├── dataset.py          #   loader ARCADE
│   ├── isic_dataset.py     #   loader ISIC 2019
│   └── train.py            #   pętla treningowa + AMP + early stopping
├── processing/             # przygotowanie danych
│   ├── compress_images.py  #   kompresja ARCADE (JPEG/JP2/AVIF)
│   ├── compress_isic.py    #   kompresja ISIC
│   └── measure_quality.py  #   PSNR / SSIM / Compression Ratio
├── experiments/
│   ├── experiment_a.py     #   train na skompresowanych
│   └── experiment_b.py     #   test na skompresowanych
└── analysis/
    ├── statistical_analysis_corrected.py   # paired t-test, Wilcoxon
    ├── feature_analysis.py                 # spectral / Shannon entropy
    └── generate_tables_plots.py            # tabele i wykresy do artykułu
```

> Po refaktoringu: **20 → 12 plików Python** (~40% redukcji kodu, brak duplikacji logiki).

---

## Pomiar jakości obrazu

Dla **każdego** skompresowanego pliku liczone są trzy metryki:

- **PSNR** (Peak Signal-to-Noise Ratio, dB) — globalny błąd rekonstrukcji. Im więcej, tym wierniej.
- **SSIM** (Structural Similarity Index, 0–1) — podobieństwo strukturalne; bliżej percepcji ludzkiej niż PSNR.
- **Compression Ratio (CR)** = `rozmiar_oryginał / rozmiar_skompresowany` — efektywność formatu.

### Co dają mapy CR per-obraz?

CR mapy są zapisywane w `dataset/cr_maps/syntax_*_cr_map.json` osobno dla **każdego pliku**. Dzięki temu mamy **pełną dystrybucję CR**, a nie tylko średnią — co pozwala wykryć anomalie (np. obrazy, dla których kompresja praktycznie nie działa).

> Właśnie ta analiza ujawniła problem ze słabą efektywnością JPEG2000 (omówiony dalej).

---

## 5. Wyniki — Eksperyment A · ResNet-50

Test accuracy (test na oryginałach, train na skompresowanych):

| Q | JPEG | JPEG2000 | AVIF |
|---|---|---|---|
| 100 | 18.3% | 20.7% | 21.7% |
| 85  | 20.0% | 20.7% | 20.7% |
| 70  | 20.3% | 21.0% | 22.3% |
| 50  | 21.3% | 17.7% | **22.7%** |
| 30  | 21.7% | 17.3% | 16.7% |
| 10  | 17.3% | 18.0% | 22.3% |

**Obserwacja:** wahania ±4 p.p. na całym zakresie Q, brak systematycznego trendu spadkowego. To **nie** błąd metodyki kompresji — to sygnał, że klasyfikator **nie nauczył się zadania** (diagnoza dalej).

---

## Wyniki — Eksperyment A · EfficientNet-B0

| Q | JPEG | JPEG2000 | AVIF |
|---|---|---|---|
| 100 | 17.7% | 17.3% | 18.0% |
| 85  | 19.7% | 17.3% | 21.3% |
| 70  | 18.7% | 20.7% | 19.3% |
| 50  | 18.0% | 21.3% | 18.7% |
| 30  | 19.7% | 15.3% | 18.7% |
| 10  | 16.7% | 17.7% | 18.0% |

**Identyczny obraz** jak dla ResNet-50: brak monotonicznego spadku z Q, wszystkie wyniki w przedziale **15–22%**. Skoro dwie różne architektury zachowują się tak samo, bottleneckiem nie jest model — jest nim **zbiór danych** (wyniki w obrębie szumu klasyfikatora losowego).

---

## 6. Diagnoza problemu (26.03.2026)

Po wytrenowaniu wszystkich 36 modeli i analizie wyników przeprowadziłem audyt zbioru ARCADE / task **syntax**:

🚨 **Trzy zidentyfikowane problemy:**

1. **Dwie klasy nie istnieją w zbiorze treningowym** (klasa **11** i **25**). Sieć fizycznie nie ma jak nauczyć się ich rozpoznawać, a ich obecność w zbiorze testowym obniża accuracy.
2. **18-krotny imbalance między klasami** — najliczniejsza klasa ma 108 próbek, najmniejsza zaledwie 6. Bez class weights / oversamplingu model uczy się głównie klas dominujących.
3. **JPEG2000 ma efektywny CR ≈ 1.10×** — przy zastosowanych ustawieniach format praktycznie nie redukuje rozmiaru pliku, co czyni porównanie z JPEG/AVIF nieadekwatnym.

**Wniosek:** sieć nie miała szans nauczyć się klasyfikacji. Wahania ±4 p.p. są **w obrębie szumu losowego**, a nie efektem kompresji.

📌 *Status: oczekuję na konsultację z promotorem co do strategii naprawczej.*

---

## Co już zrobione ✅

### Infrastruktura badawcza
- Pełna implementacja **kompresji** (JPEG / JPEG2000 / AVIF) i metryk **PSNR / SSIM / CR**.
- Pipeline treningowy z **AMP, early stopping, CosineLR**.
- Loadery dla **ARCADE** i **ISIC 2019**.
- Refactor: 20 → 12 plików Python (~40% redukcji).

### Eksperymenty i kod analityczny
- **Eksperyment A: 36 trenowań ukończonych**, wszystkie checkpointy zapisane.
- 6 plików CSV z wynikami (ResNet-50 i EfficientNet-B0 × JPEG/JP2/AVIF).
- Kod **analizy statystycznej** (paired t-test, Wilcoxon dla Eksp. B) — 888 linii, gotowy do uruchomienia.
- Kod **feature maps analysis** (spectral entropy, Shannon entropy, effective rank, stable rank) — 680 linii, gotowy.

### Dokumentacja
- `STATUS.md`, `DECISIONS.md` — pełen log decyzji projektowych i postępu.

---

## Co jeszcze do zrobienia 🚧

### Priorytet krytyczny — po konsultacji z promotorem
1. **Naprawić problem z ARCADE** — usunąć / zmergować klasy 11 i 25, wprowadzić *class weights* lub oversampling, ewentualnie zmienić task (np. `stenosis` zamiast `syntax`).
2. **Powtórzyć Eksperyment A** na poprawionym zbiorze.
3. **Uruchomić Eksperyment B** (train na oryginałach → test na skompresowanych) dla obu modeli i wszystkich formatów.

### Wymagane do artykułu i obrony
4. **Analiza statystyczna** wyników — p-value, paired t-test, Wilcoxon. Kod gotowy.
5. **Feature maps analysis** dla wybranych checkpointów (Q=100 vs Q=50 vs Q=10).
6. **Walidacja na ISIC 2019** — replikacja wniosków na drugim zbiorze medycznym.
7. **Tabele i wykresy** do artykułu (`generate_tables_plots.py`).
8. **Aktualizacja artykułu** — obecna wersja jest nieaktualna względem stanu projektu.

---

## Plan dalszej pracy — kolejność i szacunki

1. ✏️ **Spotkanie z promotorem** — decyzja o strategii naprawy zbioru ARCADE.  *(1 tydzień)*
2. 🔁 **Re-run Eksperymentu A** na poprawionym zbiorze.  *(~3–4 dni GPU)*
3. 🆕 **Eksperyment B** — pełen przebieg dla obu modeli i 3 formatów.  *(~3–4 dni GPU)*
4. 📊 **Analiza statystyczna + feature maps** (Q=100 vs Q=50 vs Q=10).  *(~2 dni)*
5. 🧪 **ISIC 2019** — replikacja wyników na drugim zbiorze.  *(~3 dni)*
6. 📝 **Aktualizacja artykułu** + wykresy końcowe.  *(~1 tydzień)*
7. 📄 **Finalizacja pracy magisterskiej** i przygotowanie do obrony.

> **Bilans:** ~75% projektu zrobione (cała infrastruktura, kod, pierwsza tura eksperymentów). ~25% pozostało — głównie powtórzenie eksperymentów po naprawie zbioru i napisanie analizy końcowej. **Szacowany czas do ukończenia: 4–6 tygodni intensywnej pracy.**

---

<!-- _class: lead -->

# Dziękuję za uwagę
