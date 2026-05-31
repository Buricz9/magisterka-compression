# Kwestie metodologiczne do konsultacji z promotorem

**Projekt:** Wpływ kompresji obrazu (JPEG / JPEG2000 / AVIF) na klasyfikację
multi-label segmentów tętnic wieńcowych (zbiór ARCADE, 26 klas SYNTAX Score).

**Data utworzenia:** 2026-05-21

**Cel dokumentu.** Zebranie w jednym miejscu kwestii metodologicznych, które
wymagają decyzji lub akceptacji promotora. Punkty wynikają z przeglądu kodu
potoku analitycznego oraz z pierwszych uruchomień eksperymentu. Dokument jest
uzupełniany na bieżąco — kolejne tematy dopisywane są na końcu.

**Status ogólny.** Żaden z poniższych punktów nie jest błędem obliczeniowym —
dane i kod są poprawne. Są to wybory metodologiczne i sposoby prezentacji
wyników, które warto uzgodnić, aby praca była spójna i odporna na uwagi.

**Aktualizacja (2026-05-29): decyzje podjęte.** Wszystkie 9 punktów zostało
rozstrzygniętych (sekcje **DECYZJA** poniżej) i wcielonych w kod oraz artykuł.
Skrót decyzji:

| # | Kwestia | Decyzja |
|---|---------|---------|
| 1 | Schemat testów | **Przyjęto** sparowany/blokowy (Friedman + paired t/Wilcoxon + Holm) |
| 2 | Metryka wiodąca | **Przyjęto** mAP; bez strojenia progu; F1/Hamming pomocnicze |
| 3 | n=1 vs replikacja | **Przyjęto n=1** z uczciwym ujęciem („brak dowodu na efekt" + TOST); replikacja = praca przyszła |
| 4 | Q=100 poza matched-CR | **Przyjęto**: porównanie formatów na Q=10–95, Q=100 informacyjnie |
| 5 | Ogon JPEG2000 | Mediana potwierdza ranking; zdanie-caveat w tekście (pełne IQR zbędne) |
| 6 | Klasy puste/rzadkie | **Przyjęto** maskowanie klas zerowych; opis jako ograniczenie zbioru |
| 7 | Granica Δ dla TOST | **Δ = 0.02 mAP** główne (≈2× szum), sensitivity 0.01/0.015; zakres Q=10–95 |
| 8 | PSNR/SSIM a skuteczność | Test korelacji **dodany**; teza dopuszczalna jako wynik (2 architektury) z caveatem n=1 |
| 9 | Tendencja AVIF | Raportować jako „możliwa drobna, nieistotna tendencja, nieodróżnialna od szumu" |

---

## 1. Zmiana metodyki analizy statystycznej (testy sparowane zamiast niesparowanych)

**Kontekst.** Dotychczasowa analiza porównywała formaty testem t-Studenta dla
prób niezależnych, jednoczynnikową ANOVA i testem Kruskala-Wallisa, traktując
13 poziomów jakości Q jako niezależną próbę IID. Jest to niezgodne z konstrukcją
badania: poziomy Q są **wspólne** dla wszystkich trzech formatów (ta sama
siatka), czyli stanowią czynnik blokujący — obserwacje są **sparowane** względem
Q. (README projektu odnotowywał już, że „testy t-test/ANOVA na 13 punktach Q
jako próbie IID są dyskusyjne".)

**Co zrobiono.** Analizę przepisano na poprawny schemat sparowany/blokowy:
- test omnibus: **test Friedmana** (formaty = warianty, poziomy Q = bloki)
  z miarą efektu **Kendall's W**;
- porównania parami: **sparowany test t** oraz **test kolejności par Wilcoxona**;
- korekta na wielokrotne porównania: **Holm-Bonferroni**.

**Pytanie do promotora.** Czy akceptuje Pani to podejście jako docelowe dla
artykułu? Alternatywa prostsza: rezygnacja z testów istotności i raportowanie
wyłącznie statystyk opisowych (średnia ± odchylenie standardowe po Q).

**DECYZJA (przyjęta).** Schemat sparowany/blokowy jest jedynym poprawnym dla tej
konstrukcji (poziomy Q wspólne dla formatów). Przyjmujemy go jako docelowy:
Friedman + Kendall's W (omnibus), sparowany t-test i Wilcoxon (pary),
Holm-Bonferroni (korekta). Statystyki opisowe raportujemy dodatkowo, nie zamiast
testów. *Zaimplementowane i wykonane.*

---

## 2. Metryka wiodąca: mAP zamiast F1

**Kontekst.** Model trenowany jest z funkcją straty `BCEWithLogitsLoss` i wagami
klas `pos_weight` (do ~999 dla klas skrajnie rzadkich). Wagi te radzą sobie
z imbalansem, ale **przesuwają optymalny próg decyzyjny** z domyślnego 0.5.
F1 liczone przy stałym progu 0.5 jest więc systematycznie nieoptymalne.

**Co zrobiono.** Jako metrykę wiodącą przyjęto **mAP** (mean Average Precision):
jest bezprogowa, niezależna od kalibracji progu i standardowa dla klasyfikacji
multi-label przy silnym imbalansie. F1 oraz Hamming accuracy pozostają jako
metryki pomocnicze.

**Pytanie do promotora.** Czy akceptuje Pani mAP jako metrykę główną w artykule?
Alternatywa: dobór progu decyzyjnego per klasa na zbiorze walidacyjnym —
poprawiłby wiarygodność F1, ale wymaga dodatkowej procedury i jest sam w sobie
dyskusyjny przy bardzo małych zbiorach walidacyjnych dla klas rzadkich.

**DECYZJA (przyjęta).** mAP jako metryka wiodąca — bezprogowa, standardowa dla
multi-label przy silnym imbalansie. **Strojenia progu per klasa NIE wprowadzamy**:
przy klasach skrajnie rzadkich (id=12: 6 próbek walidacyjnych) kalibracja progu
jest niewiarygodna i dokłada wątpliwy stopień swobody. F1\@0.5 i Hamming
raportujemy jako pomocnicze, jawnie oznaczone jako progowe. *Wcielone (raporty
oznaczają metrykę PRIMARY/SECONDARY).*

---

## 3. Brak replikacji — n = 1 na warunek (format × Q)

**Kontekst.** Dla każdej pary (format, poziom Q) trenowany jest **jeden** model
(jedno ziarno losowe). Testy statystyczne operują więc na 13 punktach Q jako
próbie. Oznacza to ograniczoną moc statystyczną i brak oszacowania zmienności
wynikającej z losowości samego treningu.

Dodatkowo trening **nie jest w pełni bitowo-reprodukowalny**: ustawiane są ziarna
wszystkich generatorów (random / numpy / torch / CUDA) oraz `cudnn.deterministic`,
ale nie jest wywoływane `torch.use_deterministic_algorithms(True)` — część
operacji na GPU pozostaje niedeterministyczna, zwłaszcza przy treningu w trybie
mieszanej precyzji (AMP). Powtórzenie tego samego treningu może więc dać nieco
inny model. (Same metryki liczone z gotowych checkpointów są w pełni
odtwarzalne — ewaluacja jest deterministyczna.) Konfiguracji tej świadomie nie
zmieniamy w trakcie badania, aby wszystkie przebiegi — baseline i Eksperyment A —
działały w identycznych warunkach.

**Dowód empiryczny (pierwszy pełny przebieg — ResNet-50 / JPEG, 13 poziomów Q).**
Wynik okazał się statystycznie płaski: korelacja Q↔mAP = 0.24 (Pearson p≈0.43,
Spearman p≈0.33 — nieistotne), nachylenie regresji ≈ +0.005 mAP na całym zakresie
Q=10–100, a odchylenie standardowe mAP po 13 poziomach Q wynosi ≈0.007. Wynik
baseline (trening+test na PNG, mAP=0.655) leży **wewnątrz** rozrzutu wyników
skompresowanych (0.640–0.666), a poziomy Q=60 i Q=75 baseline **przewyższają** —
co jest przyczynowo niemożliwe jako realny efekt kompresji i dowodzi, że
obserwowana zmienność Q-do-Q jest szumem treningu pojedynczego ziarna, nie
sygnałem. Praktyczny wniosek: jeżeli efekt kompresji JPEG istnieje, jest mniejszy
niż ~1–2 pp — **poniżej progu wykrywalności przy n=1**. Płaski wynik należy więc
raportować jako „brak dowodu na efekt", a NIE „dowód braku efektu".

Potwierdzenie międzyarchitekturowe (po ukończeniu 5/6 przebiegów): kierunek
korelacji jakość→skuteczność zmienia znak między architekturami (ResNet-50:
dodatni; EfficientNet-B0: ujemny — oba nieistotne), a „anomalie" przy tych samych
poziomach Q (np. Q=75, Q=80) odchylają się w **przeciwnych** kierunkach w obu
architekturach (zgodność znaku różnicy formatów tylko 5/13 — na poziomie losowym).
Gdyby zmienność Q-do-Q była realnym efektem kompresji, byłaby spójna między
architekturami; jej brak spójności jest niezależnym dowodem, że to szum
pojedynczego ziarna. EfficientNet-B0 ma przy tym wyraźnie większą wariancję
wyników niż ResNet-50 (silniejsze przeuczenie, luka train/val ~15–25 pp), co
dodatkowo wspiera tę interpretację.

**Pytanie do promotora.** Czy przyjmujemy schemat n=1 i opisujemy go jako
ograniczenie pracy (formułując wnioski wyłącznie jako „brak dowodu na efekt",
ewentualnie z testem równoważności TOST), czy powtarzamy trening dla kilku ziaren
losowych na warunek (np. 3–5), aby oszacować zmienność wewnątrz-warunkową i
odzyskać moc do wykrycia efektu rzędu 1–2 pp? Wariant z replikacją istotnie
zwiększa wiarygodność, ale mnoży czas obliczeń ~3–5×.

**DECYZJA (przyjęta).** Dla pracy magisterskiej przyjmujemy **schemat n=1** z
uczciwym ujęciem. Uzasadnienie: (a) wynik główny to *równoważność/null* — jest on
mocniejszy poznawczo, gdy dowód „braku efektu" oparty jest na TOST, a nie na
brakującej mocy; (b) zbieżne dowody (baseline wewnątrz rozrzutu, zmiana znaku
korelacji między architekturami) jednoznacznie wskazują, że ewentualny efekt
< szum ziarna; (c) replikacja 3–5× to dni obliczeń, które nie zmieniłyby wniosku
jakościowego. Wnioski formułujemy jako **„brak dowodu na efekt"** + równoważność
TOST, a n=1 opisujemy jako kluczowe ograniczenie. Replikacja wieloziarnowa =
rekomendowana **praca przyszła** (jedyny sposób rozstrzygnięcia tendencji AVIF
~0.5 pp — pkt 9). Determinizmu treningu świadomie nie zmieniamy w trakcie badania
(spójność wszystkich przebiegów). *Wcielone w sekcję Ograniczenia.*

---

## 4. Punkt Q=100 poza reżimem dopasowanej kompresji (matched-CR)

**Kontekst.** Metodyka zakłada porównanie formatów przy **dopasowanym**
współczynniku kompresji (matched-CR). Dla Q=10–95 jest to spełnione — CR zgodny
między formatami w granicach ~3%. Dla **Q=100** JPEG2000 i AVIF mają fizyczną
dolną granicę kompresji stratnej i nie osiągają tak dużego pliku jak JPEG;
przy Q=100 trzy formaty NIE są przy równym CR.

**Konsekwencja.** Wiersz Q=100 nie jest uczciwym porównaniem formatów — JPEG ma
tam wyższe PSNR głównie dlatego, że słabiej skompresował obraz, a nie dlatego,
że jest lepszym kodekiem.

**Propozycja.** Ograniczyć ranking formatów do zakresu Q=10–95, a punkt Q=100
raportować jedynie informacyjnie, z wyraźną adnotacją „poza reżimem matched-CR".

**Pytanie do promotora.** Czy akceptuje Pani takie ujęcie?

**DECYZJA (przyjęta).** Tak. Główne porównanie formatów odnosimy do reżimu
matched-CR (Q=10–95); Q=100 pokazujemy informacyjnie z adnotacją. Zweryfikowano,
że wykluczenie Q=100 **nie zmienia wniosku**: Friedman dla mAP daje p=0.37
(ResNet-50) i p=0.34 (EfficientNet-B0) na Q=10–95, a równoważność TOST przy
Δ=0.015/0.02 utrzymuje się dla wszystkich par. *Wcielone w tekst (adnotacja
+ nota o odporności wniosku).*

---

## 5. JPEG2000 — ogon trudnych obrazów w metrykach jakości

**Kontekst.** Na ~2% obrazów o wysokiej entropii (dużo drobnej tekstury
wysokoczęstotliwościowej) falka 9/7 JPEG2000 wypada wyraźnie słabiej. Zaniża to
nieco **średnie** PSNR/SSIM dla JPEG2000, choć dla ~98% obrazów format zachowuje
się dobrze. Jest to znana cecha kodeka, nie błąd potoku.

**Propozycja.** W tabelach raportować **medianę i rozstęp międzykwartylowy
(IQR)** obok średniej — mediana jest odporna na ten ogon i lepiej oddaje typowe
zachowanie formatu.

**Pytanie do promotora.** Czy taki sposób raportowania jest odpowiedni?
(To kwestia wyłącznie prezentacji — dane są poprawne.)

**DECYZJA (przyjęta, w wersji minimalnej).** Ogon dotyczy ~2% obrazów i **nie
zmienia uporządkowania formatów** (AVIF ≳ JPEG2000 > JPEG zachodzi zarówno na
średnich, jak i medianach PSNR/SSIM). Dlatego w~tabelach pozostawiamy średnie
(spójne z literaturą kodeków), a~ogon JPEG2000 odnotowujemy jednym zdaniem-caveat
w~omówieniu jakości. Pełne tabele mediana+IQR uznajemy za zbędne dla wniosków.
*Caveat dodany w~sekcji „Jakość kompresji".*

---

## 6. Klasy puste i skrajnie rzadkie w zbiorze ARCADE

**Kontekst.** W zbiorze ARCADE/Syntax:
- klasa id=26 ma **0 przykładów pozytywnych we wszystkich trzech splitach** —
  F1 i AP są dla niej nieokreślone;
- klasa id=12 jest skrajnie rzadka (1 przykład w train, 6 w val, 0 w test).

**Co zrobiono.** Metryki (F1, AP) liczone są z **pominięciem klas bez przykładów
pozytywnych** w danym splicie; raportowana jest liczba faktycznie obecnych klas
(`n_classes_present`).

**Pytanie do promotora.** Czy akceptuje Pani takie podejście (pominięcie klas
zerowych) oraz opis tego jako ograniczenia zbioru danych?

**DECYZJA (przyjęta).** Tak. Maskowanie klas bez przykładów pozytywnych w~danym
splicie (z~raportowaniem `n_classes_present`) jest standardowym, poprawnym
podejściem — F1/AP są dla takich klas nieokreślone i~ich wymuszanie zaniżałoby
metryki sztucznie. Opisujemy to jako ograniczenie **zbioru danych** (a~nie metody)
w~sekcji Ograniczenia. *Wcielone.*

---

## 7. Granica równoważności (Δ) dla testu równoważności TOST

**Kontekst.** Pierwsze porównanie formatów (ResNet-50, JPEG vs JPEG2000, 13
poziomów Q) daje średnią różnicę mAP ≈ 0 (sparowany test t p≈0.92, Wilcoxon
p≈0.79) — formaty są nieodróżnialne. Aby zaraportować to jako **pozytywną
równoważność** („formaty są praktycznie równoważne"), a nie tylko jako „brak
istotnej różnicy" (co myli brak efektu z brakiem mocy — por. pkt 3), stosuje się
test równoważności **TOST**. TOST wymaga jednak z góry zadanej **granicy
równoważności Δ** — progu różnicy uznawanej za praktycznie nieistotną. Przy
Δ = ±0.01 mAP TOST daje p≈0.024 (równoważność potwierdzona).

**Problem.** Granica Δ musi być uzasadniona merytorycznie (próg klinicznie lub
praktycznie nieistotnej różnicy w skuteczności), a NIE dobrana po obejrzeniu
danych — inaczej test jest bezwartościowy. Dodatkowo wynik TOST jest wrażliwy na
zakres: z punktem Q=100 p≈0.024, bez Q=100 p≈0.044 (Q=100 jest poza reżimem
matched-CR — por. pkt 4 — więc porównanie formatów uczciwie liczyć na Q=10–95).

**Pytanie do promotora.** Jaką granicę równoważności Δ przyjąć dla testu TOST i na
jakiej podstawie (literatura / wymaganie kliniczne / arbitralny próg z
uzasadnieniem)? Czy porównanie formatów raportować na zakresie Q=10–95
(z wyłączeniem niedopasowanego pod względem CR punktu Q=100)?

**DECYZJA (przyjęta).** Przyjmujemy **Δ = 0.02 mAP (2 pp)** jako główną granicę
równoważności, z~uzasadnieniem zdefiniowanym *a priori* (niezależnym od wyniku
porównania): jest to ≈2× próg szumu pojedynczego ziarna (σ ≈ 0.007–0.012 mAP) —
czyli najmniejsza różnica, którą układ n=1 mógłby w~ogóle przypisać formatowi, a~nie
losowości treningu — oraz próg praktycznie nieistotny w~kontekście wspomagania
decyzji/triażu. Dla przejrzystości raportujemy **drabinę czułości** Δ ∈ {0.01,
0.015, 0.02}. Wynik: przy Δ=0.015 i~0.02 **wszystkie pary formatów są równoważne
w~obu architekturach i~w~obu zakresach Q** (przy Δ=0.01 — 2/3 par, bo para z~AVIF
ma tendencję ~0.5 pp). Porównanie raportujemy na **Q=10–95** (matched-CR);
zweryfikowano odporność na wykluczenie Q=100. *Wcielone: `TOST_PRIMARY_DELTA=0.02`,
raporty regenerowane.*

---

## 8. Relacja metryk percepcyjnych (PSNR/SSIM) a skuteczność klasyfikacji

**Kontekst.** Pomiary jakości obrazu pokazały, że JPEG2000 i AVIF mają przy
dopasowanym CR wyraźnie lepsze PSNR/SSIM niż JPEG. Jednocześnie pierwsze pełne
porównanie skuteczności klasyfikacji (ResNet-50, 3 formaty, 13 poziomów Q) NIE
wykazało istotnej różnicy między formatami (Friedman p=0.37; pary po korekcie
Holm p≥0.28; TOST Δ=±0.015 potwierdza równoważność). Nasuwa się wniosek, że
przewaga kodeka w metrykach percepcyjnych nie przekłada się na skuteczność
klasyfikacyjną.

**Problem.** Wniosek ten jest kuszący, ale na obecnym etapie **nieuprawniony jako
mocna teza pracy**: (a) Eksperyment A nie zawiera żadnego formalnego testu
korelacji mAP ↔ PSNR/SSIM — to byłaby obserwacja jakościowa, nie wynik testu;
(b) brak jeszcze danych dla EfficientNet-B0, więc wniosek dotyczyłby jednej
architektury; (c) przy n=1 (por. pkt 3) „brak różnicy w mAP" może maskować efekt
< 1–2 pp — to „brak dowodu na efekt", nie „dowód jego braku". Dopuszczalne dziś
sformułowanie jest wyłącznie hipotetyczne („sugeruje", „kontrastuje z metrykami
percepcyjnymi") i ograniczone do ResNet-50.

**Pytanie do promotora.** Czy dodać do potoku formalny test predykcyjności
(korelacja/regresja mAP względem PSNR/SSIM oraz względem CR na wspólnej siatce
Q)? Czy tezę o (nie)adekwatności metryk percepcyjnych jako proxy skuteczności
klasyfikacyjnej formułować dopiero po ukończeniu EfficientNet-B0 — i jak mocno
(hipotetycznie czy jako wniosek)?

**DECYZJA (przyjęta).** Formalny test korelacji mAP↔PSNR/SSIM/CR **dodany** do
potoku (Spearman + Pearson, per format i~łącznie, Q=10–95 i~Q=10–100, z~notą o~
pseudoreplikacji). Mając komplet **dwóch architektur**, tezę formułujemy jako
**wynik o~umiarkowanej sile** (nie tylko hipotezę): „przy dopasowanym CR wierność
percepcyjna nie jest predyktorem skuteczności klasyfikacji w~tym zadaniu" —
poparte (a) brakiem istotnej różnicy formatów mimo dużych różnic PSNR/SSIM oraz
(b) zmianą znaku korelacji mAP↔PSNR między architekturami (ResNet +0.45 vs
EffNet −0.16). Zachowujemy caveat n=1 (korelacje łączone opisowe). *Wcielone w~
podrozdział „Analiza statystyczna" i~„Porównanie formatów".*

---

## 9. Spójna kierunkowo, lecz nieistotna przewaga AVIF — jak raportować

**Kontekst.** Po ukończeniu całego Eksperymentu A (2 architektury × 3 formaty ×
13 Q) AVIF ma **najwyższą średnią mAP w OBU architekturach** (ResNet-50: +0.43 pp,
EfficientNet-B0: +0.65 pp nad JPEG; AVIF > JPEG przy 9/13 poziomów Q w każdej
architekturze). To jakościowo inny sygnał niż zmienność Q-do-Q (pkt 3), bo jest
**spójny kierunkowo** między architekturami, a nie zmienia znaku.

**Dlaczego mimo to nie można twierdzić, że „AVIF jest lepszy":**
- po korekcie Holm na porównania parami nic nie jest istotne (Holm p ≥ 0.12 w obu
  architekturach);
- sygnał nie jest odporny na wybór metryki — dla f1_macro liderem w ResNet-50
  jest JPEG2000, nie AVIF;
- jednostką replikacji dla spójności międzyarchitekturowej jest **architektura
  (n=2)** — żaden test permutacyjny/sparowany nie może wtedy zejść poniżej
  p=0.25 niezależnie od wielkości efektu; pozornie mocny wynik łączony (t-test na
  26 sparowanych poziomach, p=0.006) jest **artefaktem pseudoreplikacji**
  (wspólna siatka Q + pojedyncze ziarno, n=1);
- efekt rzędu ~0.5 pp jest mniejszy niż odchylenie standardowe treningu po Q
  (~0.7–1.2 pp), więc przy n=1 nieodróżnialny od szumu.

**Proponowany język do pracy:** „możliwa drobna, lecz nieistotna statystycznie
tendencja AVIF (kierunkowo spójna w obu architekturach), nieodróżnialna od szumu
treningu przy pojedynczym ziarnie" — NIE „AVIF poprawia skuteczność".

**Pytanie do promotora.** Czy raportować tę tendencję jako „possible minor
advantage, indistinguishable from noise", czy całkowicie pominąć? Czy rozstrzygać
ją replikacją wieloziarnową (por. pkt 3) — efekt ~0.5 pp jest jedynym
potencjalnym sygnałem w całym eksperymencie, ale jego potwierdzenie wymaga 3–5
ziaren na warunek (koszt ~3–5× czasu obliczeń).

**DECYZJA (przyjęta).** Raportujemy tendencję AVIF **uczciwie, jako możliwą drobną
i~nieistotną statystycznie tendencję** (najwyższa średnia mAP w~obu
architekturach, ale po korekcie Holm p ≥ 0.12, niespójna między metrykami,
nieodróżnialna od szumu ziarna). **Nie** twierdzimy, że „AVIF jest lepszy". Nie
pomijamy jej (jest jedynym kierunkowo spójnym sygnałem i~warto ją odnotować jako
hipotezę do dalszej weryfikacji). Rozstrzygnięcie odkładamy do replikacji
wieloziarnowej jako pracy przyszłej (pkt 3). *Wcielone — patrz pkt analizy
wyników i~wnioski (H2).*

---

## Kolejne kwestie

_(miejsce na dopisywanie następnych tematów w trakcie pracy)_
