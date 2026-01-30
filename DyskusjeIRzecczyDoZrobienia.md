DICOM – protokuł jako zestaw tagów (tekstowe/numeryczne/obraz) – prodkreśl że to stratne kompresje danych – też standard DICOM

Porównanie różnych kompresji czy najgorszy będzie najgorzej działał

Sprawdzić poziomy kopresji w stowarzyszeniach radiologii w jepg i jpeg2000 które są apcewtowalne

jpeg-> Jaki jpeg wyrzucił wielkość obrazu jaki compression retio i wrzucam to do aviv i jpeg2000

odnieść się do tych artykułów że ten zbiór ma sens

Skoncentrować się na segmentacji jaki wpływm ma kompresja na segmentacje klasyfikacja w jest mniej ważna



Krok 2 — Eksperymenty A (6 komend, odpalaj po kolei):


Done - python src/experiment_a.py --model resnet50 --task syntax --format jpeg --mvp --device cuda
Done - python src/experiment_a.py --model resnet50 --task syntax --format jpeg2000 --mvp --device cuda
Done - python src/experiment_a.py --model resnet50 --task syntax --format avif --mvp --device cuda
POMINIĘTE - stenosis (patrz uwaga poniżej)

Krok 3 — Eksperymenty B (tylko syntax):


Done - python src/experiment_b.py --model resnet50 --task syntax --format jpeg --mvp --device cuda
Done - python src/experiment_b.py --model resnet50 --task syntax --format jpeg2000 --mvp --device cuda
Done - python src/experiment_b.py --model resnet50 --task syntax --format avif --mvp --device cuda
POMINIĘTE - stenosis (patrz uwaga poniżej)

## UWAGA: Rezygnacja z task "stenosis" dla klasyfikacji

Dataset ARCADE stenosis zawiera tylko jedną kategorię adnotacji ("stenosis", id=26),
mimo że plik categories definiuje 26 kategorii (te same co syntax).
W rezultacie wszystkie obrazy otrzymują tę samą etykietę dominantną (klasa 25),
co powoduje trywialne 100% accuracy — model po prostu zawsze przewiduje tę samą klasę.

Stenosis w ARCADE jest przeznaczony do **segmentacji** (lokalizacja zwężeń na obrazie),
a nie do klasyfikacji binarnej. Brak negatywnych przykładów (obrazy bez stenozy)
uniemożliwia sensowną klasyfikację binarna.

**Decyzja:** Eksperymenty klasyfikacyjne przeprowadzamy tylko na task "syntax" (26 klas),
który daje wartościowe i zróżnicowane wyniki.