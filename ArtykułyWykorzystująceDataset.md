Artykuły wykorzystujące dataset ARCADE (dowolna wersja)

1. Dataset for Automatic Region-based Coronary Artery Disease Diagnostics Using X-Ray Angiography Images
Link: <https://www.nature.com/articles/s41597-023-02871-z>
Główny artykuł opisujący dataset ARCADE opublikowany w Scientific Data (styczeń 2024) przez Popov et al. Przedstawia dataset zawierający 3000 ekspertowo oznaczonych obrazów angiografii rentgenowskiej: 1500 do klasyfikacji segmentów naczyń według SYNTAX Score i 1500 do detekcji zwężeń. Dataset został opublikowany dla MICCAI 2023 challenge i zawiera baseline modele (U-Net, YOLOv8).
2. LASF: a local adaptive segmentation framework for coronary angiogram segments
Link: <https://pubmed.ncbi.nlm.nih.gov/39881813/>
Artykuł z Health Information Science and Systems (styczeń 2025) rozwijający framework LASF oparty na YOLOv8 z algorytmami dylatacji i erozji. Autorzy wzbogacili dataset ARCADE o dodatkowe adnotacje proksymalnych i dystalnych segmentów naczyń. Model przewyższa UNet i DeepLabV3Plus w zadaniach segmentacji naczyń wieńcowych.
3. Evaluating Stenosis Detection with Grounding DINO, YOLO, and DINO-DETR
Link: <https://arxiv.org/html/2503.01601v1>
Badanie porównawcze z marca 2025 ewaluujące trzy architektury detekcji obiektów (Grounding DINO, YOLO, DINO-DETR) na datasecie ARCADE. Używa metryk IoU, Average Precision i Average Recall do analizy dokładności detekcji zwężeń w obrazach angiografii. Dostarcza porównanie wydajności różnych architektur dla diagnostyki CAD.
4. Accurate segmentation and labeling of coronary artery segments in X-ray angiography with an improved UNet-based cGAN architecture
Link: <https://www.sciencedirect.com/science/article/abs/pii/S1746809425013230>
Artykuł z Biomedical Signal Processing and Control (październik 2025) przedstawiający model UCNet oparty na conditional GAN. Ewaluowany na ARCADE Challenge datasets z MICCAI 2023, osiąga średni F1 score 84.43% na 20 segmentach tętnic wieńcowych. Model wprowadza nową funkcję straty ("segment loss") poprawiającą dokładność klasyfikacji i segmentacji granic.
5. Coronary Tree Segmentation and Labelling in X-ray Angiography Images Using Graph Deep Learning
Link: <https://link.springer.com/chapter/10.1007/978-3-658-47422-5_51>
Publikacja ze Springer (2025) wykorzystująca graph convolutional networks do labelowania segmentów tętnic wieńcowych. Reprezentuje strukturę drzewa wieńcowego jako graf wyekstrahowany z obrazu angiografii. Trenowany i ewaluowany na datasecie ARCADE, osiągnął F1-score 53.68 dla klasyfikacji węzłów grafu według schematu SYNTAX.
6. Deep vessel segmentation with U-Net and texture representation of image (TRI) features
Link: <https://www.sciencedirect.com/science/article/abs/pii/S0169260725004894>
Artykuł z Computer Methods and Programs in Biomedicine (wrzesień 2025) integrujący cechy tekstury Haralicka i Law'a z architekturą U-Net. Model osiąga 98% accuracy i 0.89 F1-score na klinicznych danych, a także jest benchmarkowany na datasecie ARCADE (F1: 0.78). Wykorzystuje zaawansowane preprocessowanie do poprawy jakości obrazu przed segmentacją.
7. CAG-VLM: Fine-Tuning of a Large-Scale Model to Recognize Angiographic Images for Next-Generation Diagnostic Systems
Link: <https://arxiv.org/abs/2505.04964>
Artykuł z maja 2025 o fine-tuningu vision-language models dla diagnostyki CAG. Autorzy cytują ARCADE jako benchmark zawierający 1,200 obrazów z adnotacjami według metodologii SYNTAX Score. Wprowadzają dwujęzyczny (japońsko-angielski) dataset i fine-tunują VLM (PaliGemma2, Gemma3) do generowania raportów klinicznych z obrazów angiografii.
8. CoronaryDominance: Angiogram dataset for coronary dominance classification
Link: <https://www.nature.com/articles/s41597-025-04676-8>
Artykuł z Scientific Data (luty 2025) prezentujący nowy dataset dla klasyfikacji dominacji wieńcowej. Autorzy cytują ARCADE jako najnowszy dataset zawierający 1,500 obrazów do segmentacji naczyń wieńcowych, klasyfikacji i detekcji zwężeń. Wykorzystują ARCADE jako punkt odniesienia dla porównania z własnym datasetem obejmującym 1,574 badania angiograficzne.
