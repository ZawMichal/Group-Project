# System Rozpoznawania Twarzy i Detekcji Obiektów
## Dokumentacja Techniczna

---

## Spis Treści
1. [Wstęp](#wstęp)
2. [Architektura Systemu](#architektura-systemu)
3. [Detekcja Obiektów - YOLO](#detekcja-obiektów---yolo)
4. [Rozpoznawanie Twarzy - FaceNet](#rozpoznawanie-twarzy---facenet)
5. [Baza Danych Twarzy](#baza-danych-twarzy)
6. [Proces Rozpoznawania](#proces-rozpoznawania)
7. [Struktura Projektu](#struktura-projektu)
8. [Konfiguracja i Wymagania](#konfiguracja-i-wymagania)

---

## Wstęp

Projekt stanowi kompleksowy system analizy wizyjnej łączący dwa zaawansowane modele uczenia głębokim: You Only Look Once (YOLO) do detekcji obiektów w czasie rzeczywistym oraz FaceNet (oparty na InceptionResNetV1) do rozpoznawania twarzy. System został zaprojektowany w architekturze modułowej, pozwalającej na niezależne użycie poszczególnych komponentów poprzez ujednolicony interfejs (`AIEngine`).

Aplikacja wspomagana jest graficznym interfejsem użytkownika (GUI) zbudowanym w bibliotece Tkinter, umożliwiającym interaktywne zarządzanie systemem, przeprowadzanie detekcji/rozpoznawania w czasie rzeczywistym oraz administrowanie bazą danych znanych twarzy.

---

## Architektura Systemu

### Struktura Katalogów

```
Group-Project/
├── ai_core/                    # Rdzeń systemu AI (backend)
│   ├── __init__.py
│   ├── config.py               # Konfiguracja globalna i wybór urządzenia
│   ├── engine.py               # Główny silnik AI (orkiestra)
│   ├── object_system.py        # Interfejs YOLO
│   ├── face_system.py          # System rozpoznawania twarzy
│   ├── face_encoder.py         # Kodowanie i dopasowywanie twarzy
│   ├── manager.py              # Zarządzanie bazą danych twarzy
│   └── __pycache__/
├── known_faces/                # Baza danych znanych twarzy
│   ├── Osoba_1/
│   │   ├── Osoba_1_1.jpg
│   │   ├── Osoba_1_2.jpg
│   │   └── ...
│   ├── Osoba_2/
│   │   ├── Osoba_2_1.jpg
│   │   └── ...
│   └── ...
├── gui_app.py                  # Interfejs graficzny
├── yolo11n.pt                  # Model YOLO (nano)
├── yolo11s.pt                  # Model YOLO (small)
├── yolo11m.pt                  # Model YOLO (medium)
├── yolo11l.pt                  # Model YOLO (large)
├── yolo11x.pt                  # Model YOLO (xlarge)
├── requirments.txt             # Zależności Python
└── dokumentacja/               # Dokumentacja projektu
    ├── Dokumentacja_ogólna.md
    ├── Dokumentacja_GUI.md
    └── Dokumentacja_użytkownika.md
```

### Przepływ Danych

```
Strumień Wideo Kamera
    ↓
[AIEngine] - Główny Orkiestrator
    ├─→ [ObjectSystem] (YOLO) - Detekcja Obiektów
    │        ↓
    │   Pudełka z klasami obiektów
    │
    └─→ [FaceSystem] (FaceNet) - Rozpoznawanie Twarzy
             ├─→ [MTCNN] - Detekcja Twarzy
             ├─→ [InceptionResNetV1] - Kodowanie Twarzy
             └─→ [FaceEncoder] - Dopasowywanie z Bazą
                      ↓
                 Znana osoba/Nieznana
    ↓
Anotowany Obraz
    ↓
[GUI] - Wyświetlenie
```

### Zasada Modułowości

Każdy komponent (`ObjectSystem`, `FaceSystem`, `PersonManager`) jest niezależny i może być użyty poza GUI. Głównym punktem wejścia jest `AIEngine`, które zapewnia ujednolicony interfejs:

```python
from ai_core.engine import AIEngine
import cv2

engine = AIEngine(log_callback=print)
engine.enable_yolo = True
engine.enable_faces = True

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        result = engine.process_frame(frame)  # Przetworzony obraz z anotacjami
        cv2.imshow('Result', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## Detekcja Obiektów - YOLO

### Teoria YOLO

You Only Look Once (YOLO) to rodzina modeli sieci neuronowych konwolucyjnych opracowana na potrzeby detekcji obiektów w czasie rzeczywistym. Głównym przełomem YOLO jest spojrzenie na problem detekcji jako jeden problem regresji od pikseli obrazu do współrzędnych pudełek (bounding boxes) i prawdopodobieństw klas.

#### Warianty Modeli

Projekt wykorzystuje YOLO v11, dostępne w pięciu wariantach:

| Model | Rozmiar | Szybkość | Dokładność | Zastosowanie |
|-------|---------|----------|-----------|--------------|
| yolo11n (nano) | ~2.6 MB | Bardzo szybka | Niska | Urządzenia mobilne, real-time |
| yolo11s (small) | ~9.1 MB | Szybka | Średnia | Embedded, monitoring |
| yolo11m (medium) | ~20.1 MB | Średnia | Dobra | GPU, pożądany balans |
| yolo11l (large) | ~35.5 MB | Powolna | Wysoka | Zaawansowana analiza |
| yolo11x (xlarge) | ~56.9 MB | Bardzo powolna | Bardzo wysoka | Aplikacje krytyczne |

Wybór wariantu jest kompromisem pomiędzy szybkością przetwarzania a dokładnością detekcji. W projekcie domyślnie używany jest wariant nano (`yolo11n.pt`) dla maksymalnej wydajności.

### Architektura Sieci YOLO

YOLO v11 składa się z trzech głównych części:

1. **Backbone** - Ekstraktor cech (konwolucje, pooling, residual blocks)
   - Zmniejsza wymiary obrazu wejściowego (np. 640×640 → 80×80)
   - Wydobywa hierarchiczne cechy (tekstury, kształty, obiekty)

2. **Neck** - Fusion module
   - Łączy cechy z różnych poziomów piramidy
   - Umożliwia detekcję obiektów różnych rozmiarów

3. **Head** - Dekoder predykcji
   - Przewiduje: współrzędne pudełek (x, y, w, h)
   - Przewiduje: prawdopodobieństwo obiektu (objectness)
   - Przewiduje: prawdopodobieństwa klas dla każdego pudełka

### Proces Detekcji Krok po Kroku

```
1. Wejście: Obraz RGB 640×640
   ↓
2. Normalizacja pikseli: [0,255] → [0,1]
   ↓
3. Backbone: Ekstrakcja cech hierarchicznych
   - Layer 1: Cechy niskiego poziomu (krawędzie, tekstury)
   - Layer 2: Cechy średniego poziomu (części obiektów)
   - Layer 3: Cechy wysokiego poziomu (całe obiekty)
   ↓
4. Neck: Łączenie cech z różnych skal
   ↓
5. Head: Generowanie predykcji
   - Dla każdego punktu gridki (80×80):
     * 3 przewidywania (różne aspect ratio)
     * Dla każdego: (x, y, w, h, confidence, class_probabilities)
   ↓
6. Non-Maximum Suppression (NMS)
   - Usuwanie duplikatów
   - Jeśli IOU > threshold: usuń mniej pewne pudełko
   ↓
7. Wyjście: Lista pudełek z klasami i pewnością
```

### Procedura Filtrowania

```python
def process_frame(self, frame):
    # Confidence threshold (domyślnie 0.5)
    # Tylko pudełka z confidence >= 0.5 są zwracane
    results = self.model.predict(frame, conf=0.5)
    return results[0]
```

W kodzie projektu (`object_system.py`):

```python
self.conf = 0.5  # Próg pewności
results = self.model.predict(frame, verbose=False, conf=self.conf)
```

### Śledzenie Obiektów (Tracking)

Projekt obsługuje opcjonalne śledzenie obiektów między klatkami:

```python
if self.enable_tracking:
    results = self.model.track(frame, persist=True, verbose=False, conf=self.conf)
```

Parametr `persist=True` pozwala modelowi zapamiętać obiekty z poprzednich klatek, przypisując im ID (śledzenie wieloklatkami).

### Implementacja w Projekcie

Klasa `ObjectSystem` (`ai_core/object_system.py`):

```python
class ObjectSystem:
    def __init__(self, model_variant="yolo11n.pt", enable_tracking=True):
        self.model = YOLO(model_variant)  # Załadowanie modelu
        self.conf = 0.5                    # Próg pewności
        self.enable_tracking = enable_tracking

    def process_frame(self, frame):
        """Przetworzenie jednej klatki"""
        if self.enable_tracking:
            results = self.model.track(frame, persist=True, conf=self.conf)
        else:
            results = self.model.predict(frame, conf=self.conf)
        return results[0]

    def draw_results(self, frame, results):
        """Rysowanie pudełek i etykiet na obrazie"""
        annotator = Annotator(frame, line_width=2)
        
        if results.boxes is not None:
            for box in results.boxes:
                label = f"{self.model.names[int(box.cls)]} {float(box.conf):.2f}"
                annotator.box_label(box.xyxy[0], label)
        
        return frame
```

---

## Rozpoznawanie Twarzy - FaceNet

### Teoria FaceNet

FaceNet to sieć neuronowa konwolucyjna opracowana przez firmę Google do wyliczania osadzenia twarzy (face embedding). Zamiast klasyfikować osoby bezpośrednio, FaceNet uczy się transformować zdjęcie twarzy w wktor przestrzeni n-wymiarowej (128 wymiarów w naszym projekcie), gdzie twarze tej samej osoby znajdują się blisko siebie, a twarze różnych osób są oddalone.

#### Kluczowe Komponenty

Projekt wykorzystuje dwa zaawansowane modele:

**1. MTCNN (Multi-task Cascaded Convolutional Networks)**
- Detekcja twarzy na obrazie
- Zwraca pudełka (bounding boxes) dla każdej twarzy
- Trzy kaskadowe sieci neuronowe:
  - P-Net (Proposal Network): Szybka segmentacja kandydatów
  - R-Net (Refine Network): Filtrowanie fałszywych pozytywów
  - O-Net (Output Network): Dokładne lokalizacje i punkty orientacyjne

**2. InceptionResNetV1 (FaceNet)**
- Kodowanie twarzy do wektora 128-wymiarowego
- Trenowany na zbiorze VGGFace2 (miliony twarzy)
- Dostępny poprzez bibliotekę `facenet-pytorch`

### Proces Rozpoznawania Twarzy

```
Klatka wideo (RGB)
    ↓
[MTCNN Detekcja]
    ├─→ P-Net: Generowanie kandydatów
    ├─→ R-Net: Filtrowanie i calibracja
    └─→ O-Net: Dokładne pudełka i landmarki
    ↓
Dla każdej wykrytej twarzy:
    ↓
    ├─→ 1. Ekstrakcja twarzy (160×160 pikseli)
    ├─→ 2. Normalizacja i preprocessing
    ├─→ 3. InceptionResNetV1 encoding → wktor 128-wymiarowy
    └─→ 4. Porównanie z bazą danych
            ├─→ Obliczenie dystansu euklidesowego do każdej znanej twarzy
            ├─→ Znalezienie najbliższej znanej twarzy
            └─→ Jeśli dystans < próg: rozpoznana osoba
                Jeśli dystans >= próg: "Unknown"
    ↓
Wynik: ID osoby lub "Unknown"
```

### Kodowanie Twarzy - Szczegóły Techniczne

#### Funkcja Reprezentacji Facenet

FaceNet uczy się funkcji:
$$f: \text{obraz} \rightarrow \mathbb{R}^{128}$$

Gdzie:
- Wejście: Zdjęcie twarzy (160×160×3 RGB)
- Wyjście: Wektor 128-wymiarowy (embedding)

#### Funkcja Straty (Triplet Loss)

Sieć trenowana jest z tzw. "triplet loss", która minimalizuje:

$$L = \max(d^+ - d^- + \alpha, 0)$$

Gdzie:
- $d^+$ = dystans do twarzy tej samej osoby (powinien być mały)
- $d^-$ = dystans do twarzy innej osoby (powinien być duży)
- $\alpha$ = margines bezpieczeństwa (zwykle 0.2)

Cel: Twarze tej samej osoby mają małe $d^+$, twarze różnych osób mają duże $d^-$.

#### Dopasowywanie Twarzy w Projekcie

```python
# Obliczenie dystansu euklidesowego
distances = np.linalg.norm(known_encodings - test_encoding, axis=1)

# Znalezienie minimum
min_distance = np.min(distances)

# Konwersja dystansu na pewność (confidence)
confidence = max(0, 1 - (min_distance / 1.2))

# Porównanie z progiem (domyślnie 0.6)
if confidence >= config.confidence_threshold:
    recognized = True
else:
    recognized = False  # "Unknown"
```

### Implementacja w Projekcie

#### Klasa `FaceEncoder` (ai_core/face_encoder.py)

```python
class FaceEncoder:
    def __init__(self):
        # MTCNN - detekcja twarzy
        self.mtcnn = MTCNN(
            image_size=160,      # Rozmiar docelowy
            margin=20,           # Margines wokół twarzy
            device='cuda'        # GPU
        )
        
        # InceptionResNetV1 - kodowanie twarzy
        self.resnet = InceptionResnetV1(
            pretrained='vggface2'  # Trenowany na VGGFace2
        ).to('cuda')
        
        # Baza znanych kodowań
        self.known_encodings = {}  # Dict[osoba] = array kodowań
        self.known_names = []      # Lista znanych osób

    def encode_face(self, image_path):
        """Kodowanie jednej twarzy z pliku"""
        img = Image.open(image_path).convert('RGB')
        
        # Detekcja
        boxes, probs = self.mtcnn.detect(img)
        if boxes is None:
            return None, None, 'no_face'
        
        # Ekstrakcja twarzy (160×160)
        faces = self.mtcnn(img)
        face = faces[0]  # Pierwsza twarz
        
        # Kodowanie
        with torch.no_grad():
            encoding = self.resnet(face.unsqueeze(0)).cpu().numpy()[0]
        
        return encoding, prob, None

    def load_known_faces(self):
        """Załadowanie bazy danych"""
        for person_dir in self.known_faces_dir.iterdir():
            person_name = person_dir.name
            encodings = []
            
            # Kodowanie każdego zdjęcia osoby
            for image_file in person_dir.glob('*.jpg'):
                encoding, _, err = self.encode_face(str(image_file))
                if encoding is not None:
                    encodings.append(encoding)
            
            if encodings:
                self.known_encodings[person_name] = np.array(encodings)

    def recognize_faces(self, frame_rgb):
        """Rozpoznawanie twarzy na klatce"""
        img = Image.fromarray(frame_rgb)
        
        # Detekcja wszystkich twarzy
        boxes, probs = self.mtcnn.detect(img)
        if boxes is None:
            return []
        
        faces = self.mtcnn(img)  # Tensor: (N, 3, 160, 160)
        
        # Kodowanie wszystkich twarzy
        with torch.no_grad():
            encodings = self.resnet(faces).cpu().numpy()
        
        results = []
        for i, box in enumerate(boxes):
            test_encoding = encodings[i]
            
            # Porównanie z bazą
            matches = []
            for person_name, known_encs in self.known_encodings.items():
                distances = np.linalg.norm(known_encs - test_encoding, axis=1)
                min_dist = np.min(distances)
                conf = max(0, 1 - (min_dist / 1.2))
                
                matches.append({
                    'name': person_name,
                    'confidence': conf,
                    'distance': min_dist
                })
            
            matches.sort(key=lambda x: x['confidence'], reverse=True)
            top_match = matches[0]
            
            # Zastosowanie progu
            if top_match['confidence'] >= 0.6:
                name = top_match['name']
            else:
                name = "Unknown"
            
            results.append({
                'box': box.astype(int),
                'name': name,
                'confidence': top_match['confidence']
            })
        
        return results
```

---

## Baza Danych Twarzy

### Struktura i Organizacja

Baza danych twarzy zorganizowana jest w hierarchiczną strukturę folderów:

```
known_faces/
├── Anna_Kowalski/
│   ├── Anna_Kowalski_1.jpg
│   ├── Anna_Kowalski_2.jpg
│   ├── Anna_Kowalski_3.jpg
│   └── Anna_Kowalski_4.jpg
├── Jan_Nowak/
│   ├── Jan_Nowak_1.jpg
│   ├── Jan_Nowak_2.jpg
│   └── Jan_Nowak_3.jpg
└── Marta_Lewandowska/
    ├── Marta_Lewandowska_1.jpg
    └── Marta_Lewandowska_2.jpg
```

Każdy folder reprezentuje jedną osobę. Liczba zdjęć na osobę wpływa na dokładność:

| Liczba Zdjęć | Dokładność | Notatki |
|--------------|-----------|---------|
| 1-2 | Niska | Niepewne, podatne na błędy |
| 3-5 | Średnia | Akceptowalne w warunkach stałych |
| 5-10 | Dobra | Rekomendowane minimum |
| 10+ | Bardzo dobra | Optymalne dla produkcji |

### Proces Zapamiętywania Twarzy

#### 1. Rejestracja Osoby (Capture Mode)

```python
def save_training_photo(self, name, frame):
    """Zapisanie jednego zdjęcia treningowego"""
    
    # Tworzenie folderu dla osoby
    person_dir = self.create_person_folder(name)
    
    # Numeracja zdjęć
    existing = list(person_dir.glob("*.jpg"))
    count = len(existing) + 1
    
    # Zapis
    filename = person_dir / f"{name}_{count}.jpg"
    cv2.imwrite(filename, frame)
    
    return filename
```

Procedura w GUI:
1. Użytkownik wprowadza imię i nazwisko
2. Naciska "Capture"
3. GUI co ~100 ms (przy ~10 FPS) zapisuje klatkę z kamery
4. Zdjęcia zapisywane są w formacie JPG do `known_faces/ImieNazwisko/`
5. Po ~30 sekundach rejestracji mamy ~10-15 zdjęć

#### 2. Kodowanie Bazy (Loading Phase)

Przy pierwszym uruchomieniu rozpoznawania twarzy:

```python
def load_known_faces(self):
    """Załadowanie i kodowanie całej bazy"""
    
    print(f"Skanowanie {known_faces_dir}")
    
    # Dla każdej osoby w bazie
    for person_dir in known_faces_dir.iterdir():
        person_name = person_dir.name
        encodings = []
        
        # Dla każdego zdjęcia osoby
        for image_file in person_dir.glob('*.jpg'):
            encoding, prob, err = self.encode_face(str(image_file))
            
            if encoding is not None:
                encodings.append(encoding)  # Tensor 128-wymiarowy
            else:
                print(f"[SKIP] {image_file}: {err}")
        
        if encodings:
            # Zapisanie średnich kodowań
            self.known_encodings[person_name] = np.array(encodings)
            self.known_names.append(person_name)
    
    print(f"Załadowano {len(self.known_names)} osób")
```

### Cechy Zdjęć Treningowych

Aby osiągnąć najlepsze wyniki, zdjęcia powinny zawierać:

1. **Różne Kąty Głowy**
   - Frontalna (0°)
   - Z lewej (±30°)
   - Z prawej (±30°)
   - Z góry/dołu (±15°)

2. **Różne Oświetlenie**
   - Naturalne światło dzienne
   - Oświetlenie sztuczne
   - Półcień
   - Różne intensywności

3. **Zbliżenia i Dystansy**
   - Bliska odległość (twarze głowy)
   - Średnia odległość
   - Większa odległość

4. **Różne Wyrażenia**
   - Neutralne
   - Z uśmiechem
   - Poważne

5. **Zmienne Accessoria**
   - Okulary (i bez)
   - Kapelusz/czapka
   - Inna fryzura

Ilość i różnorodność zdjęć bezpośrednio wpływa na niezawodność rozpoznawania w warunkach rzeczywistych.

### Obliczanie Kodowań - Koszt Obliczeniowy

Dla każdego zdjęcia:
1. MTCNN detekcja: ~50-100 ms (GPU)
2. InceptionResNetV1 kodowanie: ~30-50 ms (GPU)
3. I/O: ~10-20 ms

**Razem: ~100-150 ms na zdjęcie**

Dla bazy z 10 osób × 10 zdjęć = 100 zdjęć:
- **Całkowity czas załadowania: 10-15 sekund**

W praktyce obserwujemy czasami do 400+ sekund ze względu na:
- Pierwszorazową inicjalizację CUDA (30-50 sekund)
- Cache'owanie modeli PyTorch
- Kompilację kerneli GPU

---

## Proces Rozpoznawania

### Przepływ Przetwarzania Klatki

```
Wideo z Kamery (np. 1920×1080, 30 FPS)
    ↓
[Skalowanie do 640×h] (dla wydajności)
    ↓
┌─────────────────────────────────────────────────────┐
│ Jeśli enable_yolo = True:                           │
│   ├─→ YOLO inference (predict lub track)            │
│   ├─→ Zwrócenie pudełek z klasami                   │
│   └─→ Rysowanie na obrazie                          │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Jeśli enable_faces = True:                          │
│   ├─→ Konwersja BGR → RGB                           │
│   ├─→ MTCNN detekcja twarzy → pudełka               │
│   ├─→ InceptionResNetV1 kodowanie każdej twarzy     │
│   ├─→ Porównanie z bazą znanych kodowań             │
│   ├─→ Obliczenie dystansu euklidesowego             │
│   ├─→ Konwersja dystansu na pewność (confidence)    │
│   └─→ Rysowanie pudełek z imionami i pewnością      │
└─────────────────────────────────────────────────────┘
    ↓
Anotowany obraz (z pudełkami i etykietami)
    ↓
[GUI Wyświetlenie]
```

### Formalna Procedura Rozpoznawania

Niech:
- $\mathbf{t}_i$ = kodowanie $i$-tej twarzy z klatki (128-wymiarowy wektor)
- $\mathbf{k}_{j,m}$ = kodowanie $m$-tego zdjęcia osoby $j$ z bazy
- $\mathcal{N}$ = zbiór wszystkich osób w bazie
- $\tau$ = próg pewności (domyślnie 0.6)

Algorytm:

1. **Dla każdej twarzy $i$ na klatce:**
   
   2. **Obliczenie dystansu do każdej osoby $j$:**
      $$d_{i,j} = \min_{m} \|\mathbf{t}_i - \mathbf{k}_{j,m}\|_2$$
      
      (Euklidesowy dystans do najbliższego zdjęcia osoby)
   
   3. **Konwersja dystansu na pewność:**
      $$c_i = \max(0, 1 - \frac{d_i}{1.2})$$
      
      gdzie $d_i = \min_j d_{i,j}$
   
   4. **Znalezienie osoby z maksymalną pewnością:**
      $$j^* = \arg\max_j c_i \text{ dla osoby } j$$
   
   5. **Decyzja rozpoznania:**
      $$\text{result}_i = \begin{cases}
      \text{osoba } j^* & \text{jeśli } c_i \geq \tau \\
      \text{"Unknown"} & \text{jeśli } c_i < \tau
      \end{cases}$$

### Wybór Progu - Kompromis Precision/Recall

```
confidence_threshold = 0.6 (domyślnie)

Niski próg (0.3-0.4):
  ✓ Mniej fałszywych negatywów (Unknown)
  ✗ Więcej fałszywych pozytywów (błędna identyfikacja)

Wysoki próg (0.7-0.8):
  ✓ Mniej fałszywych pozytywów
  ✗ Więcej fałszywych negatywów (Unknown)

0.6 = rozsądny kompromis dla większości aplikacji
```

### Kodowanie AI Engine

```python
def process_frame(self, frame):
    """Główna pętla przetwarzania"""
    annotated_frame = frame.copy()

    # 1. YOLO
    if self.enable_yolo:
        yolo_results = self.object_system.process_frame(frame)
        annotated_frame = self.object_system.draw_results(
            annotated_frame, yolo_results
        )

    # 2. Face Recognition
    if self.enable_faces:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_system.recognize_faces(rgb_frame)
        annotated_frame = self.face_system.draw_results(
            annotated_frame, face_results
        )

    return annotated_frame
```

---

## Struktura Projektu

### ai_core/config.py

Moduł konfiguracyjny zarządzający:
- Detekcją urządzenia (CUDA/MPS/CPU)
- Ścieżkami projektu
- Progami i parametrami

```python
class Config:
    device: str                    # 'cuda', 'mps', lub 'cpu'
    project_root: Path             # Katalog projektu
    known_faces_dir: Path          # known_faces/
    confidence_threshold: float    # 0.6 (domyślnie)
    detection_threshold: float     # 0.9 (domyślnie)
    image_size: int                # 160 (dla MTCNN)
    margin: int                    # 20 (piksel. wokół twarzy)
```

### ai_core/engine.py

Główny orkiestrator systemu. Zapewnia jednolity interfejs:

```python
class AIEngine:
    enable_yolo: bool              # Włączenie detekcji YOLO
    enable_faces: bool             # Włączenie rozpoznawania twarzy
    manager: PersonManager         # Zarządzanie bazą danych
    
    def process_frame(frame) -> np.ndarray
        # Przetworzenie klatki (YOLO + Face)
    
    def reload_faces() -> None
        # Ponowne załadowanie bazy twarzy
    
    def change_yolo_model(model_name) -> bool
        # Zmiana wariantu YOLO
```

### ai_core/object_system.py

Interfejs YOLO:

```python
class ObjectSystem:
    model: YOLO                    # Model YOLO
    conf: float                    # Próg pewności (0.5)
    enable_tracking: bool          # Śledzenie obiektów
    
    def process_frame(frame) -> YOLOResults
    def draw_results(frame, results) -> np.ndarray
    def change_model(variant) -> bool
```

### ai_core/face_encoder.py

Jądro rozpoznawania twarzy:

```python
class FaceEncoder:
    mtcnn: MTCNN                   # Detektor twarzy
    resnet: InceptionResNetV1     # Enkoder twarzy
    known_encodings: dict          # {osoba: [kodowania]}
    known_names: list              # [lista osób]
    
    def encode_face(image_path) -> Tuple[np.ndarray, float, str]
        # Kodowanie jednej twarzy
    
    def load_known_faces() -> None
        # Załadowanie bazy danych
    
    def recognize_faces(frame_rgb) -> List[dict]
        # Rozpoznawanie wszystkich twarzy na klatce
```

### ai_core/face_system.py

Nadklasa rozszerzająca `FaceEncoder`:

```python
class FaceSystem(FaceEncoder):
    def __init__(self):
        super().__init__()
        self.load_known_faces()  # Automatyczne załadowanie
    
    def draw_results(frame, results) -> np.ndarray
        # Rysowanie pudełek na klatce
```

### ai_core/manager.py

Zarządzanie plikami bazy danych:

```python
class PersonManager:
    def get_people_list() -> List[str]
        # Lista znanych osób
    
    def create_person_folder(name) -> Path
        # Tworzenie folderu nowej osoby
    
    def save_training_photo(name, frame) -> str
        # Zapis zdjęcia treningowego
    
    def delete_person(name) -> bool
        # Usunięcie osoby z bazy
    
    def rename_person(old_name, new_name) -> bool
        # Zmiana nazwy osoby
```

### gui_app.py

Interfejs graficzny (Tkinter):
- Zarządzanie kamerą
- Kontrola YOLO i Face Recognition
- Administracja bazą danych

---

## Konfiguracja i Wymagania

### Wymagania Systemowe

**Minimalne:**
- Python 3.10+
- 4 GB RAM
- CPU wielordzeniowy
- Kamera internetowa (USB/wbudowana)

**Rekomendowane (dla GPU):**
- NVIDIA GPU z CUDA Compute Capability 3.5+
- 8+ GB RAM GPU (VRAM)
- CUDA Toolkit 11.8+
- cuDNN 8.9+

### Zależności Python

```
# Główne
opencv-python==4.13.0.90
ultralytics==8.4.6
torch==2.7.1+cu118              # Z CUDA 11.8
torchvision==0.22.1+cu118       # Matching torch version
facenet-pytorch==2.5.3
Pillow==12.1.0
numpy==2.4.1

# Wspomagające (automatycznie instalowane)
scipy==1.17.0
matplotlib==3.10.8
requests==2.32.5
PyYAML==6.0.3
```

### Instalacja Środowiska

```bash
# 1. Stwórz wirtualne środowisko
python -m venv venv

# 2. Aktywuj środowisko
# Na Windows:
venv\Scripts\activate
# Na Linux/Mac:
source venv/bin/activate

# 3. Instalacja zależności
pip install -r requirments.txt

# 4. Uruchomienie GUI
python gui_app.py
```

### Konfiguracja GPU (NVIDIA CUDA)

Aby użyć GPU:

```bash
# Sprawdzenie CUDA w PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Powinno wypisać: True
```

Jeśli `False`, zainstaluj:
1. NVIDIA CUDA Toolkit 11.8
2. cuDNN 8.9 (do CUDA 11.x)
3. Przeinstaluj torch z CUDA support:

```bash
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### Automatyczna Detekcja Urządzenia

Kod automatycznie wybiera najlepsze dostępne urządzenie:

```python
# Z config.py
if torch.cuda.is_available():
    device = 'cuda'  # NVIDIA GPU
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'   # Apple Silicon
else:
    device = 'cpu'   # Fallback
```

---

## Streszczenie Techniczne

Projekt stanowi komprehensywny system analizy wizyjnej łączący dwa zaawansowane modele:

1. **YOLO v11** - Detekcja obiektów w czasie rzeczywistym
   - 5 wariantów dla różnych zastosowań
   - Tracking obiektów między klatkami
   - Outputy: pudełka, klasy, pewność

2. **FaceNet (InceptionResNetV1 + MTCNN)** - Rozpoznawanie twarzy
   - MTCNN do detekcji twarzy
   - InceptionResNetV1 do kodowania 128-wymiarowego
   - Dopasowanie poprzez dystans euklidesowy
   - Próg adaptacyjny: 0.6

3. **Architektura Modułowa**
   - Niezależne komponenty (YOLO, Face, Manager)
   - Jednolity interfejs (AIEngine)
   - Możliwość użycia bez GUI

4. **Interfejs Graficzny**
   - Tkinter dla kompatybilności
   - Zarządzanie bazą danych
   - Real-time monitoring

System osiąga balans między szybkością (GPU acceleration), dokładnością (zaawansowane modele) i elastycznością (architektura modułowa).

---

## Aktualizacje funkcjonalne

### Blacklist użytkowników
- Dodano plik CSV z flagą 0/1 dla każdej osoby.
- Status blacklist jest wykorzystywany podczas rysowania wyników Face Recognition.
- Kolor czerwony jest zarezerwowany wyłącznie dla osób z blacklisty.

### YOLO Warnings
- Dodano opcję ostrzeżeń dla wykrytych zwierząt (żółty alert).
- Dodano alert krytyczny dla `knife` i `baseball bat` (czerwony alert).
- Lista klas jest pobierana bezpośrednio z `model.names` w runtime.

### Progi detekcji i Top‑K etykiet
- Konfigurowalne progi dla YOLO i Face Recognition.
- Możliwość wyświetlania 1–3 najbardziej prawdopodobnych etykiet twarzy.

---

## Bibliografia i Źródła

- Szegedy, C., et al. "Rethinking the Inception Architecture for Computer Vision" (Inception V3/V4)
- Ioffe, S., Szegedy, C. "Batch Normalization: Accelerating Deep Network Training" (2015)
- Schroff, F., et al. "FaceNet: A Unified Embedding for Face Recognition and Clustering" (2015)
- Redmon, J., Divvala, S., et al. "You Only Look Once: Unified, Real-Time Object Detection" (2016)
- Ultralytics YOLOv8/v11 Documentation
- facenet-pytorch GitHub Repository
- OpenCV Documentation

---

**Koniec Dokumentacji**
