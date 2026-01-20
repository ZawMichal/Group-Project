# Dokumentacja użytkownika

Ten dokument opisuje wszystkie funkcje dostępne z poziomu GUI oraz sposób ich użycia.

---

## 1. Panel główny i podgląd wideo
- **Prawy panel** to podgląd obrazu z kamery.
- Gdy kamera nie jest podłączona, widać komunikat „KAMERA WYŁĄCZONA”.
- Po podłączeniu kamery obraz jest wyświetlany w czasie rzeczywistym.
- Jeśli AI jest włączone, obraz jest przetwarzany i opisywany (ramki, etykiety, metryki).

---

## 2. Blok „Camera Source” (połączenie z kamerą)
**Co robi:** umożliwia wybór ID kamery i podłączenie.

**Jak użyć:**
1. Wybierz numer kamery w polu „Camera ID”.
2. Kliknij **✓ Connect**.
3. Status obok zmieni się na „Camera X Active”, jeśli połączenie się uda.

**Co się dzieje w środku:**
- Aplikacja uruchamia wątek, tworzy `cv2.VideoCapture`, testuje klatkę i aktywuje strumień.

---

## 3. Blok „YOLO Detection” (detekcja obiektów)
**Co robi:** wykrywa obiekty (np. osoba, samochód, pies) w czasie rzeczywistym.

**Elementy:**
- **Model**: wybór wariantu YOLO (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x).
- **Conf**: włącza/wyłącza wyświetlanie pewności (confidence).
- **Labels**: włącza/wyłącza wyświetlanie nazw klas.
- **Warnings**: włącza alerty ostrzegawcze (żółte dla zwierząt, czerwone dla `knife` i `baseball bat`).
- **▶ Run**: start YOLO.
- **⏹ Stop**: zatrzymanie YOLO.

**Scenariusze użycia:**
- Chcesz tylko ramki bez opisów → odznacz **Labels** i **Conf**.
- Chcesz ostrzeżenia o zwierzętach/niebezpiecznych obiektach → zaznacz **Warnings**.

---

## 4. Blok „Face Recognition” (rozpoznawanie twarzy)
**Co robi:** rozpoznaje twarze zapisane w bazie.

**Elementy:**
- **▶ Run**: start rozpoznawania twarzy.
- **⏹ Stop**: zatrzymanie rozpoznawania.
- **Black list filter**: włącza czerwone oznaczanie osób z blacklisty.

**Scenariusze użycia:**
- Chcesz zwykłe rozpoznawanie twarzy → włącz **Run**.
- Chcesz alarmy dla blacklisty → zaznacz **Black list filter**.

---

## 5. Blok „Database Manager” (baza osób)
**Co robi:** pozwala dodawać i usuwać osoby z bazy twarzy.

### Dodawanie osoby
1. Wpisz imię i nazwisko w polu „Add Person”.
2. Kliknij **📸 Capture**.
3. Aplikacja zacznie zapisywać próbki twarzy.
4. Kliknij ponownie, aby zakończyć zapis.

### Usuwanie osoby
1. Wybierz osobę z listy „Manage People”.
2. Kliknij **🗑 Delete**.

### Blacklist (checkbox)
- Po wybraniu osoby zaznacz/odznacz **Black list**.
- Zmiana zapisuje się w CSV (0/1).

---

## 6. Blok „Display Metrics” (metryki wydajności)
**Co robi:** włącza/wyłącza nakładki z wydajnością (FPS/CPU/GPU/Inference).

- **Metrics ON** → pokazuje metryki.
- **Metrics OFF** → ukrywa metryki.

---

## 7. Okno „Settings” (ustawienia)
**Jak otworzyć:** kliknij **⚙ Settings** w bloku kamery.

### Zakładka GPU Optimization
- Wybór precyzji (FP32/FP16/INT8).
- Opcje GPU (Tensor Cores, Memory Layout).

### Zakładka Performance
- Włączenie metryk.
- Wybór trybu wykresów: Off / Compact / Full.

### Zakładka Display
- Włączenie/wyłączenie logowania CSV z metrykami.

### Zakładka Detection
- **YOLO confidence threshold** – próg wykrycia obiektów.
- **Face detection threshold** – próg detekcji twarzy (MTCNN).
- **Face recognition threshold** – próg rozpoznania twarzy.
- **Top labels (1–3)** – liczba wyświetlanych etykiet twarzy.

---

## 8. Wyświetlanie obrazu (oryginalny vs. przetworzony)
- Jeśli YOLO/Face Recognition są wyłączone → wyświetlany jest obraz oryginalny.
- Jeśli są włączone → obraz jest przetwarzany i rysowane są ramki/etykiety.

---

## 9. Zamykanie aplikacji
- Zamknij okno standardowo (X).
- Kamera zostanie zwolniona automatycznie.

---

## 10. Jak rozpoznać alerty
- **Czerwony** → osoba z blacklisty lub obiekt krytyczny YOLO (np. `knife`).
- **Żółty** → zwierzę wykryte przez YOLO (gdy Warnings włączone).
- **Zielony** → osoba rozpoznana jako znana.
- **Pomarańczowy** → osoba nierozpoznana (Unknown).
