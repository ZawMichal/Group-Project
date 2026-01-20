# Dokumentacja techniczna GUI (tkinter)

Poniżej znajduje się pełny opis **100% kodu** pliku gui_app.py, z podziałem na funkcjonalności i z blokami kodu.

---

## Importy i zależności
**Cel:** przygotowanie bibliotek GUI, obrazu, wątków, czasu oraz modułów AI i monitoringu.

```python
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import datetime
import psutil

from ai_core.engine import AIEngine
from ai_core.config import config
from ai_core.performance_monitor import PerformanceMonitor
```

---

## Klasa SettingsWindow (blok ustawień)
**Cel:** okno ustawień dotyczące optymalizacji GPU, monitoringu i logowania.

### Inicjalizacja okna
Tworzy okno podrzędne, ustawia rozmiar, blokuje interakcje z oknem głównym i uruchamia zakładki.

```python
class SettingsWindow:
    """Settings popup window for GPU optimization and performance tuning"""
    def __init__(self, parent, engine, main_app):
        self.engine = engine
        self.main_app = main_app
        self.window = tk.Toplevel(parent)
        self.window.title("Settings")
        self.window.geometry("500x600")
        self.window.resizable(False, False)
        
        # Center window on parent
        self.window.transient(parent)
        self.window.grab_set()
        
        self.create_tabs()
```

### Zakładki i przyciski
Tworzy `Notebook` z trzema zakładkami oraz przyciski „Apply” i „Close”.

```python
    def create_tabs(self):
        """Create notebook with tabs"""
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # GPU Optimization Tab
        gpu_frame = ttk.Frame(notebook)
        notebook.add(gpu_frame, text="GPU Optimization")
        self.create_gpu_tab(gpu_frame)
        
        # Performance Tab
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text="Performance")
        self.create_performance_tab(perf_frame)
        
        # Display Tab
        display_frame = ttk.Frame(notebook)
        notebook.add(display_frame, text="Display")
        self.create_display_tab(display_frame)
        
        # Buttons
        button_frame = tk.Frame(self.window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(button_frame, text="Apply", command=self.apply_settings, 
             bg="#27ae60", fg="white", width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Close", command=self.window.destroy,
             bg="#95a5a6", fg="white", width=10).pack(side=tk.LEFT, padx=5)
```

### Zakładka GPU
Ustawia tryb precyzji (FP32/FP16/INT8) i opcje GPU (Tensor Cores, układ pamięci).

```python
    def create_gpu_tab(self, parent):
        """GPU Optimization settings"""
        frame = ttk.LabelFrame(parent, text="GPU Precision Mode", padding=10)
        frame.pack(fill=tk.BOTH, padx=10, pady=10)
        
        self.quantization_var = tk.StringVar(value=config.quantization_mode)
        
        ttk.Radiobutton(frame, text="FP32 (Full Precision) - Slower, Most Accurate", 
                       variable=self.quantization_var, value="fp32").pack(anchor=tk.W, pady=5)
        ttk.Radiobutton(frame, text="FP16 (Half Precision) - 2× Faster, <0.5% Accuracy Loss", 
                       variable=self.quantization_var, value="fp16").pack(anchor=tk.W, pady=5)
        ttk.Radiobutton(frame, text="INT8 (8-bit Quantization) - 3-4× Faster, 1-2% Accuracy Loss", 
                       variable=self.quantization_var, value="int8").pack(anchor=tk.W, pady=5)
        
        info_label = tk.Label(frame, text="Note: Changing this requires reloading face recognition models.\nWill restart face system on Apply.",
                            bg="#ecf0f1", fg="#7f8c8d", font=("Arial", 8), justify=tk.LEFT, wraplength=450)
        info_label.pack(pady=10)
        
        # Tensor Cores
        frame2 = ttk.LabelFrame(parent, text="GPU Features", padding=10)
        frame2.pack(fill=tk.BOTH, padx=10, pady=10)
        
        self.tensor_cores_var = tk.BooleanVar(value=config.use_tensor_cores)
        ttk.Checkbutton(frame2, text="Enable Tensor Cores (if available)", 
                       variable=self.tensor_cores_var).pack(anchor=tk.W, pady=5)
        
        self.optimize_mem_var = tk.BooleanVar(value=config.optimize_memory)
        ttk.Checkbutton(frame2, text="Optimize Memory Layout", 
                       variable=self.optimize_mem_var).pack(anchor=tk.W, pady=5)
```

### Zakładka Performance
Włącza/wyłącza metryki i wybór trybu wykresów (off/compact/full).

```python
    def create_performance_tab(self, parent):
        """Performance monitoring settings"""
        frame = ttk.LabelFrame(parent, text="Performance Monitoring", padding=10)
        frame.pack(fill=tk.BOTH, padx=10, pady=10)
        
        self.show_metrics_var = tk.BooleanVar(value=config.show_performance_metrics)
        ttk.Checkbutton(frame, text="Show Performance Metrics (FPS/CPU/GPU/Inference)", 
                       variable=self.show_metrics_var).pack(anchor=tk.W, pady=5)
        
        # Graph mode
        frame2 = ttk.LabelFrame(parent, text="Performance Graphs", padding=10)
        frame2.pack(fill=tk.BOTH, padx=10, pady=10)
        
        self.graph_mode_var = tk.StringVar(value=config.graph_mode)
        ttk.Radiobutton(frame2, text="Off - No graphs", 
                       variable=self.graph_mode_var, value="off").pack(anchor=tk.W, pady=3)
        ttk.Radiobutton(frame2, text="Compact - One metric as line graph",
                       variable=self.graph_mode_var, value="compact").pack(anchor=tk.W, pady=3)
        ttk.Radiobutton(frame2, text="Full - All metrics as graphs",
                       variable=self.graph_mode_var, value="full").pack(anchor=tk.W, pady=3)
        
        # Compact metric selector
        compact_frame = ttk.Frame(frame2)
        compact_frame.pack(fill=tk.X, pady=5)
        ttk.Label(compact_frame, text="Compact metric:").pack(side=tk.LEFT)
        self.compact_metric_var = tk.StringVar(value=config.compact_metric)
        metric_combo = ttk.Combobox(compact_frame, textvariable=self.compact_metric_var,
                                   values=["fps", "cpu", "gpu", "inference"],
                                   width=12, state="readonly")
        metric_combo.pack(side=tk.LEFT, padx=5)
        
        # Current device info
        frame3 = ttk.LabelFrame(parent, text="System Information", padding=10)
        frame3.pack(fill=tk.BOTH, padx=10, pady=10)
        
        device_text = f"Current Device: {config.device.upper()}\nQuantization: {config.quantization_mode.upper()}"
        device_label = tk.Label(frame3, text=device_text, bg="#ecf0f1", fg="#212529",
                               font=("Consolas", 9), justify=tk.LEFT)
        device_label.pack(pady=10)
```

### Zakładka Display
Włącza/wyłącza CSV logging z metrykami i stanem silników.

```python
    def create_display_tab(self, parent):
        """Display and UI settings"""
        frame = ttk.LabelFrame(parent, text="Display Options", padding=10)
        frame.pack(fill=tk.BOTH, padx=10, pady=10)
        
        # CSV Logging for debugging
        ttk.Label(frame, text="Performance Debugging:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(10, 5))
        
        self.enable_csv_logging_var = tk.BooleanVar(value=config.enable_csv_logging)
        ttk.Checkbutton(frame, text="Enable CSV Performance Log (performance_debug.csv)", 
                       variable=self.enable_csv_logging_var).pack(anchor=tk.W, pady=5)
        
        ttk.Label(frame, text="Logs configuration, AI state, and performance metrics\nto debug if display affects performance.",
                 font=("Arial", 8), foreground="gray").pack(anchor=tk.W, pady=(0, 15))
```

### Zastosowanie ustawień
Aktualizuje konfigurację, włącza/wyłącza CSV logger oraz przeładowuje system twarzy przy zmianie kwantyzacji.

```python
    def apply_settings(self):
        """Apply all settings"""
        new_quantization = self.quantization_var.get()
        config.show_performance_metrics = self.show_metrics_var.get()
        config.use_tensor_cores = self.tensor_cores_var.get()
        config.optimize_memory = self.optimize_mem_var.get()
        config.graph_mode = self.graph_mode_var.get()
        config.compact_metric = self.compact_metric_var.get()
        config.enable_csv_logging = self.enable_csv_logging_var.get()
        
        # Start/stop CSV logging if needed
        if config.enable_csv_logging and not hasattr(self.main_app, 'csv_logger'):
            from ai_core.csv_logger import PerformanceCSVLogger
            self.main_app.csv_logger = PerformanceCSVLogger(config.csv_log_path)
            self.main_app.csv_logger.start()
            self.main_app._log(f"CSV logging started: {config.csv_log_path}")
        elif not config.enable_csv_logging and hasattr(self.main_app, 'csv_logger'):
            self.main_app.csv_logger.stop()
            del self.main_app.csv_logger
            self.main_app._log("CSV logging stopped")
        
        # If quantization mode changed, reload face system
        if new_quantization != config.quantization_mode:
            config.quantization_mode = new_quantization
            self.main_app._log(f"Switching to {new_quantization} quantization...")
            
            # Reload face system in background
            def reload_faces():
                try:
                    self.engine._face_system = None  # Force reload
                    _ = self.engine.face_system  # Trigger reload
                    self.main_app._log(f"Face system reloaded with {new_quantization}")
                except Exception as e:
                    self.main_app._log(f"Error reloading face system: {e}")
            
            threading.Thread(target=reload_faces, daemon=True).start()
        
        self.window.destroy()
```

### Reset ustawień
Przywraca wartości domyślne w GUI.

```python
    def reset_to_defaults(self):
        """Reset to default settings"""
        self.quantization_var.set("fp16")
        self.show_metrics_var.set(True)
        self.tensor_cores_var.set(True)
        self.optimize_mem_var.set(False)
        self.graph_mode_var.set("off")
        self.compact_metric_var.set("fps")
        self.enable_csv_logging_var.set(False)
```

---

## Klasa AIVisionAPP (główna aplikacja)
**Cel:** tworzy okno, panele, obsługuje kamerę, AI, bazę osób, metryki i wyświetlanie.

### Inicjalizacja i stan aplikacji (blok główny okna)
Ustawia tytuł, rozmiar okna, tworzy monitor wydajności, inicjuje AI i uruchamia pętlę odświeżania.

```python
class AIVisionAPP:
    def __init__(self, root):
        self.root = root
        self.root.title("Group Project v0.3.1")
        self.root.geometry("1300x850")
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()
        self.metrics_update_thread = None
        self.last_resize_time = time.time()
        self.cached_photo_image = None
        self.cached_display_size = (0, 0)
        self.frame_skip_counter = 0
        self.FRAME_SKIP = 2  # Skip every 2nd frame for resize

        # Clear previous CSV log on each app run for fresh captures
        self._clear_csv_log()
        
        self.create_ui()
        
        self._log("Initializing AI Engine...")
        self.engine = AIEngine(log_callback=self._log)
        self._log("AI Engine initialized successfully")
        
        self.engine.enable_yolo = False
        self.engine.enable_faces = False
        
        self.cap = None
        self.is_running = True
        self.recording_mode = False
        self.target_person_name = ""
        self.current_camera_index = None
        
        # Initialize database list now that engine exists
        self._refresh_db_list()
        
        # Uruchamiamy główną pętlę odświeżania GUI (nawet jak nie ma kamery)
        self._update_frame()
```

### Budowa interfejsu (blok głównego okna)
Tworzy nagłówek, panel lewy z kartami i panel prawy z podglądem wideo.

```python
    def create_ui(self):
        # 1. HEADER
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=70)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, text="Projekt Grupowy v0.3.1", 
            font=("Arial", 20, "bold"), bg="#2c3e50", fg="white"
        )
        title_label.pack(side=tk.LEFT, padx=20)

        # 2. GŁÓWNY PODZIAŁ
        self.paned_window = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=5, bg="#bdc3c7")
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # --- LEWY PANEL ---
        self.left_panel_container = tk.Frame(self.paned_window, bg="#ecf0f1", width=450)
        self.paned_window.add(self.left_panel_container, minsize=400)
        
        self.canvas = tk.Canvas(self.left_panel_container, bg="#ecf0f1", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.left_panel_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#ecf0f1", padx=10, pady=10)

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- PRAWY PANEL ---
        self.right_panel = tk.Frame(self.paned_window, bg="#2c3e50")
        self.paned_window.add(self.right_panel, minsize=600)

        # Ekran wideo
        self.video_frame = tk.Frame(self.right_panel, bg="black")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Domyślny tekst gdy nie ma kamery
        self.video_label = tk.Label(
            self.video_frame, 
            bg="black", 
            text="KAMERA WYŁĄCZONA\nWybierz urządzenie z listy po lewej stronie...", 
            fg="#7f8c8d",
            font=("Arial", 14)
        )
        self.video_label.pack(expand=True, fill=tk.BOTH)

        # --- TWORZENIE KART ---
        self.create_camera_card()
        self.create_yolo_card()
        self.create_face_rec_card()
        self.create_database_card()
        self.create_metrics_card()
```

### Generator kart UI (wspólny element dla paneli)
Zapewnia spójny wygląd kart w lewym panelu.

```python
    def create_card_frame(self, title, description):
        card = tk.Frame(self.scrollable_frame, bg="#ffffff", relief=tk.RAISED, bd=2)
        card.pack(fill=tk.X, pady=8, ipadx=5, ipady=5)

        header = tk.Frame(card, bg="#ffffff")
        header.pack(fill=tk.X, padx=10, pady=5)
        
        lbl_title = tk.Label(header, text=title, font=("Arial", 12, "bold"), bg="#ffffff", fg="#2c3e50")
        lbl_title.pack(anchor=tk.W)
        
        lbl_desc = tk.Label(header, text=description, font=("Arial", 9), bg="#ffffff", fg="#7f8c8d")
        lbl_desc.pack(anchor=tk.W)
        
        tk.Frame(card, bg="#ecf0f1", height=1).pack(fill=tk.X, padx=5, pady=5)
        
        controls = tk.Frame(card, bg="#ffffff", padx=10)
        controls.pack(fill=tk.X)
        
        return controls
```

---

## Blok łączenia z kamerą
### UI kamery
Tworzy wybór ID kamery i przycisk „Connect”.

```python
    def create_camera_card(self):
        content = self.create_card_frame("Camera Source", "Set camera device ID")
        
        tk.Label(content, text="Camera ID:", bg="white").pack(side=tk.LEFT)
        self.camera_id_var = tk.StringVar(value="0")
        self.camera_spin = ttk.Spinbox(content, from_=0, to=9, textvariable=self.camera_id_var, width=5, state="readonly")
        self.camera_spin.pack(side=tk.LEFT, padx=5)
        
        btn = tk.Button(content, text="✓ Connect", command=self._connect_camera, bg="#27ae60", fg="white", relief=tk.FLAT)
        btn.pack(side=tk.LEFT, padx=5)
        
        btn_settings = tk.Button(content, text="⚙ Settings", command=self._open_settings, bg="#3498db", fg="white", relief=tk.FLAT)
        btn_settings.pack(side=tk.LEFT, padx=2)
        
        self.camera_status = tk.Label(content, text="Status: Idle", bg="white", fg="#7f8c8d")
        self.camera_status.pack(side=tk.LEFT, padx=10)
```

### Kliknięcie Connect (co się dzieje)
- Pobiera ID z `Spinbox`.
- Uruchamia osobny wątek, aby GUI nie zamarzło.

```python
    def _connect_camera(self):
        """Connect to manually selected camera ID"""
        try:
            cam_id = int(self.camera_id_var.get())
            # Run in background thread to not freeze GUI
            threading.Thread(target=self._start_camera_stream, args=(cam_id,), daemon=True).start()
        except ValueError:
            self._log("Invalid camera ID")
            self.camera_status.config(text="Status: Invalid ID", fg="#e74c3c")
```

### Fizyczne otwarcie kamery
- Tworzy `cv2.VideoCapture`.
- Sprawdza dostępność, ustawia bufor i autofocus.
- Testuje odczyt klatki.

```python
    def _start_camera_stream(self, idx):
        if self.cap: 
            self.cap.release()
            self.cap = None
        
        self._log(f"Attempting to open Camera {idx}...")
        try:
            self.cap = cv2.VideoCapture(idx)
            
            if not self.cap.isOpened():
                self._log(f"Error: Camera {idx} not available or failed to open")
                self.camera_status.config(text=f"Status: Failed (ID {idx})", fg="#e74c3c")
                self.cap = None
                return
            
            # First, check what resolution camera actually supports
            # Don't request unsupported resolutions
            self._log(f"Detecting Camera {idx} native resolution...")
            
            # Get native resolution
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self._log(f"Camera {idx} native: {frame_width}x{frame_height} @ {fps:.0f}fps")
            
            # Only set buffer size - don't change resolution unless it's very low
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Enable autofocus if available
            try:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            except:
                pass
            
            self.current_camera_index = idx
            
            # Test read to ensure camera actually works
            self._log("Testing camera stream...")
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self._log(f"Error: Camera {idx} opened but cannot read frames")
                self.camera_status.config(text=f"Status: Failed (ID {idx})", fg="#e74c3c")
                self.cap.release()
                self.cap = None
                return
            
            self._log(f"Camera {idx} ready - {frame_width}x{frame_height} @ {fps:.0f}fps")
            self.camera_status.config(text=f"Status: Camera {idx} Active", fg="#27ae60")
            
        except Exception as e:
            self._log(f"Error starting camera: {e}")
            self.camera_status.config(text="Status: Error", fg="#e74c3c")
            if self.cap:
                self.cap.release()
                self.cap = None
```

---

## Blok YOLO (detekcja obiektów)
### UI YOLO
Tworzy wybór modelu i opcje „Conf” oraz „Labels”.

```python
    def create_yolo_card(self):
        content = self.create_card_frame("YOLO Detection", "Real-time object detection")
        
        opts_frame = tk.Frame(content, bg="white")
        opts_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(opts_frame, text="Model:", bg="white").pack(side=tk.LEFT)
        self.yolo_model_var = tk.StringVar(value="yolo11n")
        self.yolo_combo = ttk.Combobox(opts_frame, textvariable=self.yolo_model_var, 
                                       values=["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"], 
                                       width=10, state="readonly")
        self.yolo_combo.pack(side=tk.LEFT, padx=5)
        self.yolo_combo.bind("<<ComboboxSelected>>", self._on_yolo_model_change)
        
        checks_frame = tk.Frame(content, bg="white")
        checks_frame.pack(fill=tk.X)
        
        self.var_conf = tk.BooleanVar(value=True)
        self.var_labels = tk.BooleanVar(value=True)
        
        tk.Checkbutton(checks_frame, text="Conf", variable=self.var_conf, command=self._update_yolo_opts, bg="white").pack(side=tk.LEFT)
        tk.Checkbutton(checks_frame, text="Labels", variable=self.var_labels, command=self._update_yolo_opts, bg="white").pack(side=tk.LEFT)

        btn_frame = tk.Frame(content, bg="white", pady=10)
        btn_frame.pack(fill=tk.X)
        
        self.btn_yolo_run = tk.Button(btn_frame, text="▶ Run", command=self._start_yolo, 
                                      bg="#27ae60", fg="white", font=("Arial", 10, "bold"), width=8)
        self.btn_yolo_run.pack(side=tk.LEFT, padx=2)
        
        self.btn_yolo_stop = tk.Button(btn_frame, text="⏹ Stop", command=self._stop_yolo, 
                                       bg="#e74c3c", fg="white", font=("Arial", 10, "bold"), width=8, state=tk.DISABLED)
        self.btn_yolo_stop.pack(side=tk.LEFT, padx=2)
```

### Zmiana modelu
Zmiana w comboboxie odpala wątek, który przełącza model w silniku AI.

```python
    def _on_yolo_model_change(self, event):
        model = self.yolo_model_var.get()
        self._log(f"Loading YOLO model: {model}...")
        def change():
            success = self.engine.change_yolo_model(model)
            if success: self.root.after(0, lambda: self._log(f"Model changed to {model}"))
        threading.Thread(target=change, daemon=True).start()
```

### Przełączniki Conf i Labels
Zmieniają parametry wizualizacji w systemie obiektów.

```python
    def _update_yolo_opts(self):
        self.engine.object_system.show_conf = self.var_conf.get()
        self.engine.object_system.show_labels = self.var_labels.get()
```

### Start/Stop YOLO
Steruje flagą w silniku i stanami przycisków.

```python
    def _start_yolo(self):
        self.engine.enable_yolo = True
        self.btn_yolo_run.config(state=tk.DISABLED)
        self.btn_yolo_stop.config(state=tk.NORMAL)
        self.yolo_combo.config(state=tk.DISABLED)
        self._log("YOLO Detection STARTED")

    def _stop_yolo(self):
        self.engine.enable_yolo = False
        self.btn_yolo_run.config(state=tk.NORMAL)
        self.btn_yolo_stop.config(state=tk.DISABLED)
        self.yolo_combo.config(state="readonly")
        self._log("YOLO Detection STOPPED")
```

---

## Blok Face Recognition
### UI i sterowanie
Uruchamia i zatrzymuje rozpoznawanie twarzy.

```python
    def create_face_rec_card(self):
        content = self.create_card_frame("Face Recognition", "Detect and identify people")
        tk.Label(content, text="Uses Facenet Pytorch & MTCNN", bg="white", fg="gray").pack(anchor=tk.W)

        btn_frame = tk.Frame(content, bg="white", pady=10)
        btn_frame.pack(fill=tk.X)
        
        self.btn_face_run = tk.Button(btn_frame, text="▶ Run", command=self._start_face, 
                                      bg="#27ae60", fg="white", font=("Arial", 10, "bold"), width=8)
        self.btn_face_run.pack(side=tk.LEFT, padx=2)
        
        self.btn_face_stop = tk.Button(btn_frame, text="⏹ Stop", command=self._stop_face, 
                                       bg="#e74c3c", fg="white", font=("Arial", 10, "bold"), width=8, state=tk.DISABLED)
        self.btn_face_stop.pack(side=tk.LEFT, padx=2)

    def _start_face(self):
        self.engine.enable_faces = True
        self.btn_face_run.config(state=tk.DISABLED)
        self.btn_face_stop.config(state=tk.NORMAL)
        self._log("Face Recognition STARTED")

    def _stop_face(self):
        self.engine.enable_faces = False
        self.btn_face_run.config(state=tk.NORMAL)
        self.btn_face_stop.config(state=tk.DISABLED)
        self._log("Face Recognition STOPPED")
```

---

## Blok bazy osób (dodawanie/usuwanie)
### UI bazy
Umożliwia dodawanie i usuwanie osób z bazy danych twarzy.

```python
    def create_database_card(self):
        content = self.create_card_frame("Database Manager", "Add/Remove faces")
        
        tk.Label(content, text="Add Person:", font=("Arial", 9, "bold"), bg="white").pack(anchor=tk.W, pady=(5,0))
        cap_frame = tk.Frame(content, bg="white")
        cap_frame.pack(fill=tk.X)
        
        self.entry_name = tk.Entry(cap_frame, width=15)
        self.entry_name.pack(side=tk.LEFT, padx=(0,5))
        
        self.btn_capture = tk.Button(cap_frame, text="📸 Capture", command=self._toggle_capture, 
                                     bg="#3498db", fg="white")
        self.btn_capture.pack(side=tk.LEFT)

        tk.Frame(content, bg="#ecf0f1", height=1).pack(fill=tk.X, pady=10) 
        
        tk.Label(content, text="Manage People:", font=("Arial", 9, "bold"), bg="white").pack(anchor=tk.W)
        del_frame = tk.Frame(content, bg="white")
        del_frame.pack(fill=tk.X)
        
        self.people_list_var = tk.StringVar()
        self.people_combo = ttk.Combobox(del_frame, textvariable=self.people_list_var, state="readonly", width=15)
        self.people_combo.pack(side=tk.LEFT, padx=(0,5))
        
        self.btn_delete = tk.Button(del_frame, text="🗑 Delete", command=self._delete_person, 
                                    bg="#e67e22", fg="white")
        self.btn_delete.pack(side=tk.LEFT)
        
        tk.Button(del_frame, text="🔄", command=self._refresh_db_list, relief=tk.FLAT, bg="white").pack(side=tk.LEFT)
```

### Przełączanie trybu przechwytywania
- Jeśli tryb nagrywania jest wyłączony, wymaga wpisania nazwy.
- W trybie nagrywania zapisuje próbki w pętli `_update_frame()`.

```python
    def _toggle_capture(self):
        if not self.recording_mode:
            name = self.entry_name.get().strip()
            if not name:
                messagebox.showwarning("Input Error", "Please enter a name first!")
                return
            self.target_person_name = name
            self.recording_mode = True
            self.btn_capture.config(text="⏹ Stop Capture", bg="#e74c3c")
            self.entry_name.config(state=tk.DISABLED)
            self._log(f"Started capturing faces for: {name}")
        else:
            self.recording_mode = False
            self.btn_capture.config(text="📸 Capture", bg="#3498db")
            self.entry_name.config(state=tk.NORMAL)
            self.target_person_name = ""
            self._log("Capture finished.")
            self._refresh_db_list()
            self.engine.reload_faces()
```

### Usuwanie osoby
Wywołuje metodę w managerze i odświeża bazę w UI.

```python
    def _delete_person(self):
        name = self.people_list_var.get()
        if not name: return
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {name}?"):
            success = self.engine.manager.delete_person(name)
            if success:
                self._log(f"Deleted person: {name}")
                self._refresh_db_list()
                self.engine.reload_faces()
            else:
                messagebox.showerror("Error", "Could not delete person.")
```

### Odświeżenie listy
Synchronizuje listę osób z bazą.

```python
    def _refresh_db_list(self):
        people = self.engine.manager.get_people_list()
        self.people_combo['values'] = people
        if people: self.people_combo.current(0)
```

---

## Blok wyświetlania obrazu
### Główna pętla (oryginalny lub przetworzony obraz)
- Zczytuje klatki z kamery.
- Jeśli AI włączone, przetwarza klatkę przez `self.engine.process_frame()`.
- Jeśli AI wyłączone, wyświetla oryginalną klatkę.

```python
    def _update_frame(self):
        """Główna pętla programu - Optimized with frame skipping and metrics"""
        if not self.is_running: return

        self.perf_monitor.frame_start()

        # Jeśli kamera działa i jest otwarta
        if self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.perf_monitor.mark_frame_dropped()
                    self.root.after(10, self._update_frame)
                    return
                
                # Resize only for display, not for processing
                # Target: 640 width for balance between quality and performance
                h, w = frame.shape[:2]
                if w > 800:
                    scale = 640 / w
                    process_frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)
                else:
                    process_frame = frame

                # Track what's happening for CSV logging
                is_processing = self.engine.enable_yolo or self.engine.enable_faces
                
                # 1. Przetwarzanie w silniku AI TYLKO gdy włączone
                if is_processing:
                    processed_frame = self.engine.process_frame(process_frame)
                else:
                    processed_frame = process_frame
                
                # 2. Logika nagrywania (save faces every 1 second, not every frame)
                if self.recording_mode:
                    if self.target_person_name:
                        # Save every 1s (~10 frames at ~100Hz)
                        if int(time.time() * 10) % 10 == 0:
                            path = self.engine.manager.save_training_photo(self.target_person_name, process_frame)
                            if path: self._log(f"Saved photo for {self.target_person_name}")

                # Track if drawing anything
                drawing_metrics = config.show_performance_metrics
                drawing_graph = config.show_performance_metrics and config.graph_mode in ['compact', 'full']
                
                # 3. Add performance metrics if enabled
                if config.show_performance_metrics:
                    processed_frame = self._draw_performance_metrics(processed_frame)
                
                # 4. Wyświetlanie (Optimized - Frame skipping and faster resize)
                self.frame_skip_counter += 1
                
                # Only resize every N frames
                if self.frame_skip_counter >= self.FRAME_SKIP:
                    self.frame_skip_counter = 0
                    self._display_frame(processed_frame)
                    
            except Exception as e:
                # Camera read error - likely camera disconnected
                self._log(f"Camera error: {e}")
                self.cap = None
        
        # Update system metrics periodically (every 250ms to avoid CPU spikes)
        self.metrics_update_counter = getattr(self, 'metrics_update_counter', 0) + 1
        if self.metrics_update_counter >= 25:  # ~250ms at 100Hz loop
            self.metrics_update_counter = 0
            threading.Thread(target=self.perf_monitor.update_system_metrics, daemon=True).start()
        
        self.perf_monitor.frame_end()
        
        # CSV Logging - log current state and performance
        if config.enable_csv_logging and hasattr(self, 'csv_logger'):
            try:
                metrics_dict = self.perf_monitor.get_metrics()
                state = {
                    'ai_running': int(is_processing),
                    'face_loading': int(getattr(self.engine._face_system, '_loading', False) if self.engine._face_system else False),
                    'objects_loading': int(getattr(self.engine._object_system, '_loading', False) if self.engine._object_system else False),
                    'drawing_metrics': int(drawing_metrics),
                    'drawing_graph': int(drawing_graph)
                }
                self.csv_logger.log_frame(config, state, metrics_dict)
            except Exception as e:
                print(f"CSV logging error: {e}")
        
        # Pętla kręci się co 10ms, niezależnie czy kamera działa
        self.root.after(10, self._update_frame)
```

### Renderowanie klatki
Konwertuje BGR→RGB, skaluje do okna, stosuje ImageTk i wyświetla.

```python
    def _display_frame(self, processed_frame):
        """Display frame with optimized resize and PhotoImage handling"""
        try:
            rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            
            # Get current window size
            w_win = self.video_frame.winfo_width()
            h_win = self.video_frame.winfo_height()
            
            # Zabezpieczenie przed minimalnym oknem
            if w_win > 10 and h_win > 10:
                # Check if window was resized
                if (w_win, h_win) != self.cached_display_size:
                    self.cached_display_size = (w_win, h_win)
                    self.cached_photo_image = None  # Invalidate cache
                
                # Use cached size if window hasn't changed
                aspect_ratio = 4/3
                new_w = w_win
                new_h = int(new_w / aspect_ratio)
                
                if new_h > h_win:
                    new_h = h_win
                    new_w = int(new_h * aspect_ratio)
                
                # Use faster BILINEAR instead of LANCZOS
                img = img.resize((new_w, new_h), Image.BILINEAR)
                
                # Avoid excessive PhotoImage recreation - only update if needed
                try:
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk, text="")
                except:
                    # If PhotoImage fails, skip this frame update
                    pass
        except Exception as e:
            print(f"Display error: {e}")
```

---

## Blok metryk i wykresów
### Logowanie i przełączanie metryk

```python
    def _log(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")

    def _set_metrics(self, enabled: bool):
        config.show_performance_metrics = bool(enabled)
        state = "enabled" if enabled else "disabled"
        self._log(f"Performance metrics {state}")
```

### Czyszczenie logu CSV

```python
    def _clear_csv_log(self):
        """Remove old performance_debug.csv at app start to avoid stale data"""
        try:
            if config.csv_log_path.exists():
                config.csv_log_path.unlink()
        except Exception as e:
            self._log(f"Could not clear CSV log: {e}")
```

### Rysowanie metryk i wykresów
- Nakładka tekstowa z metrykami.
- Opcjonalne wykresy compact/full.

```python
    def _draw_performance_metrics(self, frame):
        """Draw performance metrics on frame (translucent box with sharp text)"""
        try:
            metrics = self.perf_monitor.get_metrics_string()
            
            # Prepare text
            text_lines = [
                f"FPS: {metrics['fps']}",
                f"CPU: {metrics['cpu']}",
                f"GPU: {metrics['gpu']}",
                f"Inference: {metrics['inference_ms']}",
                f"Dropped: {metrics['frames_dropped']}"
            ]
            
            # Create overlay
            overlay = frame.copy()
            
            # Draw semi-transparent background
            box_h = 5 + (len(text_lines) * 20) + 5
            cv2.rectangle(overlay, (10, 10), (220, 10 + box_h), (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Draw text (sharper font)
            for i, text in enumerate(text_lines):
                y_pos = 25 + (i * 20)
                cv2.putText(frame, text, (20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            
            # Draw graphs if enabled
            if config.graph_mode == 'compact':
                frame = self._draw_compact_graph(frame)
            elif config.graph_mode == 'full':
                frame = self._draw_full_graphs(frame)
            
            return frame
        except Exception as e:
            print(f"Metrics drawing error: {e}")
            return frame
```

```python
    def _draw_compact_graph(self, frame):
        """Draw single metric as line graph"""
        try:
            h, w = frame.shape[:2]
            graph_w, graph_h = 300, 100
            graph_x = w - graph_w - 10
            graph_y = h - graph_h - 10
            
            histories = self.perf_monitor.get_histories()
            metric = config.compact_metric
            data = histories.get(metric, [])
            
            if not data or len(data) < 2:
                return frame
            
            # Create graph background
            overlay = frame.copy()
            cv2.rectangle(overlay, (graph_x, graph_y), (w - 10, h - 10), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # Draw border
            cv2.rectangle(frame, (graph_x, graph_y), (w - 10, h - 10), (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(frame, f"{metric.upper()}", (graph_x + 5, graph_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Scale data
            if data:
                max_val = max(max(data), 1)
                min_val = 0
            else:
                return frame
            
            # Draw line graph
            scale_h = (graph_h - 30) / max(max_val - min_val, 1)
            for i in range(1, len(data)):
                x1 = graph_x + 5 + int((i - 1) * (graph_w - 15) / len(data))
                x2 = graph_x + 5 + int(i * (graph_w - 15) / len(data))
                y1 = graph_y + graph_h - 5 - int((data[i-1] - min_val) * scale_h)
                y2 = graph_y + graph_h - 5 - int((data[i] - min_val) * scale_h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            return frame
        except Exception as e:
            print(f"Compact graph error: {e}")
            return frame
```

```python
    def _draw_full_graphs(self, frame):
        """Draw all metrics as graphs below the numbers"""
        try:
            h, w = frame.shape[:2]
            graph_h = 80
            total_height = graph_h + 10
            
            # Create overlay area at bottom
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, h - total_height - 10), (w - 10, h - 10), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # Draw border
            cv2.rectangle(frame, (10, h - total_height - 10), (w - 10, h - 10), (0, 255, 0), 2)
            
            histories = self.perf_monitor.get_histories()
            metrics = ['fps', 'cpu', 'gpu', 'inference']
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            
            graph_x_start = 20
            graph_w_each = (w - 50) // 4
            
            for idx, (metric, color) in enumerate(zip(metrics, colors)):
                data = histories.get(metric, [])
                if not data or len(data) < 2:
                    continue
                
                graph_x = graph_x_start + idx * graph_w_each
                graph_y = h - total_height
                
                # Label
                cv2.putText(frame, metric.upper(), (graph_x + 5, graph_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                
                # Scale
                if metric == 'inference':
                    max_val = max(max(data), 10)
                elif metric in ['cpu', 'gpu']:
                    max_val = 100
                else:  # fps
                    max_val = max(max(data), 30)
                
                scale_h = (graph_h - 25) / max(max_val, 1)
                
                # Draw line
                for i in range(1, len(data)):
                    x1 = graph_x + 5 + int((i - 1) * (graph_w_each - 10) / len(data))
                    x2 = graph_x + 5 + int(i * (graph_w_each - 10) / len(data))
                    y1 = h - 5 - int(data[i-1] * scale_h)
                    y2 = h - 5 - int(data[i] * scale_h)
                    cv2.line(frame, (x1, y1), (x2, y2), color, 1)
            
            return frame
        except Exception as e:
            print(f"Full graphs error: {e}")
            return frame
```

---

## Blok ustawień (uruchomienie okna SettingsWindow)
Wywoływany przyciskiem „Settings” w bloku kamery.

```python
    def _open_settings(self):
        """Open settings window"""
        SettingsWindow(self.root, self.engine, self)
```

**Uwaga:** w pliku metoda `_open_settings()` występuje dwa razy w tej samej klasie i obie wersje są identyczne. W praktyce ostatnia definicja nadpisuje poprzednią, więc zachowanie jest takie samo.

---

## Blok zamknięcia aplikacji
Zatrzymuje pętlę i zwalnia kamerę.

```python
    def on_close(self):
        self.is_running = False
        if self.cap: self.cap.release()
        self.root.destroy()
```

---

## Uruchomienie programu (main)
Tworzy okno i uruchamia pętlę zdarzeń tkinter.

```python
if __name__ == "__main__":
    root = tk.Tk()
    app = AIVisionAPP(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
```

---

## Aktualizacje funkcjonalne (GUI)

### Blacklist (GUI + CSV)
- Każda osoba ma przypisany status blacklist w pliku CSV.
- W sekcji bazy osób dodano checkbox „Black list”, który ustawia 0/1 w CSV.
- W sekcji Face Recognition dodano przełącznik „Black list filter”, który rysuje czerwone obramowania i wykrzyknik dla osób z blacklisty.

```python
# GUI: checkbox blacklist w bazie
self.blacklist_var = tk.BooleanVar(value=False)
tk.Checkbutton(blacklist_frame, text="Black list", variable=self.blacklist_var,
               command=self._toggle_blacklist_status, bg="white").pack(anchor=tk.W)

# Face Recognition: filtr blacklist
self.var_blacklist_filter = tk.BooleanVar(value=config.show_blacklist_alerts)
tk.Checkbutton(content, text="Black list filter", variable=self.var_blacklist_filter,
               command=self._update_face_opts, bg="white").pack(anchor=tk.W, pady=(5, 0))
```

### YOLO Warnings
- Nowa opcja „Warnings” w bloku YOLO.
- **Żółty** alert dla wykrycia zwierząt.
- **Czerwony** alert dla klas krytycznych (np. `knife`, `baseball bat`).

```python
self.var_warnings = tk.BooleanVar(value=config.show_yolo_warnings)
tk.Checkbutton(checks_frame, text="Warnings", variable=self.var_warnings,
               command=self._update_yolo_opts, bg="white").pack(side=tk.LEFT)
```

### Ustawienia progów detekcji
Dodano zakładkę „Detection” w ustawieniach, gdzie można ustawić progi dla YOLO i Face Recognition oraz liczbę etykiet (Top-1..3).

```python
self.yolo_conf_var = tk.DoubleVar(value=config.yolo_confidence_threshold)
self.face_det_var = tk.DoubleVar(value=config.detection_threshold)
self.face_conf_var = tk.DoubleVar(value=config.confidence_threshold)
self.face_topk_var = tk.IntVar(value=config.face_top_k)
```

### Multi‑label w Face Recognition
Zamiast jednej etykiety można wyświetlić do 3 najbardziej prawdopodobnych osób.

```python
label_lines = [f"{l['name']} ({l['confidence']:.2f})" for l in labels]
```
