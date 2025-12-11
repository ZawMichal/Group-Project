import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import datetime

from ai_core.engine import AIEngine

class RetroAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Group Project v2.1")
        self.root.geometry("1300x850")
        
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

    def create_ui(self):
        # 1. HEADER
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=70)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, text="Projekt Grupowy v2.1", 
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

        # Panel Logów
        log_label = tk.Label(self.right_panel, text="Status & Output:", font=("Arial", 10, "bold"), bg="#2c3e50", fg="white")
        log_label.pack(anchor=tk.W, padx=10)
        
        self.output_text = scrolledtext.ScrolledText(
            self.right_panel, height=8, font=("Consolas", 9), 
            bg="#f8f9fa", fg="#212529"
        )
        self.output_text.pack(fill=tk.X, padx=10, pady=(0, 10))

        # --- TWORZENIE KART ---
        self.create_camera_card()
        self.create_yolo_card()
        self.create_face_rec_card()
        self.create_database_card()

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

    # --- MODUŁY ---
    def create_camera_card(self):
        content = self.create_card_frame("Camera Source", "Set camera device ID")
        
        tk.Label(content, text="Camera ID:", bg="white").pack(side=tk.LEFT)
        self.camera_id_var = tk.StringVar(value="0")
        self.camera_spin = ttk.Spinbox(content, from_=0, to=9, textvariable=self.camera_id_var, width=5, state="readonly")
        self.camera_spin.pack(side=tk.LEFT, padx=5)
        
        btn = tk.Button(content, text="✓ Connect", command=self._connect_camera, bg="#27ae60", fg="white", relief=tk.FLAT)
        btn.pack(side=tk.LEFT, padx=5)
        
        self.camera_status = tk.Label(content, text="Status: Idle", bg="white", fg="#7f8c8d")
        self.camera_status.pack(side=tk.LEFT, padx=10)

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
        
        self.var_fps = tk.BooleanVar(value=True)
        self.var_conf = tk.BooleanVar(value=True)
        self.var_labels = tk.BooleanVar(value=True)
        
        tk.Checkbutton(checks_frame, text="FPS", variable=self.var_fps, command=self._update_yolo_opts, bg="white").pack(side=tk.LEFT)
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

    # --- LOGIKA ---

    def _log(self, msg):
        timestamp = __import__('datetime').datetime.now().strftime("%H:%M:%S")
        self.output_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.output_text.see(tk.END)

    def _connect_camera(self):
        """Connect to manually selected camera ID"""
        try:
            cam_id = int(self.camera_id_var.get())
            # Run in background thread to not freeze GUI
            threading.Thread(target=self._start_camera_stream, args=(cam_id,), daemon=True).start()
        except ValueError:
            self._log("Invalid camera ID")
            self.camera_status.config(text="Status: Invalid ID", fg="#e74c3c")

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

    def _update_frame(self):
        """Główna pętla programu"""
        if not self.is_running: return

        # Jeśli kamera działa i jest otwarta
        if self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    return
                
                # Resize only for display, not for processing
                # Target: 640 width for balance between quality and performance
                h, w = frame.shape[:2]
                if w > 800:
                    scale = 640 / w
                    process_frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)
                else:
                    process_frame = frame

                # 1. Przetwarzanie w silniku AI TYLKO gdy włączone
                if self.engine.enable_yolo or self.engine.enable_faces:
                    processed_frame = self.engine.process_frame(process_frame)
                else:
                    processed_frame = process_frame
                
                # 2. Logika nagrywania
                if self.recording_mode:
                    cv2.circle(processed_frame, (20, 20), 10, (0, 0, 255), -1)
                    cv2.putText(processed_frame, "REC", (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    
                    if self.target_person_name:
                        if int(time.time() * 10) % 5 == 0:
                            # Zapisujemy oryginalną (ew. przeskalowaną) klatkę
                            path = self.engine.manager.save_training_photo(self.target_person_name, process_frame)
                            if path: self._log(f"Saved photo for {self.target_person_name}")

                # 3. Wyświetlanie
                try:
                    rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    
                    # Skalowanie do rozmiaru okna w GUI
                    w_win = self.video_frame.winfo_width()
                    h_win = self.video_frame.winfo_height()
                    
                    # Zabezpieczenie przed minimalnym oknem
                    if w_win > 10 and h_win > 10:
                        # Zachowaj proporcje 4:3 żeby nie rozciągać
                        aspect_ratio = 4/3
                        new_w = w_win
                        new_h = int(new_w / aspect_ratio)
                        
                        if new_h > h_win:
                            new_h = h_win
                            new_w = int(new_h * aspect_ratio)
                            
                        img = img.resize((new_w, new_h), Image.LANCZOS)
                    
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk, text="")
                except Exception as e:
                    print(f"Display error: {e}")
            except Exception as e:
                # Camera read error - likely camera disconnected
                self._log(f"Camera error: {e}")
                self.cap = None
        
        # Pętla kręci się co 10ms, niezależnie czy kamera działa
        self.root.after(10, self._update_frame)

    # --- OBSŁUGA POZOSTAŁYCH PRZYCISKÓW ---
    def _on_yolo_model_change(self, event):
        model = self.yolo_model_var.get()
        self._log(f"Loading YOLO model: {model}...")
        def change():
            success = self.engine.change_yolo_model(model)
            if success: self.root.after(0, lambda: self._log(f"Model changed to {model}"))
        threading.Thread(target=change, daemon=True).start()

    def _update_yolo_opts(self):
        self.engine.object_system.show_fps = self.var_fps.get()
        self.engine.object_system.show_conf = self.var_conf.get()
        self.engine.object_system.show_labels = self.var_labels.get()

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

    def _refresh_db_list(self):
        people = self.engine.manager.get_people_list()
        self.people_combo['values'] = people
        if people: self.people_combo.current(0)

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

    def on_close(self):
        self.is_running = False
        if self.cap: self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RetroAIApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()