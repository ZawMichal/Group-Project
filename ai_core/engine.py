import cv2
import time
from .object_system import ObjectSystem
from .face_system import FaceSystem
from .manager import PersonManager

class AIEngine:
    """
    Główny silnik łączący YOLO, Face Recognition i Managera Plików.
    """
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self._log("[ENGINE] Initializing AI Engine...")
        
        self.manager = PersonManager()
        self._log("[ENGINE] Manager initialized")
        
        # Lazy loading - initialize to None, load on first use
        self._object_system = None
        self._face_system = None
        
        self.enable_yolo = False
        self.enable_faces = False
        
        self._log("[ENGINE] Ready. Awaiting camera connection...")

    def _log(self, msg):
        """Log to both terminal and GUI"""
        print(msg)
        if self.log_callback:
            self.log_callback(msg)

    @property
    def object_system(self):
        """Lazy load YOLO model on first access"""
        if self._object_system is None:
            self._log("[YOLO] Loading model (this may take a moment)...")
            start = time.time()
            self._object_system = ObjectSystem(model_variant="yolo11n.pt", log_callback=self.log_callback)
            elapsed = time.time() - start
            self._log(f"[YOLO] Model loaded in {elapsed:.1f}s")
        return self._object_system
    
    @property
    def face_system(self):
        """Lazy load FaceNet model on first access"""
        if self._face_system is None:
            self._log("[FACE] Loading FaceNet models (this may take a moment)...")
            start = time.time()
            self._face_system = FaceSystem(log_callback=self.log_callback)
            elapsed = time.time() - start
            self._log(f"[FACE] Models loaded in {elapsed:.1f}s")
        return self._face_system

    def process_frame(self, frame):
        """Główna pętla przetwarzania obrazu."""
        annotated_frame = frame.copy()

        # 1. Obiekty (YOLO)
        if self.enable_yolo:
            yolo_results = self.object_system.process_frame(frame)
            annotated_frame = self.object_system.draw_results(annotated_frame, yolo_results)

        # 2. Twarze (FaceNet)
        if self.enable_faces:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_system.recognize_faces(rgb_frame)
            annotated_frame = self.face_system.draw_results(annotated_frame, face_results)

        return annotated_frame

    def reload_faces(self):
        """Wymusza ponowne załadowanie bazy twarzy (po edycji/dodaniu)."""
        self._log("[FACE] Reloading face database...")
        self.face_system.load_known_faces()
        self._log("[FACE] Face database reloaded")
        
    def change_yolo_model(self, model_name):
        """Zmienia model YOLO (n, s, m, l, x)."""
        self._log(f"[YOLO] Switching to {model_name}...")
        variant = f"{model_name}.pt" # np. yolo11s.pt
        result = self.object_system.change_model(variant)
        if result:
            self._log(f"[YOLO] Model switched to {model_name}")
        else:
            self._log(f"[YOLO] Failed to switch to {model_name}")
        return result