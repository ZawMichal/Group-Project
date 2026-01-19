# ai_core/object_system.py

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
import time
import torch
from .config import config

class ObjectSystem:
    """
    Klasa obsługująca YOLO z możliwością konfiguracji wyświetlania.
    """
    def __init__(self, model_variant="yolo11n.pt", enable_tracking=True, log_callback=None):
        self.log_callback = log_callback
        self._log(f"Initializing YOLO model: {model_variant}")
        self.model = self._load_model(model_variant)
        self.enable_tracking = enable_tracking
        self.conf = 0.5
        
        # Apply quantization if on GPU
        if config.device == 'cuda':
            self._apply_gpu_optimization()
        
        # --- Bajery wizualne z poprzedniej apki ---
        self.show_conf = True
        self.show_labels = True
        # Disable YOLO's own FPS overlay to avoid double counters (GUI metrics handles FPS)
        self.show_fps = False
        self._fps_prev_time = time.time()
    
    def _log(self, msg):
        """Log to both terminal and GUI"""
        print(f"[YOLO] {msg}")
        if self.log_callback:
            self.log_callback(f"[YOLO] {msg}")
    
    def _apply_gpu_optimization(self):
        """Apply GPU optimizations"""
        try:
            if config.use_tensor_cores:
                # Let Ultralytics handle precision/autocast to avoid dtype mismatches in tracker
                self._log("Using default Ultralytics precision (no manual half) to keep tracker stable")
                # Ensure model is on GPU if available
                try:
                    self.model.to('cuda')
                except Exception:
                    pass
                self._log("GPU ready (Tensor Cores via autocast)")
        except Exception as e:
            self._log(f"GPU optimization failed: {e}")
        
    def _load_model(self, variant):
        try:
            return YOLO(variant)
        except Exception as e:
            self._log(f"Failed to load {variant}, using nano: {e}")
            return YOLO("yolo11n.pt")

    def change_model(self, new_variant):
        self._log(f"Switching model to: {new_variant}")
        try:
            self.model = self._load_model(new_variant)
            return True
        except Exception:
            self._log("Model switch failed")
            return False

    def process_frame(self, frame):
        if self.enable_tracking:
            results = self.model.track(frame, persist=True, verbose=False, conf=self.conf)
        else:
            results = self.model.predict(frame, verbose=False, conf=self.conf)  
        return results[0]

    def draw_results(self, frame, results):
        """Rysuje wyniki uwzględniając flagi konfiguracyjne."""
        annotator = Annotator(frame, line_width=2)
        
        # 1. Rysowanie Pudełek
        if results.boxes is not None:
            for box in results.boxes:
                b = box.xyxy[0]
                c = int(box.cls)
                conf = float(box.conf)
                
                # Budowanie etykiety wg ustawień
                label = ""
                if self.show_labels:
                    label += f"{self.model.names[c]}"
                if self.show_conf:
                    label += f" {conf:.2f}"
                if box.id is not None:
                    label = f"#{int(box.id)} " + label
                
                annotator.box_label(b, label, color=colors(c, True))
        
        # 2. (FPS overlay disabled; handled by GUI metrics)
        
        return frame