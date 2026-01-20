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
    WARNING_ANIMALS = {
        "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe"
    }
    WARNING_CRITICAL = {"knife", "baseball bat"}

    def __init__(self, model_variant="yolo11n.pt", enable_tracking=True, log_callback=None):
        self.log_callback = log_callback
        self._log(f"Initializing YOLO model: {model_variant}")
        self.model = self._load_model(model_variant)
        self.enable_tracking = enable_tracking
        self.conf = config.yolo_confidence_threshold
        
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
        # Keep conf in sync with config (e.g., changed in Settings)
        self.conf = config.yolo_confidence_threshold
        if self.enable_tracking:
            results = self.model.track(frame, persist=True, verbose=False, conf=self.conf)
        else:
            results = self.model.predict(frame, verbose=False, conf=self.conf)  
        return results[0]

    def draw_results(self, frame, results):
        """Rysuje wyniki uwzględniając flagi konfiguracyjne."""
        annotator = Annotator(frame, line_width=2)

        warning_animals = self.WARNING_ANIMALS
        warning_critical = self.WARNING_CRITICAL
        
        # 1. Rysowanie Pudełek
        if results.boxes is not None:
            for box in results.boxes:
                b = box.xyxy[0]
                c = int(box.cls)
                conf = float(box.conf)
                class_name = self.model.names.get(c, str(c))
                
                # Budowanie etykiety wg ustawień
                label = ""
                if self.show_labels:
                    label += f"{class_name}"
                if self.show_conf:
                    label += f" {conf:.2f}"
                if box.id is not None:
                    label = f"#{int(box.id)} " + label

                # Warnings overlay
                if config.show_yolo_warnings:
                    if class_name in warning_critical:
                        label = f"!! {label}".strip()
                        annotator.box_label(b, label, color=(0, 0, 255))  # Red
                        continue
                    if class_name in warning_animals:
                        label = f"! {label}".strip()
                        annotator.box_label(b, label, color=(0, 255, 255))  # Yellow
                        continue

                annotator.box_label(b, label, color=colors(c, True))
        
        # 2. (FPS overlay disabled; handled by GUI metrics)
        
        return frame

    def analyze_alerts(self, results):
        """Return detected warning/critical classes from YOLO results."""
        warning = set()
        critical = set()

        if results.boxes is None:
            return {'warning': [], 'critical': []}

        for box in results.boxes:
            c = int(box.cls)
            class_name = self.model.names.get(c, str(c))
            if class_name in self.WARNING_CRITICAL:
                critical.add(class_name)
            elif class_name in self.WARNING_ANIMALS:
                warning.add(class_name)

        return {
            'warning': sorted(list(warning)),
            'critical': sorted(list(critical))
        }