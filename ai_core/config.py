import torch
import os
import platform
from pathlib import Path

class Config:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.device = self._get_device()
        # Ustalanie ścieżki względem tego pliku config.py
        # ai_core/config.py -> parent = ai_core -> parent = project_root
        self.project_root = Path(__file__).parent.parent
        self.known_faces_dir = self.project_root / "known_faces"
        self.blacklist_csv_path = self.known_faces_dir / "blacklist.csv"
        self.detections_dir = self.project_root / "detections"
        self.detections_csv_path = self.detections_dir / "detections_log.csv"
        
        # Face recognition thresholds
        self.confidence_threshold = 0.6  # Rozpoznanie (match) threshold
        self.detection_threshold = 0.9   # Detekcja twarzy (MTCNN) threshold
        self.face_top_k = 1              # 1-3 etykiety najbardziej prawdopodobnych osób
        self.face_process_interval = 5   # Co ile klatek liczyć rozpoznawanie (performance)
        self.face_downscale = 0.4        # Skala obrazu do rozpoznawania (0.4 = duże odciążenie)

        # AI input sizing (lower = faster, lower accuracy)
        self.ai_process_width = 480

        # YOLO detection threshold
        self.yolo_confidence_threshold = 0.5
        self.image_size = 160
        self.margin = 20
        
        # GPU Optimization Settings
        self.quantization_mode = 'fp16'  # 'fp32', 'fp16', 'int8'
        self.use_tensor_cores = True
        self.optimize_memory = False
        
        # Performance Monitoring
        self.show_performance_metrics = True
        self.graph_mode = 'off'  # 'off', 'compact', 'full'
        self.compact_metric = 'fps'  # 'fps', 'cpu', 'gpu', 'inference'

        # Alerts/filters
        self.show_blacklist_alerts = True
        self.show_yolo_warnings = False
        
        # CSV Logging (for debugging performance impact of display)
        self.enable_csv_logging = False
        self.csv_log_path = self.project_root / "performance_debug.csv"
        
    def _log(self, msg):
        """Log to both terminal and GUI callback if available"""
        print(msg)
        if self.log_callback:
            self.log_callback(msg)
        
    def _get_device(self):
        if torch.cuda.is_available():
            msg = f"GPU available: {torch.cuda.get_device_name(0)}"
            print(f"[CUDA] {msg}")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("[MPS] Using Apple Silicon GPU")
            return 'mps'
        else:
            print("[CPU] Using CPU (No GPU detected)")
            return 'cpu'
    
    def ensure_dirs(self):
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)

config = Config()