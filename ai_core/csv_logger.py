"""CSV Logger for performance debugging - tracks if display affects performance"""

import csv
import threading
from datetime import datetime
from pathlib import Path
from queue import Queue
from time import time


class PerformanceCSVLogger:
    """
    Thread-safe CSV logger for performance debugging.
    Logs configuration state, AI operations, display operations, and performance metrics.
    Uses queue to avoid blocking main thread.
    """
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.queue = Queue(maxsize=1000)
        self.is_running = False
        self.writer_thread = None
        self.file_handle = None
        self.csv_writer = None
        
        # Initialize CSV file with headers
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers"""
        try:
            self.file_handle = open(self.log_path, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.file_handle, fieldnames=[
                'timestamp',
                'quantization_mode',
                'use_tensor_cores',
                'optimize_memory',
                'metrics_enabled',
                'graph_mode',
                'compact_metric',
                'ai_running',
                'face_loading',
                'objects_loading',
                'drawing_metrics',
                'drawing_graph',
                'fps',
                'cpu_percent',
                'gpu_percent',
                'inference_time_ms'
            ])
            self.csv_writer.writeheader()
            self.file_handle.flush()
        except Exception as e:
            print(f"[CSV Logger] Error initializing CSV: {e}")
    
    def start(self):
        """Start the background writer thread"""
        if not self.is_running:
            self.is_running = True
            self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self.writer_thread.start()
    
    def stop(self):
        """Stop the background writer thread"""
        self.is_running = False
        if self.writer_thread:
            self.writer_thread.join(timeout=2)
        if self.file_handle:
            self.file_handle.close()
    
    def _writer_loop(self):
        """Background thread that writes queued records"""
        while self.is_running:
            try:
                # Get item with timeout to check is_running regularly
                row = self.queue.get(timeout=0.1)
                if row is None:  # Sentinel value for shutdown
                    break
                
                if self.csv_writer and self.file_handle:
                    self.csv_writer.writerow(row)
                    self.file_handle.flush()
            except Exception:
                pass  # Timeout or other issue, continue
    
    def log_frame(self, config, state: dict, metrics: dict):
        """
        Queue a frame's performance data for logging.
        
        Args:
            config: Config object with settings
            state: Dict with keys: ai_running, face_loading, objects_loading, 
                   drawing_metrics, drawing_graph
            metrics: Dict with keys: fps, cpu_percent, gpu_percent, inference_time_ms
        """
        if not self.is_running:
            return
        
        try:
            row = {
                'timestamp': datetime.now().isoformat(timespec='milliseconds'),
                'quantization_mode': config.quantization_mode,
                'use_tensor_cores': int(config.use_tensor_cores),
                'optimize_memory': int(config.optimize_memory),
                'metrics_enabled': int(config.show_performance_metrics),
                'graph_mode': config.graph_mode,
                'compact_metric': config.compact_metric,
                'ai_running': int(state.get('ai_running', 0)),
                'face_loading': int(state.get('face_loading', 0)),
                'objects_loading': int(state.get('objects_loading', 0)),
                'drawing_metrics': int(state.get('drawing_metrics', 0)),
                'drawing_graph': int(state.get('drawing_graph', 0)),
                'fps': round(metrics.get('fps', 0), 2),
                'cpu_percent': round(metrics.get('cpu_percent', 0), 1),
                'gpu_percent': round(metrics.get('gpu_percent', 0), 1),
                'inference_time_ms': round(metrics.get('inference_time_ms', 0), 1)
            }
            
            # Non-blocking put (drop oldest if queue full)
            try:
                self.queue.put_nowait(row)
            except:
                try:
                    self.queue.get_nowait()  # Drop oldest
                    self.queue.put_nowait(row)
                except:
                    pass
        except Exception as e:
            print(f"[CSV Logger] Error logging frame: {e}")
