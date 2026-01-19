import time
import psutil
import threading
from collections import deque
import numpy as np

class PerformanceMonitor:
    """
    Real-time performance monitoring for GPU/CPU utilization and frame metrics.
    Includes history tracking for graphing.
    """
    def __init__(self, history_size=200):
        self.fps_times = deque(maxlen=30)  # Last 30 frames
        self.last_time = time.time()
        self.frames_processed = 0
        self.frames_dropped = 0
        
        # History for graphing (last N frames)
        self.history_size = history_size
        self.fps_history = deque(maxlen=history_size)
        self.cpu_history = deque(maxlen=history_size)
        self.gpu_history = deque(maxlen=history_size)
        self.inference_history = deque(maxlen=history_size)
        
        # Metrics
        self.current_fps = 0.0
        self.cpu_percent = 0.0
        self.gpu_percent = 0.0
        self.inference_time = 0.0
        
        self.lock = threading.Lock()
        
    def frame_start(self):
        """Call at start of frame processing"""
        self.frame_start_time = time.time()
    
    def frame_end(self):
        """Call at end of frame processing"""
        with self.lock:
            current_time = time.time()
            self.fps_times.append(current_time)
            self.frames_processed += 1
            
            # Calculate FPS
            if len(self.fps_times) > 1:
                time_diff = self.fps_times[-1] - self.fps_times[0]
                if time_diff > 0:
                    self.current_fps = len(self.fps_times) / time_diff
            
            # Inference time (display only, in ms)
            if hasattr(self, 'frame_start_time'):
                self.inference_time = (current_time - self.frame_start_time) * 1000
            
            # Add to history
            self.fps_history.append(self.current_fps)
            self.cpu_history.append(self.cpu_percent)
            self.gpu_history.append(self.gpu_percent)
            self.inference_history.append(self.inference_time)
    
    def mark_frame_dropped(self):
        """Call when a frame is skipped/dropped"""
        with self.lock:
            self.frames_dropped += 1
    
    def update_system_metrics(self):
        """Update CPU/GPU utilization (call periodically, not every frame)"""
        try:
            with self.lock:
                self.cpu_percent = psutil.cpu_percent(interval=0.05)
                
                # Try to get GPU metrics (NVIDIA only)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.gpu_percent = (mem_info.used / mem_info.total) * 100
                    pynvml.nvmlShutdown()
                except:
                    self.gpu_percent = 0.0
        except:
            pass
    
    def get_metrics_string(self):
        """Return formatted metrics string for display"""
        with self.lock:
            return {
                'fps': f"{self.current_fps:.1f}",
                'cpu': f"{self.cpu_percent:.1f}%",
                'gpu': f"{self.gpu_percent:.1f}%",
                'inference_ms': f"{self.inference_time:.1f}ms",
                'frames_dropped': self.frames_dropped
            }
    
    def get_metrics(self):
        """Return raw metrics dict for CSV logging"""
        with self.lock:
            return {
                'fps': self.current_fps,
                'cpu_percent': self.cpu_percent,
                'gpu_percent': self.gpu_percent,
                'inference_time_ms': self.inference_time
            }
    
    def get_histories(self):
        """Return history data for graphing"""
        with self.lock:
            return {
                'fps': list(self.fps_history),
                'cpu': list(self.cpu_history),
                'gpu': list(self.gpu_history),
                'inference': list(self.inference_history)
            }
    
    def reset_dropped_frames(self):
        """Reset dropped frame counter"""
        with self.lock:
            self.frames_dropped = 0
