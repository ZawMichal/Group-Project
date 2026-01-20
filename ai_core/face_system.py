import cv2
from .face_encoder import FaceEncoder
from .config import config

class FaceSystem(FaceEncoder):
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self._log("Initializing face recognition system...")
        super().__init__(log_callback=log_callback)
        # Automatycznie ładujemy twarze przy starcie
        self._log("Loading known faces database...")
        self.load_known_faces()
        self._log(f"Loaded {len(self.known_names)} people from database")
    
    def _log(self, msg):
        """Log to both terminal and GUI"""
        print(f"[FACE] {msg}")
        if self.log_callback:
            self.log_callback(f"[FACE] {msg}")

    def draw_results(self, frame, results):
        """
        Rysuje wyniki rozpoznawania twarzy na klatce.
        """
        for result in results:
            box = result['box']
            name = result['name']
            confidence = result['confidence']
            labels = result.get('labels')
            is_blacklisted = result.get('is_blacklisted', False)
            
            x1, y1, x2, y2 = box
            
            # Kolory: czerwony zarezerwowany dla blacklisty
            if config.show_blacklist_alerts and is_blacklisted:
                color = (0, 0, 255)  # Red
            elif name == "Unknown":
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green
            
            # Rysowanie ramki
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Etykiety (1-3 linie) — font jak w YOLO
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            if labels:
                label_lines = [f"{l['name']} ({l['confidence']:.2f})" for l in labels]
            else:
                label_lines = [f"{name} ({confidence:.2f})"]

            if config.show_blacklist_alerts and is_blacklisted:
                label_lines = [f"! {line}" for line in label_lines]

            # Oblicz szerokość najdłuższej linii
            max_w = 0
            line_h = 0
            for line in label_lines:
                (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
                max_w = max(max_w, w)
                line_h = max(line_h, h)

            # Tło pod napisy
            total_h = len(label_lines) * (line_h + 4) + 4
            cv2.rectangle(frame, (x1, y1 - total_h), (x1 + max_w + 6, y1), color, -1)

            # Rysuj linie tekstu
            y_text = y1 - 6
            for line in label_lines:
                cv2.putText(frame, line, (x1 + 3, y_text),
                           font, font_scale, (255, 255, 255), thickness)
                y_text -= (line_h + 4)
            
        return frame