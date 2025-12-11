import cv2
from .face_encoder import FaceEncoder

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
            
            x1, y1, x2, y2 = box
            
            # Zielony dla znanych, Czerwony dla nieznanych
            if name == "Unknown":
                color = (0, 0, 255) # BGR: Red
            else:
                color = (0, 255, 0) # BGR: Green
            
            # Rysowanie ramki
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Etykieta
            label = f"{name} ({confidence:.2f})"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Tło pod napis
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        return frame