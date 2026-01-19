import torch
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from .config import config 

class FaceEncoder:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self._log(f"Initializing face detection models on {config.device}")
        
        self._log("Loading MTCNN detector...")
        self.mtcnn = MTCNN(
            image_size=config.image_size,
            margin=config.margin,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=config.device,
            keep_all=True
        )
        self._log("MTCNN detector loaded")
        
        self._log("Loading InceptionResnetV1...")
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(config.device)
        self._log("InceptionResnetV1 loaded")
        
        self.known_encodings = {}
        self.known_names = []
        
        # Upewnij się, że folder istnieje
        config.ensure_dirs()
    
    def _log(self, msg):
        """Log to both terminal and GUI"""
        print(f"[FACE] {msg}")
        if self.log_callback:
            self.log_callback(f"[FACE] {msg}")
    
    def encode_face(self, image_path):
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            boxes, probs = self.mtcnn.detect(img)

            if boxes is None or len(boxes) == 0:
                return None, None, 'no_face'

            faces = self.mtcnn(img)
            if faces is None:
                return None, None, 'no_face'
            
            if isinstance(faces, list):
                face = faces[0]
            else:
                if faces.dim() == 3:
                    face = faces.unsqueeze(0)
                else:
                    face = faces[0:1]
            
            face = face.to(config.device)
            
            with torch.no_grad():
                encoding = self.resnet(face).cpu().numpy()[0]
            
            return encoding, float(probs[0]), None
        except Exception as e:
            return None, None, str(e)
    
    def load_known_faces(self):
        known_dir = config.known_faces_dir
        if not known_dir.exists():
            print(f"[WARNING] Directory not found: {known_dir}")
            return
        
        print(f"[LOADING] Scanning {known_dir}")
        self.known_encodings = {}
        self.known_names = []
        
        total_images = 0
        loaded_images = 0

        # Iteracja po folderach osób
        for person_path in sorted(known_dir.iterdir()):
            if not person_path.is_dir():
                continue
            
            person_name = person_path.name
            encodings = []
            
            # Pobierz pliki zdjęć
            files = [f for f in person_path.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
            
            for file_path in files:
                total_images += 1
                encoding, prob, err = self.encode_face(str(file_path))

                if encoding is not None:
                    encodings.append(encoding)
                    loaded_images += 1
                    # print(f"[LOADED] {person_name}/{file_path.name}") # Opcjonalne logowanie
                else:
                    print(f"[SKIP] {person_name}/{file_path.name}: {err}")

            if encodings:
                self.known_encodings[person_name] = np.array(encodings)
                if person_name not in self.known_names:
                    self.known_names.append(person_name)

        print(f"[COMPLETE] Loaded {loaded_images} images for {len(self.known_names)} people.")
    
    def recognize_faces(self, frame_rgb):
        try:
            # Konwersja numpy array (RGB) na PIL Image
            img_pil = Image.fromarray(frame_rgb)

            boxes, probs = self.mtcnn.detect(img_pil)
            if boxes is None:
                return []

            faces, face_probs = self.mtcnn(img_pil, return_prob=True)
            if faces is None:
                return []

            # Normalizacja tensorów twarzy
            if isinstance(faces, torch.Tensor):
                if faces.dim() == 3:
                    faces_tensor = faces.unsqueeze(0).to(config.device)
                else:
                    faces_tensor = faces.to(config.device)
            else:
                # Fallback dla starszych wersji
                try:
                    faces_tensor = torch.stack(faces).to(config.device)
                except:
                    return []

            encodings = []
            with torch.no_grad():
                encodings = self.resnet(faces_tensor).cpu().numpy()
            
            results = []
            
            # Dopasowywanie wykrytych twarzy do bazy
            for i, box in enumerate(boxes):
                prob = float(probs[i]) if probs is not None else 1.0
                if len(encodings) <= i or prob < config.detection_threshold:
                    continue

                encoding = encodings[i]
                
                matches = []
                if self.known_encodings:
                    for person_name, known_encs in self.known_encodings.items():
                        distances = np.linalg.norm(known_encs - encoding, axis=1)
                        min_dist = np.min(distances)
                        conf = max(0, 1 - (min_dist / 1.2)) # Trochę luźniejsza skala
                        
                        matches.append({
                            'name': person_name,
                            'confidence': conf,
                            'distance': min_dist
                        })
                    
                    matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)
                    top_match = matches[0]
                    
                    if top_match['confidence'] >= config.confidence_threshold:
                        name = top_match['name']
                        final_conf = top_match['confidence']
                    else:
                        name = "Unknown"
                        final_conf = top_match['confidence'] # Zwracamy conf nawet jak niski
                else:
                    name = "Unknown"
                    final_conf = 0.0
                
                results.append({
                    'box': box.astype(int),
                    'name': name,
                    'confidence': final_conf
                })
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Recognition failed: {e}")
            return []