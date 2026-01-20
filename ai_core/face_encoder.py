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
        
        # Apply quantization based on config
        self._apply_quantization()
        
        self._log("InceptionResnetV1 loaded")
        
        self.known_encodings = {}
        self.known_names = []
        
        # Upewnij się, że folder istnieje
        config.ensure_dirs()

    def _match_model_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cast tensor to the dtype of the model weights if available."""
        target_dtype = None
        try:
            target_dtype = next(self.resnet.parameters()).dtype
        except Exception:
            target_dtype = getattr(self.resnet, "dtype", None)
        if target_dtype is not None and tensor.dtype != target_dtype:
            tensor = tensor.to(target_dtype)
        return tensor
    
    def _ensure_model_device(self, tensor: torch.Tensor):
        """Ensure model and tensor are on compatible device.
        (INT8 support removed due to GPU incompatibility)"""
        return tensor.to(config.device)
    
    def _log(self, msg):
        """Log to both terminal and GUI"""
        print(f"[FACE] {msg}")
        if self.log_callback:
            self.log_callback(f"[FACE] {msg}")
    
    def _apply_quantization(self):
        """Apply quantization to the model based on config settings"""
        try:
            if config.device == 'cpu':
                # CPU: force FP32 for stability and speed
                self._log("CPU detected: forcing FP32 for face recognition")
                return
            if config.quantization_mode == 'fp16':
                self._log(f"Applying FP16 quantization...")
                self.resnet = self.resnet.half()
                self._log("FP16 quantization applied (2× speedup expected)")
            elif config.quantization_mode == 'int8':
                # INT8 dynamic quantization causes issues with face recognition model
                # For now, fall back to FP32 for face encoder
                self._log(f"INT8 not supported for face recognition (GPU issues). Using FP32 instead.")
                # Keep as FP32, no conversion
            else:
                self._log(f"Using FP32 (no quantization)")
        except Exception as e:
            self._log(f"Quantization failed: {e}. Using FP32.")
            self.quantization_mode = 'fp32'
    
    def set_quantization_mode(self, mode):
        """Change quantization mode (fp32, fp16, int8)"""
        if mode not in ['fp32', 'fp16', 'int8']:
            self._log(f"Invalid quantization mode: {mode}")
            return False
        
        if config.quantization_mode == mode:
            self._log(f"Already using {mode}")
            return True
        
        self._log(f"Switching to {mode} quantization...")
        config.quantization_mode = mode
        
        try:
            # Reload model with new quantization
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(config.device)
            self._apply_quantization()
            self._log(f"Successfully switched to {mode}")
            return True
        except Exception as e:
            self._log(f"Failed to switch to {mode}: {e}")
            return False

    
    def encode_face(self, image_path):
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Normalize image input to avoid dtype inference issues
            img = Image.fromarray(np.asarray(img, dtype=np.uint8))

            boxes, probs = self.mtcnn.detect(img)

            if boxes is None or len(boxes) == 0:
                return None, None, 'no_face'

            faces, face_probs = self.mtcnn(img, return_prob=True)
            if faces is None:
                return None, None, 'no_face'

            if isinstance(faces, torch.Tensor):
                if faces.dim() == 3:
                    face = faces.unsqueeze(0)
                else:
                    face = faces[0:1]
            else:
                # list/array fallback
                try:
                    face = torch.stack(faces)[0:1]
                except Exception:
                    return None, None, 'no_face'
            
            face = face.to(config.device)
            face = self._ensure_model_device(face)
            face = self._match_model_dtype(face)
            
            with torch.no_grad():
                encoding = self.resnet(face).cpu().numpy()[0]

            conf = None
            if face_probs is not None and len(face_probs) > 0:
                conf = float(face_probs[0])

            return encoding, conf, None
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

            # Match dtype to model (handle FP16 quantized model)
            faces_tensor = self._ensure_model_device(faces_tensor)
            faces_tensor = self._match_model_dtype(faces_tensor)

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

                    top_k = max(1, min(3, int(getattr(config, 'face_top_k', 1))))
                    top_matches = matches[:top_k]
                    labels = [{'name': m['name'], 'confidence': m['confidence']} for m in top_matches]
                    
                    if top_match['confidence'] >= config.confidence_threshold:
                        name = top_match['name']
                        final_conf = top_match['confidence']
                    else:
                        name = "Unknown"
                        final_conf = top_match['confidence'] # Zwracamy conf nawet jak niski
                        # Dodaj Unknown jako pierwszy label przy niskiej pewności
                        unknown_conf = max(0.0, 1.0 - top_match['confidence'])
                        labels = [{'name': 'Unknown', 'confidence': unknown_conf}] + labels[:max(0, top_k - 1)]
                else:
                    name = "Unknown"
                    final_conf = 0.0
                    labels = [{'name': 'Unknown', 'confidence': 1.0}]
                
                results.append({
                    'box': box.astype(int),
                    'name': name,
                    'confidence': final_conf,
                    'labels': labels
                })
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Recognition failed: {e}")
            return []