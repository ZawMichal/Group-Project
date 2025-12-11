# Group Project - Developer Documentation

## Overview

AI-powered computer vision system combining YOLO object detection and FaceNet face recognition. The backend (`ai_core/`) is framework-agnostic and can be integrated with any GUI.

**Current GUI (`gui_app.py`) is temporary.** Feel free to delete it and build your own interface using the backend API.

---

## Project Structure

```
groip-project/
├── ai_core/               # Backend AI engine - DO NOT DELETE
│   ├── config.py          # Global configuration & device selection
│   ├── engine.py          # Main orchestrator - single entry point
│   ├── object_system.py   # YOLO object detection wrapper
│   ├── face_system.py     # Face recognition system
│   ├── face_encoder.py    # Face encoding & matching logic
│   └── manager.py         # Person database file operations
├── known_faces/           # Face database (folders per person)
│   └── PersonName/
│       ├── PersonName_1.jpg
│       └── PersonName_2.jpg
├── yolo11n.pt            # YOLO models (nano/small/medium/large/xlarge)
├── yolo11s.pt
├── yolo11m.pt
├── yolo11x.pt
├── gui_app.py            # DELETABLE - Current Tkinter GUI
└── requirements.txt      # Python dependencies
```

---

## Quick Start

```python
from ai_core.engine import AIEngine
import cv2

# Initialize engine
engine = AIEngine()
engine.enable_faces = True
engine.enable_yolo = True

# Process camera frames
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        processed = engine.process_frame(frame)
        cv2.imshow('Video', processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

---

## Backend API Documentation

### File: `config.py`

**Purpose:** Global configuration singleton. Manages device selection (CPU/CUDA/MPS), file paths, and recognition thresholds.

#### Class: `Config`

**Usage:**
```python
from ai_core.config import config  # Singleton instance

# Access properties
print(config.device)                 # 'cpu', 'cuda', or 'mps'
print(config.known_faces_dir)        # Path to face database
config.confidence_threshold = 0.7    # Change face matching threshold
```

**Properties:**
- `device` (str): Device used for inference ('cpu', 'cuda', 'mps')
- `project_root` (Path): Project root directory
- `known_faces_dir` (Path): Where face images are stored (`known_faces/`)
- `confidence_threshold` (float): Minimum confidence for face match (default: 0.6)
- `detection_threshold` (float): Minimum confidence for face detection (default: 0.9)
- `image_size` (int): Face crop size for encoding (default: 160)
- `margin` (int): Margin around detected face (default: 20)

**Methods:**

##### `__init__(log_callback=None)`
Initializes config and detects available hardware acceleration.
- **Parameters:**
  - `log_callback` (callable, optional): Function to receive log messages
- **Notes:** Automatically called when importing config singleton

##### `_get_device()`
Detects GPU availability (CUDA > MPS > CPU).
- **Returns:** str - 'cuda', 'mps', or 'cpu'
- **Internal use only**

##### `ensure_dirs()`
Creates `known_faces/` directory if it doesn't exist.
- **Returns:** None
- **Call before any face operations**

**Modifying Configuration:**

Change face matching strictness:
```python
config.confidence_threshold = 0.75  # Stricter (fewer false positives)
config.confidence_threshold = 0.4   # Lenient (more matches)
```

Change face detection quality:
```python
config.detection_threshold = 0.95  # Only high-quality faces
config.image_size = 224            # Higher resolution encoding (slower)
```

---

### File: `engine.py`

**Purpose:** Main orchestrator for all AI systems. Single entry point for frame processing. Handles lazy loading of models.

#### Class: `AIEngine`

**Usage:**
```python
engine = AIEngine(log_callback=print)
engine.enable_yolo = True
engine.enable_faces = True
processed_frame = engine.process_frame(raw_frame)
```

**Properties:**
- `manager` (PersonManager): Access to person database operations
- `object_system` (ObjectSystem): YOLO detection system (lazy-loaded)
- `face_system` (FaceSystem): Face recognition system (lazy-loaded)
- `enable_yolo` (bool): Toggle YOLO detection (default: False)
- `enable_faces` (bool): Toggle face recognition (default: False)

**Methods:**

##### `__init__(log_callback=None)`
Initializes engine without loading models.
```python
engine = AIEngine(log_callback=lambda msg: print(msg))
```
- **Parameters:**
  - `log_callback` (callable, optional): Function to receive status messages
- **Notes:** Models load on first access to save startup time

##### `process_frame(frame)`
Main processing function. Call this for every frame.
```python
annotated_frame = engine.process_frame(raw_frame)
```
- **Parameters:**
  - `frame` (numpy.ndarray): BGR image from OpenCV
- **Returns:** numpy.ndarray - Annotated frame with bounding boxes/labels
- **Notes:**
  - Processes YOLO if `enable_yolo=True`
  - Processes faces if `enable_faces=True`
  - Returns copy, original unchanged
  - Skip processing by setting both flags to False

##### `reload_faces()`
Reloads face database from disk.
```python
engine.reload_faces()
```
- **Returns:** None
- **When to call:** After adding/deleting people, after capturing training photos
- **Notes:** Rescans `known_faces/` and re-encodes all faces

##### `change_yolo_model(model_name)`
Switches YOLO model variant.
```python
success = engine.change_yolo_model("yolo11s")
```
- **Parameters:**
  - `model_name` (str): Model variant without .pt: "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"
- **Returns:** bool - True if successful
- **Notes:** Takes 5-30 seconds depending on model size

**Model Selection:**
- `yolo11n` - Fastest, smallest (6MB), good accuracy
- `yolo11s` - Fast, balanced (21MB)
- `yolo11m` - Medium speed/accuracy (49MB)
- `yolo11l` - Slow, high accuracy (90MB)
- `yolo11x` - Slowest, best accuracy (141MB)

---

### File: `object_system.py`

**Purpose:** Wrapper for Ultralytics YOLO. Handles object detection and tracking.

#### Class: `ObjectSystem`

**Usage:**
```python
yolo = ObjectSystem(model_variant="yolo11n.pt", enable_tracking=True)
results = yolo.process_frame(frame)
annotated = yolo.draw_results(frame, results)
```

**Properties:**
- `model` (YOLO): Loaded YOLO model instance
- `enable_tracking` (bool): Enable object tracking across frames (default: True)
- `conf` (float): Confidence threshold for detections 0.0-1.0 (default: 0.5)
- `show_conf` (bool): Display confidence scores (default: True)
- `show_labels` (bool): Display class labels (default: True)
- `show_fps` (bool): Display FPS counter (default: True)

**Methods:**

##### `__init__(model_variant="yolo11n.pt", enable_tracking=True, log_callback=None)`
Loads YOLO model immediately.
```python
yolo = ObjectSystem("yolo11n.pt", enable_tracking=True, log_callback=print)
```
- **Parameters:**
  - `model_variant` (str): Path to .pt file
  - `enable_tracking` (bool): Enable multi-object tracking
  - `log_callback` (callable, optional): Logging function
- **Notes:** Takes 5-30 seconds to load

##### `process_frame(frame)`
Detects objects in frame.
```python
results = yolo.process_frame(frame)
```
- **Parameters:**
  - `frame` (numpy.ndarray): BGR image
- **Returns:** ultralytics Results object
- **Notes:**
  - Uses `model.track()` if tracking enabled
  - Uses `model.predict()` otherwise
  - Does NOT draw on frame

##### `draw_results(frame, results)`
Draws bounding boxes and labels on frame.
```python
annotated = yolo.draw_results(frame, results)
```
- **Parameters:**
  - `frame` (numpy.ndarray): Image to draw on
  - `results` (Results): Output from `process_frame()`
- **Returns:** numpy.ndarray - Annotated frame
- **Notes:** Modifies frame in-place and returns it

##### `change_model(new_variant)`
Replaces current model.
```python
success = yolo.change_model("yolo11m.pt")
```
- **Parameters:**
  - `new_variant` (str): Path to new .pt file
- **Returns:** bool - True on success

**Modifying Behavior:**

Change detection confidence:
```python
yolo.conf = 0.7  # Only show detections >70% confidence
```

Toggle display options:
```python
yolo.show_fps = False      # Hide FPS counter
yolo.show_conf = False     # Hide confidence scores
yolo.show_labels = False   # Hide class labels
```

Disable tracking for speed:
```python
yolo.enable_tracking = False  # Faster, no object IDs
```

---

### File: `face_encoder.py`

**Purpose:** Low-level face detection and encoding using MTCNN + InceptionResnetV1 (FaceNet). Handles face matching logic.

#### Class: `FaceEncoder`

**Usage:**
```python
encoder = FaceEncoder(log_callback=print)
encoding, prob, error = encoder.encode_face("/path/to/face.jpg")
results = encoder.recognize_faces(rgb_frame)
```

**Properties:**
- `mtcnn` (MTCNN): Face detector from facenet_pytorch
- `resnet` (InceptionResnetV1): Face encoder (512-dim vectors)
- `known_encodings` (dict): Stored face vectors `{"PersonName": ndarray(N,512), ...}`
- `known_names` (list): List of person names in database

**Methods:**

##### `__init__(log_callback=None)`
Loads MTCNN and InceptionResnetV1 models.
```python
encoder = FaceEncoder(log_callback=print)
```
- **Parameters:**
  - `log_callback` (callable, optional): Logging function
- **Notes:**
  - Takes 10-30 seconds on first run
  - Downloads pretrained weights if needed
  - Models placed on GPU if available

##### `encode_face(image_path)`
Encodes a single face image to 512-dim vector.
```python
encoding, probability, error = encoder.encode_face("/path/to/image.jpg")
```
- **Parameters:**
  - `image_path` (str): Path to image file
- **Returns:** tuple of:
  - `encoding` (numpy.ndarray or None): 512-dim face vector
  - `probability` (float or None): Detection confidence 0-1
  - `error` (str or None): Error message ('no_face', etc.)
- **Notes:**
  - Returns `(None, None, 'no_face')` if no face found
  - Returns `(None, None, error_msg)` on other errors

##### `load_known_faces()`
Scans face database and encodes all faces.
```python
encoder.load_known_faces()
```
- **Returns:** None
- **Effect:**
  - Scans `known_faces/` directory
  - Encodes all JPG/PNG images
  - Populates `known_encodings` and `known_names`
- **Notes:**
  - Call after modifying database
  - Prints progress to console
  - Skips corrupt images

##### `recognize_faces(frame_rgb)`
Detects and identifies all faces in frame.
```python
results = encoder.recognize_faces(rgb_frame)
```
- **Parameters:**
  - `frame_rgb` (numpy.ndarray): RGB image (NOT BGR)
- **Returns:** list of dicts:
  ```python
  [
      {
          'box': [x1, y1, x2, y2],  # int coordinates
          'name': 'PersonName',      # or 'Unknown'
          'confidence': 0.85         # float 0-1
      },
      ...
  ]
  ```
- **Notes:**
  - Detects ALL faces in frame
  - Matches against known database
  - Returns 'Unknown' if confidence < threshold
  - Empty list if no faces detected

**Face Recognition Pipeline:**

1. **Detection:** MTCNN finds face bounding boxes in image
2. **Alignment:** MTCNN crops and aligns faces to 160x160
3. **Encoding:** InceptionResnetV1 converts face to 512-dim vector
4. **Matching:** Computes Euclidean distance to all known faces
5. **Classification:** Closest match above threshold = person identified

**Modifying Recognition Logic:**

Change distance metric (line ~152):
```python
# Current: Euclidean distance
distances = np.linalg.norm(known_encs - encoding, axis=1)

# Alternative: Cosine similarity
from scipy.spatial.distance import cosine
distances = [cosine(known_enc, encoding) for known_enc in known_encs]
```

Change confidence calculation (line ~153):
```python
# Current formula
conf = max(0, 1 - (min_dist / 1.2))

# Alternative: More lenient
conf = max(0, 1 - (min_dist / 1.5))

# Alternative: Stricter
conf = max(0, 1 - (min_dist / 1.0))
```

---

### File: `face_system.py`

**Purpose:** High-level face recognition wrapper with drawing capabilities. Inherits from `FaceEncoder`.

#### Class: `FaceSystem`

**Usage:**
```python
faces = FaceSystem(log_callback=print)
results = faces.recognize_faces(rgb_frame)
annotated = faces.draw_results(bgr_frame, results)
```

**Inherits:** All methods and properties from `FaceEncoder`

**Methods:**

##### `__init__(log_callback=None)`
Initializes and loads face database automatically.
```python
faces = FaceSystem(log_callback=print)
```
- **Parameters:**
  - `log_callback` (callable, optional): Logging function
- **Notes:**
  - Calls `super().__init__()` to load models
  - Automatically calls `load_known_faces()`
  - Ready to use immediately

##### `draw_results(frame, results)`
Draws bounding boxes and labels on frame.
```python
annotated = faces.draw_results(bgr_frame, recognition_results)
```
- **Parameters:**
  - `frame` (numpy.ndarray): BGR image to draw on
  - `results` (list): Output from `recognize_faces()`
- **Returns:** numpy.ndarray - Annotated frame
- **Notes:**
  - Green boxes for known people
  - Red boxes for unknown people
  - Modifies frame in-place

**Modifying Drawing:**

Change box colors (line ~27-32):
```python
if name == "Unknown":
    color = (0, 0, 255)    # BGR: Red
else:
    color = (0, 255, 0)    # BGR: Green

# Customize:
color = (255, 0, 0)        # Blue
color = (0, 255, 255)      # Yellow
color = (128, 128, 128)    # Gray
```

Change label format (line ~41):
```python
label = f"{name} ({confidence:.2f})"

# Alternatives:
label = f"{name} - {int(confidence*100)}%"     # Percentage
label = f"{name}"                               # Name only
label = f"{name}\n{confidence:.3f}"            # Multi-line
```

Change box thickness (line ~38):
```python
cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 2 = thickness

# Thicker box:
cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

# Filled box:
cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
```

---

### File: `manager.py`

**Purpose:** File system operations for face database. Manages person folders and training photos.

#### Class: `PersonManager`

**Usage:**
```python
manager = PersonManager()
people = manager.get_people_list()
path = manager.save_training_photo("Alice", frame)
```

**Properties:**
- `known_faces_dir` (Path): Path to `known_faces/` directory

**Methods:**

##### `__init__()`
Initializes manager and ensures directory exists.
```python
manager = PersonManager()
```
- **Effect:** Creates `known_faces/` if missing

##### `get_people_list()`
Returns list of all people in database.
```python
people = manager.get_people_list()  # ['Alice', 'Bob', 'Charlie']
```
- **Returns:** list of str - Person names (folder names)
- **Notes:** Empty list if directory doesn't exist

##### `create_person_folder(name)`
Creates folder for new person.
```python
folder_path = manager.create_person_folder("John Doe")
```
- **Parameters:**
  - `name` (str): Person name (spaces allowed)
- **Returns:** Path object or None if invalid name
- **Effect:** Creates `known_faces/John Doe/` directory
- **Notes:**
  - Sanitizes name (removes invalid characters)
  - Safe to call multiple times (idempotent)

##### `save_training_photo(name, frame)`
Saves frame as training photo.
```python
saved_path = manager.save_training_photo("Alice", camera_frame)
```
- **Parameters:**
  - `name` (str): Person name
  - `frame` (numpy.ndarray): BGR image from camera
- **Returns:** str (file path) or None on failure
- **Effect:** Saves frame as JPEG in person's folder
- **File naming:** Auto-increments: `Alice_1.jpg`, `Alice_2.jpg`, etc.
- **Notes:** Creates person folder if it doesn't exist

##### `rename_person(old_name, new_name)`
Renames person folder.
```python
success = manager.rename_person("Bob", "Robert")
```
- **Parameters:**
  - `old_name` (str): Current name
  - `new_name` (str): New name
- **Returns:** bool - True on success
- **Notes:** Fails if new name already exists

##### `delete_person(name)`
Deletes person and all their photos.
```python
success = manager.delete_person("Alice")
```
- **Parameters:**
  - `name` (str): Person name to delete
- **Returns:** bool - True on success
- **Effect:** Removes entire folder and contents
- **Notes:**
  - No undo available
  - Call `engine.reload_faces()` after deletion

**Photo Capture Strategies:**

Current GUI implementation (continuous capture):
```python
# Captures every ~0.5 seconds while recording
if int(time.time() * 10) % 5 == 0:
    path = manager.save_training_photo(name, frame)
```

Alternative: Single capture on button press:
```python
# When user clicks "Capture" button:
path = manager.save_training_photo("Alice", current_frame)
engine.reload_faces()  # Must reload after adding photos
```

Alternative: Capture N photos with delay:
```python
import time
for i in range(10):
    path = manager.save_training_photo("Alice", get_current_frame())
    time.sleep(0.5)  # 500ms between captures
engine.reload_faces()
```

Alternative: Capture only when face detected:
```python
results = face_system.recognize_faces(frame_rgb)
if len(results) > 0:  # Face detected
    path = manager.save_training_photo("Alice", frame)
```

---

## Complete Workflow Examples

### Training New Person

```python
from ai_core.engine import AIEngine
import cv2
import time

engine = AIEngine()
cap = cv2.VideoCapture(0)

# Capture 10 training photos
person_name = "Alice"
for i in range(10):
    ret, frame = cap.read()
    if ret:
        path = engine.manager.save_training_photo(person_name, frame)
        print(f"Saved {path}")
    time.sleep(0.5)

# Reload database
engine.reload_faces()
cap.release()
```

### Real-time Recognition

```python
from ai_core.engine import AIEngine
import cv2

engine = AIEngine()
engine.enable_faces = True

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        processed = engine.process_frame(frame)
        cv2.imshow('Recognition', processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### YOLO Detection Only

```python
from ai_core.engine import AIEngine
import cv2

engine = AIEngine()
engine.enable_yolo = True
engine.object_system.conf = 0.6

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        processed = engine.process_frame(frame)
        cv2.imshow('Detection', processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

---

## Camera Integration

### Basic Setup

```python
import cv2

cap = cv2.VideoCapture(0)  # 0 = first camera

# Optional: reduce latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process with AI
    processed = engine.process_frame(frame)
    
    # Display
    cv2.imshow('Video', processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### Multiple Camera Support

```python
# Scan for available cameras
available_cameras = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        available_cameras.append(i)
        cap.release()

print(f"Found cameras: {available_cameras}")
```

### Optimizing Frame Rate

```python
# Process every N frames (skip frames for speed)
frame_count = 0
while True:
    ret, frame = cap.read()
    frame_count += 1
    
    if frame_count % 2 == 0:  # Process every 2nd frame
        processed = engine.process_frame(frame)
        cv2.imshow('Video', processed)
```

---

## Configuration & Tuning

### Hardware Acceleration

```python
from ai_core.config import config
print(config.device)  # 'cpu', 'cuda', or 'mps'
```

### Recognition Tuning

Stricter matching (fewer false positives):
```python
from ai_core.config import config
config.confidence_threshold = 0.75
config.detection_threshold = 0.95
```

Lenient matching (more matches):
```python
config.confidence_threshold = 0.4
config.detection_threshold = 0.8
```

---

## Common Issues

**Face recognition not working:**
```python
# Check if database loaded
print(len(engine.face_system.known_names))  # Should be > 0

# Reload database
engine.reload_faces()

# Verify folder structure
# known_faces/Alice/Alice_1.jpg (correct)
# known_faces/Alice_1.jpg (wrong - must be in subfolder)
```

**Camera is slow:**
```python
# Use smaller YOLO model
engine.change_yolo_model("yolo11n")

# Reduce frame size
frame_small = cv2.resize(frame, (640, 480))
processed = engine.process_frame(frame_small)

# Process fewer frames
if frame_count % 3 == 0:  # Every 3rd frame
    processed = engine.process_frame(frame)
```

**Models not loading:**
```python
# Check model files exist
import os
print(os.path.exists("yolo11n.pt"))  # Should be True

# Check torch
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

---

## Building New GUI

**Minimal CLI Example:**
```python
from ai_core.engine import AIEngine
import cv2

def main():
    engine = AIEngine(log_callback=print)
    engine.enable_faces = True
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed = engine.process_frame(frame)
        cv2.imshow('App', processed)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            name = input("Enter name: ")
            engine.manager.save_training_photo(name, frame)
            engine.reload_faces()
    
    cap.release()

if __name__ == "__main__":
    main()
```

**Qt Structure:**
```python
from PyQt6.QtWidgets import QApplication, QMainWindow
from ai_core.engine import AIEngine

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = AIEngine(log_callback=self.log)
        # Setup UI, connect signals, etc.
    
    def log(self, message):
        self.log_widget.append(message)
```

**Flask Web API:**
```python
from flask import Flask, Response
from ai_core.engine import AIEngine
import cv2

app = Flask(__name__)
engine = AIEngine()
engine.enable_faces = True

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = engine.process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')
```

---

## Dependencies

```
opencv-python       # Camera and image processing
ultralytics         # YOLO object detection
torch              # PyTorch deep learning
torchvision        # Vision models
facenet-pytorch    # Face recognition
Pillow             # Image loading
numpy              # Numerical arrays
```

Install: `pip install -r requirements.txt`


```
groip-project/
├── ai_core/               # Backend AI engine (PRESERVE THIS)
│   ├── __init__.py
│   ├── config.py          # Global configuration
│   ├── engine.py          # Main AI engine orchestrator
│   ├── object_system.py   # YOLO object detection
│   ├── face_system.py     # Face recognition system
│   ├── face_encoder.py    # Face encoding/matching logic
│   └── manager.py         # Person database management
├── known_faces/           # Face database (folders per person)
│   └── PersonName/        # Each person gets a folder
│       ├── PersonName_1.jpg
│       ├── PersonName_2.jpg
│       └── ...
├── yolo11n.pt            # YOLO models (n=nano, s=small, m=medium, etc.)
├── yolo11s.pt
├── yolo11m.pt
├── yolo11x.pt
├── gui_app.py            # ⚠️ DELETABLE - Current Tkinter GUI
└── requirements.txt      # Python dependencies
```

---

## 🔧 Backend API Reference

### 1. `config.py` - Global Configuration

**Purpose:** Manages device selection (CPU/GPU), paths, and recognition thresholds.

#### Class: `Config`

```python
from ai_core.config import config  # Singleton instance

# Properties:
config.device                  # str: 'cpu', 'cuda', or 'mps'
config.project_root           # Path: Project root directory
config.known_faces_dir        # Path: Where face images are stored
config.confidence_threshold   # float: 0.6 - Min confidence for face match
config.detection_threshold    # float: 0.9 - Min confidence for face detection
config.image_size            # int: 160 - Face crop size for encoding
config.margin                # int: 20 - Margin around detected face
```

#### Methods:

##### `Config.__init__(log_callback=None)`
- **Parameters:**
  - `log_callback` (callable, optional): Function to call with log messages
- **Returns:** Config instance
- **Notes:** Automatically detects GPU availability on initialization

##### `Config._get_device()`
- **Returns:** str ('cpu', 'cuda', or 'mps')
- **Notes:** Checks torch.cuda and torch.backends.mps availability
- **Internal use only**

##### `Config.ensure_dirs()`
- **Returns:** None
- **Effect:** Creates `known_faces/` directory if it doesn't exist
- **Notes:** Call before any face operations

**How to Modify:**
- Change `confidence_threshold` to make face matching more/less strict
- Change `detection_threshold` to filter out low-quality face detections
- Modify `image_size` to change face encoding resolution (higher = more accurate, slower)

---

### 2. `engine.py` - Main AI Engine

**Purpose:** Orchestrates all AI systems (YOLO + Face Recognition). Single entry point for processing frames.

#### Class: `AIEngine`

```python
from ai_core.engine import AIEngine

engine = AIEngine(log_callback=my_log_function)
```

#### Properties:

##### `AIEngine.manager`
- **Type:** `PersonManager`
- **Purpose:** Access to person database operations
- **Usage:** `engine.manager.get_people_list()`

##### `AIEngine.object_system` (lazy-loaded property)
- **Type:** `ObjectSystem`
- **Purpose:** YOLO object detection system
- **Notes:** Only loads when first accessed (saves startup time)
- **Usage:** Automatically accessed when `enable_yolo=True`

##### `AIEngine.face_system` (lazy-loaded property)
- **Type:** `FaceSystem`
- **Purpose:** Face recognition system
- **Notes:** Only loads when first accessed
- **Usage:** Automatically accessed when `enable_faces=True`

##### `AIEngine.enable_yolo`
- **Type:** bool
- **Default:** False
- **Purpose:** Toggle YOLO detection on/off

##### `AIEngine.enable_faces`
- **Type:** bool
- **Default:** False
- **Purpose:** Toggle face recognition on/off

#### Methods:

##### `AIEngine.__init__(log_callback=None)`
```python
engine = AIEngine(log_callback=print)
```
- **Parameters:**
  - `log_callback` (callable, optional): Function to receive log messages like `lambda msg: print(msg)`
- **Returns:** AIEngine instance
- **Notes:** Does NOT load models immediately (lazy loading). Sets up manager only.

##### `AIEngine.process_frame(frame)`
```python
annotated_frame = engine.process_frame(raw_frame)
```
- **Parameters:**
  - `frame` (numpy.ndarray): BGR image from OpenCV (camera frame)
- **Returns:** numpy.ndarray - Annotated frame with bounding boxes and labels
- **Notes:** 
  - Processes YOLO if `enable_yolo=True`
  - Processes faces if `enable_faces=True`
  - Returns copy of frame, original is unchanged
  - This is the **main processing function** - call this for every frame

##### `AIEngine.reload_faces()`
```python
engine.reload_faces()
```
- **Parameters:** None
- **Returns:** None
- **Purpose:** Reloads face database from disk
- **When to call:** After adding/deleting people, after capturing new training photos
- **Notes:** Scans `known_faces/` directory and re-encodes all faces

##### `AIEngine.change_yolo_model(model_name)`
```python
success = engine.change_yolo_model("yolo11s")
```
- **Parameters:**
  - `model_name` (str): Model variant without .pt extension: "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"
- **Returns:** bool - True if model loaded successfully
- **Notes:** 
  - Loads model from `.pt` file in project root
  - Models ranked by size/speed: nano < small < medium < large < xlarge
  - Takes 5-30 seconds depending on model size

**How to Use:**
```python
# Basic usage
engine = AIEngine()
engine.enable_yolo = True
engine.enable_faces = True

# Process camera frames
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        processed = engine.process_frame(frame)
        cv2.imshow('Video', processed)
```

---

### 3. `object_system.py` - YOLO Object Detection

**Purpose:** Wraps Ultralytics YOLO for object detection and tracking.

#### Class: `ObjectSystem`

```python
from ai_core.object_system import ObjectSystem

yolo = ObjectSystem(model_variant="yolo11n.pt", enable_tracking=True, log_callback=print)
```

#### Properties:

##### `ObjectSystem.model`
- **Type:** YOLO (ultralytics)
- **Purpose:** The loaded YOLO model instance

##### `ObjectSystem.enable_tracking`
- **Type:** bool
- **Default:** True
- **Purpose:** Enable object tracking across frames (assigns IDs to objects)

##### `ObjectSystem.conf`
- **Type:** float
- **Default:** 0.5
- **Purpose:** Confidence threshold for detections (0.0-1.0)

##### `ObjectSystem.show_conf`
- **Type:** bool
- **Default:** True
- **Purpose:** Display confidence scores on bounding boxes

##### `ObjectSystem.show_labels`
- **Type:** bool
- **Default:** True
- **Purpose:** Display class labels on bounding boxes

##### `ObjectSystem.show_fps`
- **Type:** bool
- **Default:** True
- **Purpose:** Display FPS counter on frame

#### Methods:

##### `ObjectSystem.__init__(model_variant="yolo11n.pt", enable_tracking=True, log_callback=None)`
- **Parameters:**
  - `model_variant` (str): Path to .pt model file
  - `enable_tracking` (bool): Enable multi-object tracking
  - `log_callback` (callable, optional): Logging function
- **Returns:** ObjectSystem instance
- **Notes:** Loads model immediately (takes 5-30s)

##### `ObjectSystem.process_frame(frame)`
```python
results = yolo.process_frame(frame)
```
- **Parameters:**
  - `frame` (numpy.ndarray): BGR image
- **Returns:** ultralytics Results object (from `results[0]`)
- **Notes:** 
  - Uses `model.track()` if `enable_tracking=True`
  - Uses `model.predict()` otherwise
  - Does NOT draw on frame, just returns detection data

##### `ObjectSystem.draw_results(frame, results)`
```python
annotated = yolo.draw_results(frame, results)
```
- **Parameters:**
  - `frame` (numpy.ndarray): BGR image to draw on
  - `results` (Results): Output from `process_frame()`
- **Returns:** numpy.ndarray - Frame with drawings
- **Notes:**
  - Draws bounding boxes, labels, confidence scores
  - Respects `show_conf`, `show_labels`, `show_fps` flags
  - Modifies frame in-place, but also returns it

##### `ObjectSystem.change_model(new_variant)`
```python
success = yolo.change_model("yolo11m.pt")
```
- **Parameters:**
  - `new_variant` (str): Path to new .pt model file
- **Returns:** bool - True on success
- **Notes:** Replaces current model, takes time to load

**How to Modify:**

**Change detection confidence:**
```python
yolo.conf = 0.7  # Only show detections >70% confidence
```

**Disable FPS counter:**
```python
yolo.show_fps = False
```

**Use tracking vs simple detection:**
```python
yolo.enable_tracking = False  # Faster, no object IDs
```

---

### 4. `face_encoder.py` - Face Encoding Engine

**Purpose:** Low-level face detection and encoding using MTCNN + InceptionResnetV1 (FaceNet).

#### Class: `FaceEncoder`

```python
from ai_core.face_encoder import FaceEncoder

encoder = FaceEncoder(log_callback=print)
```

#### Properties:

##### `FaceEncoder.mtcnn`
- **Type:** MTCNN (facenet_pytorch)
- **Purpose:** Face detector (finds faces in images)

##### `FaceEncoder.resnet`
- **Type:** InceptionResnetV1 (facenet_pytorch)
- **Purpose:** Face encoder (converts face to 512-dim vector)

##### `FaceEncoder.known_encodings`
- **Type:** dict
- **Format:** `{"PersonName": numpy.ndarray(shape=(N, 512)), ...}`
- **Purpose:** Stores face encodings for all known people

##### `FaceEncoder.known_names`
- **Type:** list
- **Purpose:** List of person names in database

#### Methods:

##### `FaceEncoder.__init__(log_callback=None)`
- **Parameters:**
  - `log_callback` (callable, optional): Logging function
- **Returns:** FaceEncoder instance
- **Notes:** 
  - Loads MTCNN and InceptionResnetV1 models
  - Takes 10-30 seconds on first run (downloads pretrained weights)
  - Models placed on GPU if available

##### `FaceEncoder.encode_face(image_path)`
```python
encoding, probability, error = encoder.encode_face("/path/to/face.jpg")
```
- **Parameters:**
  - `image_path` (str): Path to image file
- **Returns:** Tuple of:
  - `encoding` (numpy.ndarray or None): 512-dim face vector
  - `probability` (float or None): Detection confidence (0-1)
  - `error` (str or None): Error message if failed ('no_face', etc.)
- **Notes:**
  - Returns `(None, None, 'no_face')` if no face detected
  - Returns `(None, None, error_msg)` on other errors
  - Use probability to filter low-quality detections

##### `FaceEncoder.load_known_faces()`
```python
encoder.load_known_faces()
```
- **Parameters:** None
- **Returns:** None
- **Effect:** 
  - Scans `known_faces/` directory
  - Encodes all images in person folders
  - Populates `known_encodings` and `known_names`
- **Notes:**
  - Call after modifying face database
  - Prints progress to console
  - Skips corrupt/invalid images

##### `FaceEncoder.recognize_faces(frame_rgb)`
```python
results = encoder.recognize_faces(rgb_frame)
```
- **Parameters:**
  - `frame_rgb` (numpy.ndarray): RGB image (not BGR!)
- **Returns:** List of dicts:
  ```python
  [
      {
          'box': [x1, y1, x2, y2],  # int coords
          'name': 'PersonName',      # or 'Unknown'
          'confidence': 0.85         # float 0-1
      },
      ...
  ]
  ```
- **Notes:**
  - Detects ALL faces in frame
  - Matches each against known database
  - Returns 'Unknown' if confidence < threshold
  - Empty list if no faces detected

**How Face Recognition Works:**

1. **Detection:** MTCNN finds face bounding boxes
2. **Alignment:** MTCNN crops and aligns faces
3. **Encoding:** InceptionResnetV1 converts face to 512-dim vector
4. **Matching:** Computes Euclidean distance to all known faces
5. **Classification:** Closest match above threshold = recognized

**How to Modify:**

**Change recognition threshold:**
```python
from ai_core.config import config
config.confidence_threshold = 0.7  # Stricter matching
```

**Change detection threshold:**
```python
config.detection_threshold = 0.95  # Only high-quality detections
```

**Add custom distance metric:**
Edit `recognize_faces()` line ~152:
```python
# Current: Euclidean distance
distances = np.linalg.norm(known_encs - encoding, axis=1)

# Alternative: Cosine similarity
from scipy.spatial.distance import cosine
distances = [cosine(known_enc, encoding) for known_enc in known_encs]
```

---

### 5. `face_system.py` - Face Recognition System

**Purpose:** High-level wrapper around `FaceEncoder` with drawing capabilities.

#### Class: `FaceSystem` (inherits from `FaceEncoder`)

```python
from ai_core.face_system import FaceSystem

faces = FaceSystem(log_callback=print)
```

#### Methods:

##### `FaceSystem.__init__(log_callback=None)`
- **Parameters:**
  - `log_callback` (callable, optional): Logging function
- **Returns:** FaceSystem instance
- **Notes:** 
  - Calls `super().__init__()` to load models
  - Automatically calls `load_known_faces()`
  - Ready to use immediately after init

##### `FaceSystem.draw_results(frame, results)`
```python
annotated = faces.draw_results(bgr_frame, recognition_results)
```
- **Parameters:**
  - `frame` (numpy.ndarray): BGR image to draw on
  - `results` (list): Output from `recognize_faces()`
- **Returns:** numpy.ndarray - Frame with drawings
- **Notes:**
  - Draws **green boxes** for known people
  - Draws **red boxes** for unknown people
  - Draws name + confidence label above box
  - Modifies frame in-place

**Inherited Methods:**
- All methods from `FaceEncoder` are available
- `recognize_faces(frame_rgb)` - Detect and identify faces
- `load_known_faces()` - Reload database
- `encode_face(image_path)` - Encode single image

**How to Modify:**

**Change box colors:**
Edit `draw_results()` line ~27-32:
```python
if name == "Unknown":
    color = (0, 0, 255)    # BGR: Red
else:
    color = (0, 255, 0)    # BGR: Green

# Change to:
if name == "Unknown":
    color = (128, 128, 128)  # Gray for unknown
else:
    color = (255, 0, 0)      # Blue for known
```

**Change label format:**
Edit line ~41:
```python
label = f"{name} ({confidence:.2f})"

# Change to:
label = f"{name} - {int(confidence*100)}%"  # Percentage format
```

---

### 6. `manager.py` - Person Database Manager

**Purpose:** File system operations for managing the face database.

#### Class: `PersonManager`

```python
from ai_core.manager import PersonManager

manager = PersonManager()
```

#### Properties:

##### `PersonManager.known_faces_dir`
- **Type:** Path
- **Purpose:** Path to `known_faces/` directory

#### Methods:

##### `PersonManager.__init__()`
- **Parameters:** None
- **Returns:** PersonManager instance
- **Effect:** Creates `known_faces/` if it doesn't exist

##### `PersonManager.get_people_list()`
```python
people = manager.get_people_list()  # ['Alice', 'Bob', 'Charlie']
```
- **Parameters:** None
- **Returns:** list of str - Person names (folder names)
- **Notes:** Returns empty list if directory doesn't exist

##### `PersonManager.create_person_folder(name)`
```python
folder_path = manager.create_person_folder("John Doe")
```
- **Parameters:**
  - `name` (str): Person name (allows spaces, sanitized)
- **Returns:** Path object or None if invalid name
- **Effect:** Creates `known_faces/John Doe/` directory
- **Notes:** 
  - Sanitizes name (removes invalid characters)
  - mkdir with `exist_ok=True` (safe to call multiple times)

##### `PersonManager.save_training_photo(name, frame)`
```python
saved_path = manager.save_training_photo("Alice", camera_frame)
```
- **Parameters:**
  - `name` (str): Person name
  - `frame` (numpy.ndarray): BGR image from camera
- **Returns:** str (path) or None on failure
- **Effect:** Saves frame as JPEG in person's folder
- **Naming:** Automatically numbers files: `Alice_1.jpg`, `Alice_2.jpg`, etc.
- **Notes:**
  - Creates person folder if it doesn't exist
  - Increments counter automatically

**How Photo Capture Currently Works:**

In current GUI (`gui_app.py`), photos are captured **every frame while recording**:
```python
# Current implementation (line ~330):
if int(time.time() * 10) % 5 == 0:  # Every ~0.5 seconds
    path = engine.manager.save_training_photo(name, frame)
```

**How to Modify Photo Capture:**

**Capture on button press (single photo):**
```python
# When user clicks "Capture" button:
path = manager.save_training_photo("Alice", current_frame)
engine.reload_faces()  # Important: reload after adding photos
```

**Capture N photos with delay:**
```python
import time
for i in range(10):
    path = manager.save_training_photo("Alice", get_current_frame())
    time.sleep(0.5)  # 500ms between captures
engine.reload_faces()
```

**Capture only when face detected:**
```python
results = face_system.recognize_faces(frame_rgb)
if len(results) > 0:  # Face detected
    path = manager.save_training_photo("Alice", frame)
```

##### `PersonManager.rename_person(old_name, new_name)`
```python
success = manager.rename_person("Bob", "Robert")
```
- **Parameters:**
  - `old_name` (str): Current person name
  - `new_name` (str): New person name
- **Returns:** bool - True on success
- **Effect:** Renames folder in filesystem
- **Notes:** Fails if new name already exists

##### `PersonManager.delete_person(name)`
```python
success = manager.delete_person("Alice")
```
- **Parameters:**
  - `name` (str): Person name to delete
- **Returns:** bool - True on success
- **Effect:** Deletes entire person folder and all photos
- **Notes:** 
  - Use with caution (no undo)
  - Call `engine.reload_faces()` after deletion

---

## 🎥 Camera Integration

The current GUI uses OpenCV for camera capture. Here's how to integrate it:

### Basic Camera Setup

```python
import cv2

# Open camera
cap = cv2.VideoCapture(0)  # 0 = first camera

# Configure (optional)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

# Read frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process with AI
    processed = engine.process_frame(frame)
    
    # Display
    cv2.imshow('Video', processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Frame Rate Optimization

**Current GUI issues:**
- Camera takes long to initialize (buffered frames)
- First frame display is slow

**Solutions implemented:**
```python
# Get native resolution (don't force unsupported resolutions)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Test read to ensure camera works
ret, test_frame = cap.read()
if not ret:
    print("Camera failed")
```

### Multiple Camera Support

```python
# Scan for available cameras
available_cameras = []
for i in range(10):  # Check first 10 indices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        available_cameras.append(i)
        cap.release()
```

---

## 🔄 Complete Workflow Example

### Training New Person

```python
from ai_core.engine import AIEngine
import cv2

# Initialize
engine = AIEngine()
manager = engine.manager

# Open camera
cap = cv2.VideoCapture(0)

# Capture 10 training photos
person_name = "Alice"
for i in range(10):
    ret, frame = cap.read()
    if ret:
        path = manager.save_training_photo(person_name, frame)
        print(f"Saved {path}")
    time.sleep(0.5)  # Wait between captures

# Reload face database
engine.reload_faces()

cap.release()
```

### Real-time Recognition

```python
from ai_core.engine import AIEngine
import cv2

# Initialize
engine = AIEngine()
engine.enable_faces = True
engine.enable_yolo = False  # Optional

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    processed = engine.process_frame(frame)
    
    # Display
    cv2.imshow('Recognition', processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Object Detection Only

```python
from ai_core.engine import AIEngine
import cv2

engine = AIEngine()
engine.enable_yolo = True
engine.enable_faces = False

# Configure YOLO
engine.object_system.conf = 0.6  # Higher confidence threshold
engine.object_system.show_fps = True

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        processed = engine.process_frame(frame)
        cv2.imshow('Detection', processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

---

## ⚙️ Configuration Guide

### Hardware Acceleration

**Check device:**
```python
from ai_core.config import config
print(config.device)  # 'cpu', 'cuda', or 'mps'
```

**Force CPU (for debugging):**
```python
import torch
torch.cuda.is_available = lambda: False  # Before importing ai_core
```

### Recognition Tuning

**Make recognition stricter:**
```python
from ai_core.config import config
config.confidence_threshold = 0.75  # Default: 0.6
config.detection_threshold = 0.95   # Default: 0.9
```

**Make recognition more lenient:**
```python
config.confidence_threshold = 0.4  # Accept more matches
config.detection_threshold = 0.8   # Accept lower quality faces
```

### YOLO Model Selection

| Model    | Size | Speed      | Accuracy | Use Case           |
|----------|------|------------|----------|--------------------|
| yolo11n  | 6MB  | Fastest    | Good     | Real-time, CPU     |
| yolo11s  | 21MB | Fast       | Better   | Balanced           |
| yolo11m  | 49MB | Medium     | Great    | Accuracy priority  |
| yolo11l  | 90MB | Slow       | Excellent| High accuracy      |
| yolo11x  | 141MB| Very Slow  | Best     | Maximum accuracy   |

```python
engine.change_yolo_model("yolo11s")  # Switch model
```

---

## 🐛 Common Issues & Solutions

### Issue: Face recognition not working

**Solution:**
```python
# 1. Check if database is loaded
print(len(engine.face_system.known_names))  # Should be > 0

# 2. Reload database
engine.reload_faces()

# 3. Check face folder structure
# known_faces/
#   Alice/
#     Alice_1.jpg
#     Alice_2.jpg
```

### Issue: Camera is slow

**Solution:**
```python
# 1. Use smaller YOLO model
engine.change_yolo_model("yolo11n")

# 2. Reduce frame size before processing
frame_small = cv2.resize(frame, (640, 480))
processed = engine.process_frame(frame_small)

# 3. Process every N frames
frame_count = 0
while True:
    ret, frame = cap.read()
    frame_count += 1
    if frame_count % 2 == 0:  # Process every 2nd frame
        processed = engine.process_frame(frame)
```

### Issue: Models not loading

**Solution:**
```python
# Check if model files exist
import os
print(os.path.exists("yolo11n.pt"))  # Should be True

# Check torch installation
import torch
print(torch.__version__)
```

---

## 📦 Dependencies

See `requirements.txt`:
```
opencv-python       # Camera and image processing
ultralytics         # YOLO object detection
torch              # PyTorch (deep learning)
torchvision        # Vision models
facenet-pytorch    # Face recognition
Pillow             # Image loading
numpy              # Numerical arrays
```

**Install:**
```bash
pip install -r requirements.txt
```

---

## 🚀 Building a New GUI

### Minimal Example (CLI)

```python
from ai_core.engine import AIEngine
import cv2

def main():
    engine = AIEngine(log_callback=print)
    engine.enable_faces = True
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed = engine.process_frame(frame)
        cv2.imshow('App', processed)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):  # Capture face
            name = input("Enter name: ")
            engine.manager.save_training_photo(name, frame)
            engine.reload_faces()
    
    cap.release()

if __name__ == "__main__":
    main()
```

### Qt Example Structure

```python
from PyQt6.QtWidgets import QApplication, QMainWindow
from ai_core.engine import AIEngine
import cv2

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = AIEngine(log_callback=self.log)
        self.cap = cv2.VideoCapture(0)
        # ... setup UI
    
    def log(self, message):
        self.log_widget.append(message)
    
    def process_frame(self):
        ret, frame = self.cap.read()
        if ret:
            processed = self.engine.process_frame(frame)
            # Convert to QPixmap and display
```

### Web API Example (Flask)

```python
from flask import Flask, Response
from ai_core.engine import AIEngine
import cv2

app = Flask(__name__)
engine = AIEngine()
engine.enable_faces = True

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = engine.process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
```

---

## 📝 Notes

- **Thread Safety:** None of the AI core classes are thread-safe. If using threading, ensure only one thread processes frames at a time.
- **Memory:** Face recognition loads all face encodings into RAM. Large databases (>1000 people) may require optimization.
- **Performance:** YOLO and FaceNet are both GPU-accelerated. CPU mode is significantly slower.
- **File Formats:** Only JPG and PNG supported for face images.

---

## 🔮 Future Improvements

Potential enhancements for future developers:

1. **Database Backend:** Replace file-based storage with SQLite/PostgreSQL
2. **Face Encoding Cache:** Cache encodings instead of recomputing
3. **Streaming:** Support RTSP/RTMP streams in addition to USB cameras
4. **Batch Processing:** Add video file processing capabilities
5. **REST API:** Create FastAPI backend for web/mobile frontends
6. **Multi-camera:** Support multiple simultaneous camera streams
7. **Face Clustering:** Auto-group unknown faces for easy labeling
8. **Confidence Calibration:** Better threshold tuning per person
9. **Export:** Export face database as embeddings file
10. **GPU Memory Management:** Better VRAM handling for large models

---

## 📄 License

*Add your license information here*

---

**For Questions:** Review this documentation first. The backend API is stable and well-tested. Feel free to experiment with new GUI frameworks while keeping the `ai_core/` intact.

**Happy Coding! 🚀**
