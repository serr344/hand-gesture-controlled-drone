# Hand Gesture Controlled Drone

Real-time hand gesture recognition using **MediaPipe, PyTorch, and ONNX Runtime**, deployed on Raspberry Pi 5 (2 GB) for future ESP32-based drone control.

## Dataset & Model

- **Dataset:** [HaGRID — 1000 Images per Class](https://huggingface.co/datasets/GestureDetectionConnoisseurs/hagrid_subsets/blob/main/hagrid-export_1000_images.zip)
- **Selected Classes:** 7 gestures, 7,000 images
- **Architecture:** MLP (63 → 128 → 64 → 7)
- **Test Accuracy:** 90.62%
- **Macro F1-Score:** 90.83%
- **Parameters:** 16,903

## Supported Gestures

`fist` · `like` · `dislike` · `one` · `peace` · `palm` · `no_gesture`

## Performance

- **Windows:** ~30 FPS
- **Raspberry Pi 5:** ~15–24 FPS
- **ONNX:** 100% prediction agreement with PyTorch

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/serr344/hand-gesture-controlled-drone.git
cd hand-gesture-controlled-drone
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# For Raspberry Pi
pip install -r requirements-pi.txt

# For Desktop / Windows
pip install -r requirements.txt
```

## Usage

```bash
python scripts/download_models.py
python scripts/realtime_inference.py       # Windows
python scripts/realtime_inference_pi.py    # Raspberry Pi
```

## Status

Gesture recognition and Raspberry Pi deployment completed. **ESP32 communication and drone control integration are planned.**