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
- **Raspberry Pi 5:** ~20–24 FPS
- **ONNX:** 100% prediction agreement with PyTorch

## Usage

```bash
python scripts/download_models.py
python scripts/realtime_inference.py       # Windows
python scripts/realtime_inference_pi.py    # Raspberry Pi
```

## Status

Gesture recognition and Raspberry Pi deployment completed. **ESP32 communication and drone control integration are planned.**
