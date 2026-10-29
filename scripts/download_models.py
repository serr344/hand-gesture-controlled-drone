from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

MODEL_PATH = MODEL_DIR / "hand_landmarker.task"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        print(f"Model already exists: {MODEL_PATH}")
        return

    print("Downloading MediaPipe Hand Landmarker...")

    temp_path = MODEL_PATH.with_suffix(".task.part")

    try:
        urlretrieve(MODEL_URL, temp_path)
        temp_path.replace(MODEL_PATH)
    finally:
        temp_path.unlink(missing_ok=True)

    print(f"Model downloaded successfully: {MODEL_PATH}")


if __name__ == "__main__":
    main()
