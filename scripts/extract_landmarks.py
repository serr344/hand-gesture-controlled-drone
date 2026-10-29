from pathlib import Path
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# Project configuration
ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset" / "raw" / "data"
OUTPUT_DIR = ROOT / "dataset" / "landmarks"
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"

GESTURES = [
    "fist",
    "like",
    "dislike",
    "one",
    "peace",
    "palm",
    "no_gesture",
]

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# MediaPipe configuration
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def normalize_landmarks(landmarks):
    """Normalize 21 hand landmarks relative to the wrist."""

    points = np.array(
        [[p.x, p.y, p.z] for p in landmarks],
        dtype=np.float32,
    )

    # Translate the wrist to the origin.
    points -= points[0].copy()

    # Normalize using the maximum 2D distance from the wrist.
    distances = np.linalg.norm(points[:, :2], axis=1)
    scale = np.max(distances)

    if scale < 1e-6:
        return None

    points /= scale

    return points.flatten().tolist()


def extract_features(image_path, detector):
    """Detect a hand and extract normalized landmarks."""

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(image_rgb),
    )

    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    # Initial baseline: use the first detected hand.
    landmarks = result.hand_landmarks[0]

    return normalize_landmarks(landmarks)


def process_split(split, detector):
    """Extract features for one dataset split."""

    rows = []
    failures = []
    statistics = Counter()

    for gesture in GESTURES:
        image_dir = DATASET_DIR / split / gesture

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Missing directory: {image_dir}"
            )

        images = sorted(
            path for path in image_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in tqdm(
            images,
            desc=f"{split}/{gesture}",
        ):
            features = extract_features(image_path, detector)

            if features is None:
                failures.append({
                    "split": split,
                    "label": gesture,
                    "image": image_path.name,
                })
                statistics[(gesture, "failed")] += 1
                continue

            row = {
                f"feature_{i}": value
                for i, value in enumerate(features)
            }

            row["label"] = gesture
            row["image"] = image_path.name

            rows.append(row)
            statistics[(gesture, "success")] += 1

    columns = (
        [f"feature_{i}" for i in range(63)]
        + ["label", "image"]
    )

    dataframe = pd.DataFrame(rows, columns=columns)

    output_path = OUTPUT_DIR / f"{split}.csv"
    dataframe.to_csv(output_path, index=False)

    print(f"\n{split.upper()} SUMMARY")
    print("-" * 45)

    for gesture in GESTURES:
        success = statistics[(gesture, "success")]
        failed = statistics[(gesture, "failed")]
        total = success + failed

        detection_rate = (
            100 * success / total if total > 0 else 0
        )

        print(
            f"{gesture:12s} | "
            f"{success:4d}/{total:4d} | "
            f"{detection_rate:6.2f}%"
        )

    print(f"\nSaved: {output_path}")
    print(f"Total valid samples: {len(dataframe)}")

    return failures


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    all_failures = []

    print("Starting MediaPipe landmark extraction...\n")

    with HandLandmarker.create_from_options(options) as detector:
        for split in SPLITS:
            failures = process_split(split, detector)
            all_failures.extend(failures)

    failed_path = OUTPUT_DIR / "failed_detections.csv"

    pd.DataFrame(
        all_failures,
        columns=["split", "label", "image"],
    ).to_csv(failed_path, index=False)

    print("\n" + "=" * 45)
    print("LANDMARK EXTRACTION COMPLETED")
    print("=" * 45)

    print(f"Failed detections: {len(all_failures)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
