from pathlib import Path
from collections import deque, Counter
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
LANDMARK_MODEL = ROOT / "models" / "hand_landmarker.task"
CLASSIFIER_MODEL = ROOT / "models" / "gesture_mlp.pth"

CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.90
SMOOTHING_WINDOW = 7
MIN_AGREEMENT = 5

COMMANDS = {
    "fist": "HOLD",
    "like": "MOVE UP",
    "dislike": "MOVE DOWN",
    "one": "MOVE LEFT",
    "peace": "MOVE RIGHT",
    "palm": "PAUSE",
    "no_gesture": "NO COMMAND",
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class GestureMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def normalize_landmarks(landmarks):
    points = np.array(
        [[p.x, p.y, p.z] for p in landmarks],
        dtype=np.float32,
    )

    points -= points[0].copy()

    scale = np.max(
        np.linalg.norm(points[:, :2], axis=1)
    )

    if scale < 1e-6:
        return None

    return (points / scale).flatten()


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]

    points = [
        (int(p.x * width), int(p.y * height))
        for p in landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2,
        )

    for point in points:
        cv2.circle(
            frame, point, 4, (0, 0, 255), -1
        )


def get_stable_gesture(history):
    if len(history) < SMOOTHING_WINDOW:
        return None

    label, count = Counter(history).most_common(1)[0]

    if label is None or count < MIN_AGREEMENT:
        return None

    return label


def main():
    checkpoint = torch.load(
        CLASSIFIER_MODEL,
        map_location="cpu",
        weights_only=True,
    )

    classes = checkpoint["classes"]

    classifier = GestureMLP(len(classes))
    classifier.load_state_dict(
        checkpoint["model_state_dict"]
    )
    classifier.eval()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(LANDMARK_MODEL)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError("Unable to open webcam.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    history = deque(maxlen=SMOOTHING_WINDOW)
    previous_time = time.perf_counter()
    fps = 0.0

    print("Starting real-time gesture recognition...")
    print("Press Q to quit.")
    print("Drone communication is disabled.")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as detector:

            while True:
                success, frame = camera.read()

                if not success:
                    history.clear()
                    print("Camera frame unavailable.")
                    break

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb),
                )

                result = detector.detect(mp_image)

                gesture = "NO HAND"
                confidence = 0.0
                candidate = None

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    features = normalize_landmarks(landmarks)

                    draw_landmarks(frame, landmarks)

                    if features is not None:
                        input_tensor = torch.from_numpy(
                            features.reshape(1, 63)
                        )

                        with torch.inference_mode():
                            logits = classifier(input_tensor)
                            probabilities = torch.softmax(
                                logits, dim=1
                            )

                        confidence, index = torch.max(
                            probabilities, dim=1
                        )

                        confidence = confidence.item()
                        gesture = classes[index.item()]

                        if confidence >= CONFIDENCE_THRESHOLD:
                            candidate = gesture

                history.append(candidate)
                stable_gesture = get_stable_gesture(history)

                command = (
                    COMMANDS.get(stable_gesture, "NO COMMAND")
                    if stable_gesture is not None
                    else "NO COMMAND"
                )

                current_time = time.perf_counter()
                elapsed = current_time - previous_time
                previous_time = current_time

                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    fps = (
                        0.9 * fps + 0.1 * instant_fps
                        if fps > 0 else instant_fps
                    )

                cv2.rectangle(
                    frame, (0, 0), (440, 145),
                    (30, 30, 30), -1
                )

                cv2.putText(
                    frame, f"Gesture: {gesture}",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2
                )

                cv2.putText(
                    frame, f"Confidence: {confidence:.1%}",
                    (12, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2
                )

                cv2.putText(
                    frame, f"Command: {command}",
                    (12, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2
                )

                cv2.putText(
                    frame, f"FPS: {fps:.1f}",
                    (12, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2
                )

                cv2.imshow(
                    "Hand Gesture Recognition", frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
