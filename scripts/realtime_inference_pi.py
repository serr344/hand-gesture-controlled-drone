from pathlib import Path
from collections import Counter, deque
import time

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent

HAND_MODEL = ROOT / "models" / "hand_landmarker.task"
ONNX_MODEL = ROOT / "models" / "gesture_mlp.onnx"

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CONFIDENCE_THRESHOLD = 0.90
SMOOTHING_WINDOW = 7
MIN_AGREEMENT = 5

CLASSES = [
    "fist",
    "like",
    "dislike",
    "one",
    "peace",
    "palm",
    "no_gesture",
]

COMMANDS = {
    "fist": "HOLD",
    "like": "MOVE UP",
    "dislike": "MOVE DOWN",
    "one": "MOVE LEFT",
    "peace": "MOVE RIGHT",
    "palm": "PAUSE",
    "no_gesture": "NO COMMAND",
}


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

    return (points / scale).reshape(1, 63)


def get_stable_gesture(history):
    if len(history) < SMOOTHING_WINDOW:
        return None

    label, count = Counter(history).most_common(1)[0]

    if label is None or count < MIN_AGREEMENT:
        return None

    return label


def main():
    if not HAND_MODEL.exists():
        raise FileNotFoundError(HAND_MODEL)

    if not ONNX_MODEL.exists():
        raise FileNotFoundError(ONNX_MODEL)

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(ONNX_MODEL),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(HAND_MODEL)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX, cv2.CAP_V4L2
    )

    if not camera.isOpened():
        raise RuntimeError("Unable to open USB webcam.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    history = deque(maxlen=SMOOTHING_WINDOW)

    start_time = time.perf_counter()
    previous_time = start_time
    previous_timestamp = -1

    fps = 0.0
    processed_frames = 0
    last_print_time = start_time

    print("Raspberry Pi Gesture Recognition")
    print("ONNX Runtime initialized.")
    print("Drone communication: DISABLED")
    print("Press Ctrl+C to stop.\n")

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

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb),
                )

                timestamp_ms = int(
                    (time.perf_counter() - start_time) * 1000
                )
                timestamp_ms = max(
                    timestamp_ms,
                    previous_timestamp + 1,
                )
                previous_timestamp = timestamp_ms

                result = detector.detect_for_video(
                    image, timestamp_ms
                )

                gesture = "NO HAND"
                confidence = 0.0
                candidate = None

                if result.hand_landmarks:
                    features = normalize_landmarks(
                        result.hand_landmarks[0]
                    )

                    if features is not None:
                        logits = session.run(
                            None,
                            {input_name: features},
                        )[0][0]

                        exp_logits = np.exp(
                            logits - np.max(logits)
                        )
                        probabilities = (
                            exp_logits / np.sum(exp_logits)
                        )

                        index = int(np.argmax(probabilities))
                        confidence = float(probabilities[index])
                        gesture = CLASSES[index]

                        if confidence >= CONFIDENCE_THRESHOLD:
                            candidate = gesture

                history.append(candidate)

                stable_gesture = get_stable_gesture(history)

                command = (
                    COMMANDS.get(stable_gesture, "NO COMMAND")
                    if candidate is not None
                    and stable_gesture is not None
                    else "NO COMMAND"
                )

                now = time.perf_counter()
                elapsed = now - previous_time
                previous_time = now

                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    fps = (
                        0.9 * fps + 0.1 * instant_fps
                        if fps > 0
                        else instant_fps
                    )

                processed_frames += 1

                if now - last_print_time >= 1.0:
                    print(
                        f"Gesture: {gesture:12s} | "
                        f"Confidence: {confidence:5.1%} | "
                        f"Command: {command:12s} | "
                        f"FPS: {fps:5.1f}",
                        flush=True,
                    )
                    last_print_time = now

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        camera.release()

        total_time = time.perf_counter() - start_time
        average_fps = (
            processed_frames / total_time
            if total_time > 0 else 0
        )

        print(f"\nFrames processed: {processed_frames}")
        print(f"Average FPS: {average_fps:.2f}")
        print("Camera released.")


if __name__ == "__main__":
    main()
