import cv2
import json
import time
import argparse
import numpy as np
import tensorflow as tf
import mediapipe as mp

from pathlib import Path
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "gesture_cnn.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
LANDMARKER_PATH = BASE_DIR / "models" / "hand_landmarker.task"

IMAGE_SIZE = 160
PADDING = 0.35
SMOOTHING_FRAMES = 5
CONFIDENCE_THRESHOLD = 0.50

# Output order must match the training label order:
# 0=palm, 1=fist, 2=like, 3=peace, 4=point, 5=ok


# =========================================================
# LOAD MODEL AND CLASS LABELS
# =========================================================

def load_class_names():
    """
    Supports the exported JSON structure:
    {
      "0": {"label": "palm", "display_name": "Open Palm"},
      ...
    }
    """

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict) and all(str(i) in data for i in range(6)):
        names = [
            data[str(i)]["display_name"]
            for i in range(6)
        ]
    elif isinstance(data, list):
        names = [
            item["display_name"] if isinstance(item, dict) else str(item)
            for item in data
        ]
    else:
        raise ValueError(
            "Unsupported class_names.json structure. "
            "Expected numeric keys 0-5 with display_name fields."
        )

    if len(names) != 6:
        raise ValueError(f"Expected 6 class names, found {len(names)}")

    return names


def load_model():
    print("[INFO] Loading CNN model...")

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    if tuple(model.input_shape[1:]) != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected model input shape: {model.input_shape}")

    if model.output_shape[-1] != 6:
        raise ValueError(f"Expected 6 model outputs, got {model.output_shape[-1]}")

    print("[INFO] CNN loaded.")
    print("[INFO] Output classes:", model.output_shape[-1])
    return model


def create_hand_detector():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(LANDMARKER_PATH)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35
    )

    return vision.HandLandmarker.create_from_options(options)


# =========================================================
# HAND CROPPING - SAME GEOMETRY AS TRAINING
# =========================================================

def square_hand_crop(image, landmarks, padding=PADDING):
    height, width = image.shape[:2]

    xs = np.array([point.x * width for point in landmarks])
    ys = np.array([point.y * height for point in landmarks])

    x_center = (xs.min() + xs.max()) / 2
    y_center = (ys.min() + ys.max()) / 2

    box_size = max(
        xs.max() - xs.min(),
        ys.max() - ys.min()
    ) * (1 + 2 * padding)

    box_size = max(2, int(box_size))
    half = box_size // 2

    x1 = int(x_center) - half
    y1 = int(y_center) - half
    x2 = x1 + box_size
    y2 = y1 + box_size

    left = max(0, -x1)
    top = max(0, -y1)
    right = max(0, x2 - width)
    bottom = max(0, y2 - height)

    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(width, x2)
    y2c = min(height, y2)

    crop = image[y1c:y2c, x1c:x2c]

    if crop.size == 0:
        return None

    crop = cv2.copyMakeBorder(
        crop,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    return cv2.resize(
        crop,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )


# =========================================================
# CNN PREPROCESSING AND PREDICTION
# =========================================================

def prepare_cnn_input(crop_rgb):
    """
    Training used RGB images with pixel values in [0,255].
    The exported model contains MobileNetV2 preprocessing.
    Do not normalize or preprocess a second time.
    """

    image = crop_rgb.astype(np.float32)
    return np.expand_dims(image, axis=0)


def predict_gesture(model, crop_rgb):
    model_input = prepare_cnn_input(crop_rgb)

    return model(
        model_input,
        training=False
    ).numpy()[0]


# =========================================================
# LANDMARK DRAWING
# =========================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]

    points = [
        (int(point.x * width), int(point.y * height))
        for point in landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    for point in points:
        cv2.circle(frame, point, 4, (255, 0, 255), -1)


# =========================================================
# DISPLAY
# =========================================================

def draw_dashboard(frame, gesture, confidence, fps):
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (430, 135),
        (25, 25, 25),
        -1
    )

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.1f}%",
        (25, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (25, 117),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "P: Top 3 | Q/ESC: Exit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


def print_top_three(probabilities, class_names):
    indices = np.argsort(probabilities)[::-1][:3]

    print("\nTop 3 CNN predictions:")

    for index in indices:
        print(
            f"  {class_names[index]}: "
            f"{probabilities[index] * 100:.2f}%"
        )


# =========================================================
# MAIN APPLICATION
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="CNN-Based Real-Time Hand Gesture Recognition"
    )

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    for path in (MODEL_PATH, CLASS_NAMES_PATH, LANDMARKER_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    class_names = load_class_names()

    print("[INFO] Class index mapping:")
    for index, name in enumerate(class_names):
        print(f"  {index} -> {name}")

    model = load_model()
    detector = create_hand_detector()

    camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        detector.close()
        raise RuntimeError(f"Cannot open camera {args.camera}")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prediction_history = deque(maxlen=SMOOTHING_FRAMES)

    previous_time = time.time()
    fps = 0.0
    show_preview = not args.no_preview

    print("\n[INFO] Webcam started.")
    print("[INFO] P = top three predictions")
    print("[INFO] V = toggle crop preview")
    print("[INFO] Q or ESC = exit")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("[WARNING] Cannot read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            # MediaPipe receives RGB.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = detector.detect(mp_image)

            gesture = "No Hand Detected"
            confidence = 0.0
            current_crop = None
            smoothed = None

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]

                draw_landmarks(frame, landmarks)

                # Crop from RGB frame, matching training pipeline.
                current_crop = square_hand_crop(
                    rgb_frame,
                    landmarks
                )

                if current_crop is not None:
                    probabilities = predict_gesture(
                        model,
                        current_crop
                    )

                    prediction_history.append(probabilities)

                    smoothed = np.mean(
                        np.asarray(prediction_history),
                        axis=0
                    )

                    best_index = int(np.argmax(smoothed))
                    confidence = float(smoothed[best_index]) * 100

                    gesture = (
                        class_names[best_index]
                        if confidence >= CONFIDENCE_THRESHOLD * 100
                        else "Uncertain"
                    )
                else:
                    prediction_history.clear()
                    gesture = "Crop Error"
            else:
                prediction_history.clear()

            now = time.time()
            elapsed = max(now - previous_time, 1e-6)
            previous_time = now

            instant_fps = 1.0 / elapsed
            fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps

            draw_dashboard(
                frame,
                gesture,
                confidence,
                fps
            )

            cv2.imshow(
                "CNN Hand Gesture Recognition System",
                frame
            )

            if show_preview:
                if current_crop is not None:
                    preview = cv2.cvtColor(
                        current_crop,
                        cv2.COLOR_RGB2BGR
                    )
                else:
                    preview = np.zeros(
                        (IMAGE_SIZE, IMAGE_SIZE, 3),
                        dtype=np.uint8
                    )

                cv2.imshow("CNN Input - 160x160", preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("p") and smoothed is not None:
                print_top_three(smoothed, class_names)

            elif key == ord("v"):
                show_preview = not show_preview
                if not show_preview:
                    try:
                        cv2.destroyWindow("CNN Input - 160x160")
                    except cv2.error:
                        pass

            elif key in (ord("q"), 27):
                break

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()
        print("\n[INFO] Application closed.")


if __name__ == "__main__":
    main()