import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_MODEL_PATH = Path("models/hand_landmarker.task")
CNN_MODEL_PATH = Path("models/gesture_cnn.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")
IMG_SIZE = 160
CROP_PADDING = 0.35
CONFIDENCE_THRESHOLD = 0.55
SMOOTHING_FRAMES = 5

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def load_class_names():
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    names = [data[str(i)]["display_name"] for i in range(len(data))]
    if len(names) != 6:
        raise ValueError(f"Expected 6 class names, found {len(names)}")
    return names


def load_cnn():
    model = tf.keras.models.load_model(str(CNN_MODEL_PATH), compile=False)
    if tuple(model.input_shape[1:]) != (IMG_SIZE, IMG_SIZE, 3):
        raise ValueError(f"Unexpected model input shape: {model.input_shape}")
    if model.output_shape[-1] != 6:
        raise ValueError(f"Expected 6 outputs, got {model.output_shape[-1]}")
    return model


# Exact square-crop geometry used in the Colab training pipeline.
# Input/output are RGB; output size is 160x160.
def square_hand_crop(image, landmarks, padding=CROP_PADDING):
    height, width = image.shape[:2]
    xs = np.array([p.x * width for p in landmarks])
    ys = np.array([p.y * height for p in landmarks])

    x_center = (xs.min() + xs.max()) / 2
    y_center = (ys.min() + ys.max()) / 2
    box_size = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * padding)
    box_size = max(2, int(box_size))
    half = box_size // 2

    x1, y1 = int(x_center) - half, int(y_center) - half
    x2, y2 = x1 + box_size, y1 + box_size

    left, top = max(0, -x1), max(0, -y1)
    right, bottom = max(0, x2 - width), max(0, y2 - height)

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(width, x2), min(height, y2)
    crop = image[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return None

    crop = cv2.copyMakeBorder(
        crop, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def predict_gesture(model, crop):
    if crop is None:
        return None
    # MobileNetV2 preprocessing is already embedded in the model.
    batch = np.expand_dims(crop.astype(np.float32), axis=0)
    return model(batch, training=False).numpy()[0]


def draw_hand(frame, landmarks):
    height, width = frame.shape[:2]
    points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    for point in points:
        cv2.circle(frame, point, 4, (255, 0, 255), -1)


def draw_interface(frame, gesture, confidence, fps):
    height = frame.shape[0]
    cv2.rectangle(frame, (10, 10), (490, 135), (30, 30, 30), -1)
    cv2.putText(frame, f"Gesture: {gesture}", (20, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.1f}%", (20, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "P: print top 3 | Q/ESC: exit", (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def print_top_predictions(probabilities, class_names):
    if probabilities is None:
        print("No valid CNN prediction available.")
        return
    indices = np.argsort(probabilities)[-3:][::-1]
    print("\nTop 3 CNN predictions (smoothed):")
    for index in indices:
        print(f"  {class_names[index]}: {probabilities[index] * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="CNN-Based Real-Time Hand Gesture Recognition"
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam device index")
    args = parser.parse_args()

    for path in (HAND_MODEL_PATH, CNN_MODEL_PATH, CLASS_NAMES_PATH):
        if not path.exists():
            print(f"ERROR: Required file not found: {path}")
            return

    try:
        print("Loading CNN model...")
        model = load_cnn()
        class_names = load_class_names()
        print("CNN loaded successfully.")
        print("Class order:", class_names)
    except Exception as error:
        print("MODEL ERROR:", error)
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open webcam {args.camera}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    base_options = python.BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH.resolve())
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    history = deque(maxlen=SMOOTHING_FRAMES)
    previous_time = time.time()
    last_timestamp_ms = 0
    last_probabilities = None

    print("\nHand Gesture Recognition Started")
    print("A second window shows the exact 160x160 CNN input.")
    print("Press P to print the top 3 predictions.")
    print("Press Q or ESC to exit.\n")

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Unable to read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_frame
                )

                timestamp_ms = max(
                    time.monotonic_ns() // 1_000_000,
                    last_timestamp_ms + 1
                )
                last_timestamp_ms = timestamp_ms
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                gesture = "No Hand Detected"
                confidence = 0.0
                last_probabilities = None

                if results.hand_landmarks:
                    landmarks = results.hand_landmarks[0]
                    draw_hand(frame, landmarks)
                    crop = square_hand_crop(rgb_frame, landmarks)

                    if crop is None:
                        history.clear()
                        gesture = "Crop Error"
                        try:
                            cv2.destroyWindow("CNN Input - 160x160")
                        except cv2.error:
                            pass
                    else:
                        # Exact image supplied to the CNN, shown in a separate window.
                        cv2.imshow(
                            "CNN Input - 160x160",
                            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                        )
                        probabilities = predict_gesture(model, crop)
                        history.append(probabilities)
                        average_probabilities = np.mean(history, axis=0)
                        last_probabilities = average_probabilities

                        predicted_index = int(np.argmax(average_probabilities))
                        confidence = float(average_probabilities[predicted_index]) * 100
                        gesture = (
                            class_names[predicted_index]
                            if confidence >= CONFIDENCE_THRESHOLD * 100
                            else "Uncertain"
                        )
                else:
                    history.clear()
                    try:
                        cv2.destroyWindow("CNN Input - 160x160")
                    except cv2.error:
                        pass

                now = time.time()
                fps = 1.0 / max(now - previous_time, 1e-6)
                previous_time = now
                draw_interface(frame, gesture, confidence, fps)
                cv2.imshow("CNN Hand Gesture Recognition System", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("p"):
                    print_top_predictions(last_probabilities, class_names)
                if key in (ord("q"), 27):
                    break

    except Exception as error:
        print("APPLICATION ERROR:", error)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed successfully.")


if __name__ == "__main__":
    main()