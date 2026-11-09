
import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
from collections import deque, Counter
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = Path("models/hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

FINGER_JOINTS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20)
}


# ==========================================
# GEOMETRY FUNCTIONS
# ==========================================

def distance(a, b):
    return np.linalg.norm(
        np.array([a.x, a.y]) - np.array([b.x, b.y])
    )


def angle(a, b, c):
    """Calculate angle ABC in degrees."""
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-8:
        return 0

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


# ==========================================
# FINGER DETECTION
# ==========================================

def is_finger_extended(landmarks, joints):
    mcp, pip, dip, tip = joints

    joint_angle = angle(
        landmarks[mcp],
        landmarks[pip],
        landmarks[dip]
    )

    wrist_distance_tip = distance(
        landmarks[tip], landmarks[0]
    )

    wrist_distance_pip = distance(
        landmarks[pip], landmarks[0]
    )

    return (
        joint_angle > 150
        and wrist_distance_tip > wrist_distance_pip * 1.05
    )


def is_thumb_extended(landmarks, palm_width):
    thumb_angle = angle(
        landmarks[1],
        landmarks[2],
        landmarks[3]
    )

    thumb_distance = distance(
        landmarks[4], landmarks[5]
    )

    return (
        thumb_angle > 130
        and thumb_distance / palm_width > 0.65
    )


# ==========================================
# GESTURE CLASSIFICATION
# ==========================================

def classify_gesture(landmarks):

    palm_width = distance(
        landmarks[5], landmarks[17]
    )

    if palm_width < 1e-6:
        return "Unknown"

    extended = {
        name: is_finger_extended(landmarks, joints)
        for name, joints in FINGER_JOINTS.items()
    }

    index = extended["index"]
    middle = extended["middle"]
    ring = extended["ring"]
    pinky = extended["pinky"]

    thumb = is_thumb_extended(landmarks, palm_width)

    thumb_index_distance = (
        distance(landmarks[4], landmarks[8]) / palm_width
    )

    # OK Sign: thumb and index fingertips form a small circle,
    # while the other three fingers remain extended.
    if (
        thumb_index_distance < 0.40
        and middle and ring and pinky
    ):
        return "OK Sign"

    # Open Palm: all fingers extended.
    if index and middle and ring and pinky and thumb:
        return "Open Palm"

    # Thumbs Up.
    if thumb and not any(extended.values()):
        return "Thumbs Up"

    # Peace Sign.
    if index and middle and not ring and not pinky:
        return "Peace Sign"

    # Pointing Finger.
    if index and not middle and not ring and not pinky:
        return "Pointing Finger"

    # Fist.
    if not any(extended.values()):
        return "Fist"

    return "Unknown"


# ==========================================
# DRAW LANDMARKS
# ==========================================

def draw_hand(frame, landmarks):

    height, width, _ = frame.shape
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    for point in points:
        cv2.circle(
            frame,
            point,
            4,
            (255, 0, 255),
            -1
        )


# ==========================================
# MAIN APPLICATION
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Real-Time Hand Gesture Recognition"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0
    )

    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print("ERROR: Hand landmark model not found.")
        return

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH.resolve())
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    history = deque(maxlen=5)
    previous_time = time.time()

    print("Hand Gesture Recognition Started")
    print("Press Q or ESC to exit.")

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:

            while cap.isOpened():

                success, frame = cap.read()

                if not success:
                    print("Unable to read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = time.monotonic_ns() // 1_000_000

                results = landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )

                gesture = "No Hand Detected"

                if results.hand_landmarks:

                    landmarks = results.hand_landmarks[0]

                    detected = classify_gesture(landmarks)

                    history.append(detected)

                    gesture = Counter(history).most_common(1)[0][0]

                    draw_hand(frame, landmarks)

                else:
                    history.clear()

                current_time = time.time()
                fps = 1.0 / max(
                    current_time - previous_time, 1e-6
                )
                previous_time = current_time

                cv2.rectangle(
                    frame, (10, 10), (450, 95),
                    (30, 30, 30), -1
                )

                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Press Q to Exit",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "Hand Gesture Recognition System",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

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
