
# Real-Time Hand Gesture Recognition System

## 1. Project Overview

The Real-Time Hand Gesture Recognition System is a computer vision application developed using Python, OpenCV and MediaPipe, with a Convolutional Neural Network (CNN) for gesture classification.

The system detects a human hand through a webcam, extracts hand landmarks, processes the hand region and recognizes predefined hand gestures in real time.

The project aims to demonstrate the application of computer vision and deep learning for natural human-computer interaction.

## 2. Objectives

- Detect human hands through a webcam.
- Extract and visualize 21 hand landmarks.
- Recognize six predefined hand gestures.
- Develop a CNN-based gesture classification model.
- Integrate the trained model with MediaPipe and OpenCV.
- Display recognized gestures in real time.

## 3. Supported Gestures

The system is designed to recognize:

1. Open Palm
2. Fist
3. Thumbs Up
4. Peace Sign
5. Pointing Finger
6. OK Sign

## 4. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| OpenCV | Webcam access and image processing |
| MediaPipe | Hand detection and landmark extraction |
| NumPy | Numerical operations |
| TensorFlow / Keras | CNN development and inference |
| MobileNetV2 | CNN architecture using transfer learning |
| Google Colab | GPU-based model training |

## 5. System Architecture

The proposed system consists of the following stages:

1. Webcam captures live video frames.
2. OpenCV processes the captured frames.
3. MediaPipe detects the hand and extracts 21 landmarks.
4. The hand region is extracted and preprocessed.
5. The trained CNN predicts the gesture class.
6. OpenCV displays the recognized gesture and confidence score.

## 6. Project Structure

```text
HandGestureRecognition/
│
├── backup/
│   └── main_rule_based.py
│
├── models/
│   ├── hand_landmarker.task
│   └── gesture_cnn.keras
│
├── training/
│   └── train_cnn.py
│
├── results/
│
├── main.py
├── requirements.txt
├── README.md
└── HandGesture_CNN_Training.ipynb
```

Note: Files marked as future training or integration outputs will be added as development progresses.

## 7. Installation

### Prerequisites

- Python 3.11 or later
- Webcam
- Debian or another compatible Linux distribution

### Clone or navigate to the project

```bash
cd ~/HandGestureRecognition
```

### Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download the MediaPipe hand landmark model

The model is stored in:

```text
models/hand_landmarker.task
```

## 8. Running the Application

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run the application:

```bash
python main.py
```

To select another webcam:

```bash
python main.py --camera 1
```

Press Q or ESC to close the application.

## 9. CNN Training

The CNN is planned to use MobileNetV2 transfer learning.

### Training configuration

| Parameter | Planned configuration |
|---|---|
| Architecture | MobileNetV2 |
| Input dimensions | 128 × 128 |
| Number of classes | 6 |
| Initial epochs | 8–12 |
| Batch size | 32 |
| Optimizer | Adam |
| Loss function | Categorical Crossentropy |
| Framework | TensorFlow / Keras |
| Training environment | Google Colab GPU |

The HaGRID hand gesture dataset subset is being considered for training.

Dataset source:
https://huggingface.co/datasets/ntsrigaud/hagrid-subset

The dataset will be inspected and filtered to retain the six required gesture classes.

### Model output

The trained model is intended to be exported as:

```text
models/gesture_cnn.keras
```

## 10. Model Evaluation

The trained CNN will be evaluated using:

- Training accuracy
- Validation accuracy
- Test accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Evaluation results will be documented after actual training and testing.

## 11. Testing

The application will be tested for:

- Webcam initialization.
- Hand landmark detection.
- Recognition of all six gestures.
- No-hand detection.
- Recognition stability.
- Real-time responsiveness.
- Application exit functionality.

Actual observations and screenshots will be added after testing.

## 12. Limitations

Potential limitations include:

- Performance variations due to lighting conditions.
- Gesture recognition errors caused by hand orientation.
- Partial hand visibility.
- Similarity between certain hand gestures.
- Dataset-to-webcam domain differences.
- Hardware-dependent inference performance.

## 13. Future Scope

Possible improvements include:

- Recognition of additional hand gestures.
- Dynamic gesture recognition.
- Sign language recognition.
- Gesture-controlled computer applications.
- Improved model accuracy through additional training data.
- Deployment on mobile and embedded devices.

## 14. Project Status

| Component | Status |
|---|---|
| Python environment | Completed |
| MediaPipe installation | Completed |
| Hand landmark model | Downloaded |
| Webcam integration | Tested |
| 21 landmark visualization | Tested |
| Initial rule-based classifier | Implemented |
| CNN dataset preparation | Pending |
| CNN training | Pending |
| CNN evaluation | Pending |
| CNN integration | Pending |
| Final application testing | Pending |

---

## Author

**K Bhaskara Pranav Sai**

Roll Number: 23STUCHH011373

Faculty of Science & Technology  
The ICFAI Foundation for Higher Education, Hyderabad
