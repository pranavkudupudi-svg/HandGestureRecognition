# Real-Time Hand Gesture Recognition Using CNN and MediaPipe

A webcam-based hand gesture recognition system that classifies six static hand gestures in real time. The application uses **MediaPipe** to detect the hand and locate its landmarks, then uses a **MobileNetV2 CNN** to classify a cropped RGB image.

> **Important:** This is image-based classification with landmark-guided preprocessing. The 21 MediaPipe landmark coordinates are used to locate and crop the hand; they are **not** passed directly to the CNN.

## Features

- Real-time webcam input using OpenCV
- Hand detection and 21-landmark extraction using MediaPipe Tasks
- Landmark-guided, padded square hand crop
- 160 × 160 RGB image input to the classifier
- Six-class gesture recognition using MobileNetV2
- Temporal smoothing to reduce frame-to-frame prediction flicker
- On-screen gesture label, confidence and FPS
- No-hand handling: skips classification when a hand is not detected
- Press `Q` or `Esc` to exit the application

## Supported Gestures

| Dataset class | Display label |
|---|---|
| `palm` | Open Palm |
| `fist` | Fist |
| `like` | Thumbs Up |
| `peace` | Peace Sign |
| `point` | Pointing Finger |
| `ok` | OK Sign |

## System Architecture

```text
Webcam
  |
  v
OpenCV frame capture
  |
  v
MediaPipe HandLandmarker
  | detects hand and returns 21 landmarks
  v
Landmark-guided hand crop
  | padded square crop, resized to 160 x 160 RGB
  v
MobileNetV2 CNN (TensorFlow / Keras)
  | six-class probabilities
  v
Temporal smoothing
  |
  v
Gesture label + confidence + FPS
```

If MediaPipe does not detect a hand, the application displays the no-hand state and skips cropping and CNN classification for that frame.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.13.5 | Main programming language (verified local environment) |
| OpenCV 5.0.0.93 (`opencv-contrib-python`) | Webcam capture, frame handling and display |
| MediaPipe 0.10.35 | Hand detection and landmark extraction |
| TensorFlow 2.21.0 / Keras | CNN model training and inference |
| NumPy 2.5.4 | Numerical operations and image arrays |
| Google Colab | Model training environment |
| NVIDIA Tesla T4 | Training accelerator used in Colab |
| Debian 13 (Trixie) | Local development and inference environment |

The real-time application runs locally on CPU. The NVIDIA Tesla T4 was used for training in Google Colab.
The pinned packages in `requirements.txt` match the local Python 3.13.5 environment; other Python versions have not been verified. The requirements file covers the webcam application's runtime. Notebook-specific packages are installed by the training notebook and are separate from the webcam runtime requirements.

## Model

- **Architecture:** MobileNetV2
- **Pretraining:** ImageNet
- **Approach:** Transfer learning
- **Base network:** Frozen feature extractor
- **Classification head:** Global Average Pooling, Dropout (0.3), Dense layer with six softmax outputs
- **Input:** 160 × 160 × 3 RGB image
- **Training:** 10 epochs

The saved model was loaded and checked locally: it has 2,265,670 parameters, with a frozen 2,257,984-parameter MobileNetV2 base and 7,686 trainable head parameters. It includes a training-time augmentation layer (horizontal flip, rotation, zoom and contrast) and MobileNetV2 input scaling; random augmentation is inactive during inference.

## Dataset and Training

The model was trained using a curated six-class subset of the [HaGRID dataset](https://huggingface.co/datasets/ntsrigaud/hagrid-subset).

| Split | Images |
|---|---:|
| Training | 3,648 |
| Validation | 432 |
| Test | 720 |
| **Total selected** | **4,800** |

Landmark-guided crop generation succeeded for 4,564 of the 4,800 images (95.08%). The recorded split counts are:

| Split | Selected images | Successfully processed |
|---|---:|---:|
| Training | 3,648 | 3,447 |
| Validation | 432 | 415 |
| Test | 720 | 702 |
| **Total** | **4,800** | **4,564** |

For each frame, MediaPipe's 21 landmarks determine a square hand crop with 35% padding. The crop is converted to RGB, resized to 160 × 160, and passed to the CNN as float32 pixel values in the 0–255 range. Preprocessing to the MobileNetV2 input range is embedded in the saved model, so the application does not normalize the input a second time. Training was performed in Google Colab using an NVIDIA Tesla T4 GPU. The MobileNetV2 base was frozen while the custom six-class classification head was trained.

### Model Training Notebook

The [training notebook](notebooks/HandGesture_CNN_Training.ipynb) is included in this repository and is also available in Google Colab. It contains dataset preparation, MediaPipe-based cropping, image preprocessing, MobileNetV2 transfer learning, model training, validation and test evaluation. The notebook targets Google Colab; its saved outputs were cleared before publication to avoid bundling dataset images and environment-specific output.

**[https://colab.research.google.com/drive/1edxmAFBRFrIyyfG_xHzGWF9wukTyEqoH?usp=sharing](https://colab.research.google.com/drive/1edxmAFBRFrIyyfG_xHzGWF9wukTyEqoH?usp=sharing)**

Training requires access to the HaGRID subset referenced in the notebook. Run the cells in order in a suitable Colab runtime to reproduce training and evaluation; the notebook has not been executed as part of this repository audit.

### Reported Results

| Metric | Result |
|---|---:|
| Final training accuracy | 83.35% |
| Best validation accuracy | 84.58% |
| Held-out test accuracy | **83.76%** |

Test accuracy was measured on 702 successfully processed held-out test images. It is an offline test-set result, **not** a measured live-webcam accuracy.

The live webcam application was manually checked with all six gestures. This was a qualitative functional check, not a formal statistical accuracy evaluation. Observed CPU inference speed was approximately 6.9–8.5 FPS and varied with runtime conditions.

## Project Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pranavkudupudi-svg/HandGestureRecognition.git
cd HandGestureRecognition
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the Required Model Assets

The required assets are included in the repository at the paths used by `main.py`:

- `models/gesture_cnn.keras` — trained six-class Keras model
- `models/class_names.json` — output-index to gesture-label mapping
- `models/hand_landmarker.task` — MediaPipe HandLandmarker model

No separate model download is required for these files. Keep the filenames and paths unchanged unless you also update `main.py`.

### 5. Run the Application

```bash
python main.py
```

Allow webcam access if prompted. Show one of the six supported static gestures to the camera. Press `Q` or `Esc` to close the application.
Use `--camera N` to select a different camera index, `--no-preview` to hide the separate crop-preview window, `P` to print the top three predictions, and `V` to toggle the crop preview.

## How the Application Works

1. OpenCV captures a frame from the webcam.
2. The frame is converted to the colour format required by MediaPipe.
3. MediaPipe detects the hand and returns 21 landmarks.
4. Landmark coordinates determine a padded, square hand region.
5. The crop is resized to 160 × 160 and prepared as an RGB image.
6. TensorFlow/Keras runs the image through the MobileNetV2 classifier.
7. Recent predictions are smoothed to reduce flicker.
8. OpenCV displays the predicted gesture, confidence and FPS.

MobileNetV2 preprocessing is embedded in the model. The inference pipeline must use the input range expected by the saved model and must not apply duplicate normalization.

## Limitations

- Recognises six static gestures only.
- Does not recognise dynamic gestures or full sign language.
- Lighting, occlusion, camera quality and hand orientation can affect performance.
- Similar-looking gestures can be confused.
- Live CPU inference is relatively slow on the development machine.
- Live testing was qualitative; no formal live accuracy benchmark was conducted.

## Future Improvements

- Add more gesture classes and varied training examples.
- Improve robustness to lighting, backgrounds and hand orientation.
- Fine-tune selected MobileNetV2 layers and evaluate the effect.
- Optimise inference through quantisation or other deployment techniques.
- Explore dynamic gesture recognition using frame sequences.
- Compare the CNN approach against landmark-based classifiers such as SVM or Random Forest.

## References

1. HaGRID subset: https://huggingface.co/datasets/ntsrigaud/hagrid-subset
2. MediaPipe Hand Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
3. TensorFlow: https://www.tensorflow.org/
4. MobileNetV2: https://arxiv.org/abs/1801.04381
5. OpenCV: https://opencv.org/

## Author

**K Bhaskara Pranav Sai**<br>
B.Tech — Artificial Intelligence<br>
The ICFAI Foundation for Higher Education, Hyderabad