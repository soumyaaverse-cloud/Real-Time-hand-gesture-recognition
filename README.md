# Hand Gesture Recognition

A real-time hand gesture recognition project built with Python, OpenCV, MediaPipe, and scikit-learn. The system uses custom-recorded webcam data to recognize five hand gestures and displays predictions live with confidence scores and temporal smoothing.

MediaPipe detects 21 hand landmarks, which are converted into 42 numerical features. A trained machine learning classifier then predicts the gesture based on the hand's shape.

## Features

- Real-time hand gesture recognition using a webcam
- Detection of 21 hand landmarks using MediaPipe
- 42-feature representation based on normalized hand coordinates
- Support for five predefined gestures
- Custom gesture dataset collection
- Comparison of KNN, SVM, and Random Forest classifiers
- Dataset exploration and PCA visualization
- Fresh-session model evaluation
- Prediction confidence threshold and temporal smoothing
- Average and 95th-percentile processing latency measurement

## Supported Gestures

1. Fist
2. Open Palm
3. Thumbs Up
4. Peace
5. Point

Gesture names and their corresponding collection keys can be configured in `common.py`.

## How It Works

```text
Webcam Frame
     |
     v
MediaPipe Hand Landmark Detection
     |
     v
21 Hand Landmarks
     |
     v
Feature Extraction (42 Values)
     |
     v
Trained Machine Learning Classifier
     |
     v
Confidence Threshold + Temporal Smoothing
     |
     v
Predicted Gesture Displayed on Screen
```

### Feature Extraction

The system uses the x and y coordinates of 21 hand landmarks to create a 42-dimensional feature vector.

Before classification, the features are normalized to improve consistency:

1. **Translation normalization:** The wrist landmark is moved to the origin.
2. **Scale normalization:** Coordinates are divided by the maximum distance from the wrist.
3. **Handedness normalization:** Left-hand coordinates are mirrored to help maintain consistent representations across hands.

The same feature-extraction function is used during data collection and live inference, helping ensure consistent preprocessing.

The classifier operates on landmark features rather than raw image pixels. However, lighting, hand visibility, and camera conditions can still affect landmark detection and prediction quality.

## Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV
- **Hand Tracking:** MediaPipe
- **Machine Learning:** Scikit-learn
- **Data Processing:** NumPy, Pandas
- **Model Persistence:** Joblib
- **Visualization:** Matplotlib

## Project Structure

```text
hand-gesture-recognition/
├── data/
│   ├── gestures.csv
│   └── test_session.csv
├── models/
│   └── gesture_model.joblib
├── outputs/
│   ├── confusion_matrix.png
│   └── pca.png
├── common.py
├── testwebcam.py
├── collectdata.py
├── exploredata.py
├── train.py
├── evaluate.py
├── runlive.py
├── requirements.txt
├── .gitignore
└── README.md
```

- `common.py` — Shared MediaPipe setup, landmark detection, and feature extraction.
- `testwebcam.py` — Checks webcam access and hand landmark detection.
- `collectdata.py` — Records labeled gesture samples into a CSV file.
- `exploredata.py` — Examines the dataset and generates a PCA visualization.
- `train.py` — Compares machine learning classifiers and saves the selected model.
- `evaluate.py` — Evaluates model performance on a separate recording session.
- `runlive.py` — Runs real-time gesture recognition with smoothing and latency measurements.
- `requirements.txt` — Lists project dependencies.
- `data/` — Stores collected gesture datasets.
- `models/` — Stores the trained model.
- `outputs/` — Stores evaluation plots and visualizations.

## Installation

### Prerequisites

- Python 3.9–3.11; Python 3.11 is recommended for the tested environment.
- A working webcam.
- Git, if cloning the repository.

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd hand-gesture-recognition
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the HTTPS URL of your GitHub repository.

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project uses the following dependencies:

```text
opencv-python
mediapipe==0.10.14
numpy<2
scikit-learn
pandas
matplotlib
joblib
```

MediaPipe and NumPy versions are constrained to maintain compatibility with the environment used during development.

## Usage

Run the following commands from the project root with the virtual environment activated.

### 1. Test your webcam

```bash
python testwebcam.py
```

Check that the webcam opens and MediaPipe detects your hand. If the wrong camera is selected, try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in the relevant script.

Press `Q` to quit.

### 2. Collect gesture data

```bash
python collectdata.py
```

Follow the instructions displayed by the script to record samples for the supported gestures.

The initial dataset contains 2,000 labeled samples, with 400 samples per gesture. You can collect additional samples to improve the diversity of the dataset.

For better generalization:

- Record different hand orientations and angles.
- Vary the distance between your hand and the camera.
- Test both hands.
- Collect samples under different lighting conditions.

### 3. Explore the dataset

```bash
python exploredata.py
```

This script examines the dataset and generates a PCA visualization saved to:

```text
outputs/pca.png
```

PCA projects the feature vectors into two dimensions to help visualize how gesture classes are distributed. Some overlap in this plot does not necessarily mean the classifier will confuse those classes in the original feature space.

### 4. Train the model

```bash
python train.py
```

The training pipeline compares KNN, SVM, and Random Forest classifiers using five-fold cross-validation. It then evaluates the selected model on a held-out random test split.

Generated files:

- `models/gesture_model.joblib` — Trained classifier.
- `outputs/confusion_matrix.png` — Confusion matrix for model evaluation.

### 5. Evaluate on a fresh recording session

```bash
python evaluate.py data/test_session.csv
```

The fresh-session dataset is evaluated separately from the training dataset.

A fresh recording session provides a more realistic assessment than randomly splitting neighboring frames from the same recording, although testing across additional people and recording conditions would provide stronger evidence of generalization.

If you want to collect another test session, follow the input and output behavior implemented in `collectdata.py` so the new recording is saved separately rather than unintentionally overwriting your existing dataset.

### 6. Run real-time gesture recognition

```bash
python runlive.py
```

The webcam window displays the predicted gesture, confidence score, processing latency, and estimated frame rate.

Two mechanisms help stabilize predictions:

- **Confidence threshold:** Predictions below a confidence threshold of `0.6` are ignored.
- **Temporal smoothing:** The displayed label is updated when at least four of the last seven predictions agree.

Press `Q` in the webcam window to exit.

### 7. Measure processing latency

When `runlive.py` exits, it reports the average and 95th-percentile latency for the measured hand landmark detection and classification stage.

The measurement excludes drawing and displaying the webcam frame.

If processing is slow, request a lower webcam resolution by adding these lines immediately after `cap = cv2.VideoCapture(0)` in `runlive.py`:

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

Actual performance depends on the computer, webcam, and recording conditions.

## Results

### Model Performance

| Metric | Result |
|---|---:|
| Best classifier | Random Forest |
| Random Forest cross-validation accuracy | 98.1% ± 0.5% |
| Random holdout test accuracy | 99.0% |
| Fresh-session test accuracy | 92.4% |

The random holdout result measures performance on a random split of the collected dataset. The fresh-session result is a more useful indication of performance on a separate recording session.

### Real-Time Performance

| Metric | Result |
|---|---:|
| Average processing latency | 36.4 ms |
| 95th-percentile processing latency | 58.0 ms |
| Frames measured | 725 |

These latency measurements apply to the landmark detection and classification stage during the recorded test run. They should not be interpreted as end-to-end webcam display latency or guaranteed performance on every device.

## Configuration

| Setting | File | Value |
|---|---|---|
| Supported gestures | `common.py` | 5 gestures |
| Samples per gesture in the initial dataset | `collectdata.py` | 400 |
| Minimum prediction confidence | `runlive.py` | 0.6 |
| Smoothing window | `runlive.py` | 7 predictions |
| Minimum agreement | `runlive.py` | 4 predictions |
| MediaPipe model complexity | `common.py` | 0 |

These values reflect the current project configuration and can be changed as needed.

## Troubleshooting

**Webcam does not open**
- Close other applications that may be using the camera.
- Try a different camera index, such as `1`.

**MediaPipe compatibility issues**
- Use the recommended Python version and the dependency versions listed in `requirements.txt`.
- Activate the correct virtual environment before installing or running the project.

**Fresh-session accuracy is lower than random holdout accuracy**
- Collect more varied training examples.
- Test different lighting conditions, hand orientations, and camera distances.
- Evaluate on additional independent recording sessions.

**Two gestures are frequently confused**
- Inspect `outputs/confusion_matrix.png`.
- Collect more representative samples for the confused classes.
- Retrain and evaluate the model again.

## Future Improvements

- Evaluate across more users and recording environments.
- Expand the gesture vocabulary.
- Improve robustness to occlusion and unusual hand orientations.
- Explore sequence-based models for dynamic gestures.
- Measure end-to-end latency and optimize sustained frame rate.

## Author

**Soumya Sarathe**

B.Tech, Biochemical and Biotechnology Engineering  
National Institute of Technology Agartala

## License

No license has been specified yet. Add a `LICENSE` file if you intend to define how others may use, modify, or distribute this project.