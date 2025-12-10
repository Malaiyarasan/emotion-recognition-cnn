# Emotion Recognition using CNN (Computer Vision)

This project builds a **CNN-based emotion recognition model** that classifies
facial expressions (e.g., happy, sad, neutral) from images. It is designed as
a building block for human–robot interaction and safety-aware systems.

---

## 🔍 Problem

Robots and AI systems that interact with humans should understand basic
emotional signals. This project uses a convolutional neural network (CNN)
to classify emotions from face images.

---

## 🧠 Approach

1. **Dataset**
   - Face images organized into folders, one folder per emotion:
     - `happy/`, `sad/`, `neutral/`, `angry/`, etc.
   - All folders kept under `data/emotions/`.

2. **Preprocessing**
   - Convert images to grayscale or RGB.
   - Resize to a fixed size (e.g., 48×48 or 64×64).
   - Normalize pixel values to [0, 1].

3. **Model**
   - CNN with multiple convolution + pooling layers.
   - Dense layers on top for classification.
   - Softmax output over emotion classes.

4. **Training & Evaluation**
   - Train on training set, validate on validation set.
   - Report accuracy and classification report.

---

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- OpenCV (optional, for preprocessing)
- NumPy, Pandas

---

## 📁 Project Structure

```text
emotion-recognition-cnn/
│
├── data/
│   └── emotions/                # each subfolder = emotion label (placeholder)
│       ├── happy/
│       ├── sad/
│       ├── neutral/
│       └── ...
│
├── src/
│   └── train_emotion_cnn.py     # training script
│
└── README.md
