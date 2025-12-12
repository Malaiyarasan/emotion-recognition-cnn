# Emotion Recognition (CNN)

Real-time emotion recognition from face images using a pretrained CNN model.

This repo contains a Colab notebook, a pretrained Keras model, and a live demo.

---

## 🚀 Live Demo
Try the model (upload real human face or use webcam):

👉 https://a5abcf716057fcd866.gradio.live

---

## ▶ Open in Google Colab
Open and run the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Malaiyarasan/emotion-recognition-cnn/blob/main/notebooks/emotion_recognition.ipynb)

---

## 📦 Files
- `models/emotion_pretrained.h5` — pretrained Keras model used in demo  
- `notebooks/emotion_recognition.ipynb` — Colab notebook (demo + helpers)  
- `src/train_emotion_cnn.py` — training script (optional)  
- `README.md` — this file

---

## 🔍 How it works (brief)
1. Face detection (OpenCV Haar cascade) → crop largest face.  
2. Resize & normalize image to model input (48×48 grayscale).  
3. Pretrained CNN predicts probabilities for 7 emotions:
   `angry, disgust, fear, happy, neutral, sad, surprise`.  
4. Gradio UI shows top probabilities and label.

---

## 🛠 Tech
- TensorFlow / Keras  
- OpenCV  
- Gradio  
- Python, NumPy, Pillow

---

## ⚠ Notes
- If your model expects a different input size / channels, update preprocessing in the notebook accordingly.  
- For improved real-world accuracy, fine-tune on larger datasets (AffectNet, RAF-DB) or use a pretrained backbone.

---

## 👤 Author
**Malaiyarasan M**  
AI & Data Engineer  
