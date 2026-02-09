# 🔐 Multimodal Prompt Injection & Adversarial Media Detection

A real-time multimodal AI security system to detect **prompt injection**, **deepfake audio**, and **adversarial images**, and prevent unauthorized actions in AI-powered systems.

---

## 🚀 Overview

AI systems that accept audio and image inputs are vulnerable to:
- Spoken prompt injection attacks
- Deepfake or synthetic voice impersonation
- Adversarial images with hidden instructions

This project introduces a **multimodal defense framework** that independently analyzes **audio and image inputs**, detects malicious patterns, and enforces **risk-aware security actions** — **ALLOW**, **WARNING**, or **BLOCK** — before inputs reach downstream AI agents.

---

## 🧠 System Architecture


Each modality is processed **independently**, ensuring robustness, modularity, and clear evaluation.

---

## 🎧 Audio Detection

- Supported formats: `.wav`, `.mp3`, `.m4a`
- Preprocessing: resampling, fixed-length normalization
- Feature extraction:
  - MFCC
  - Delta MFCC
  - Delta-Delta MFCC
- Lightweight neural network classifier (non-CNN)
- Optimized for low-latency inference

---

## 🖼️ Image Adversarial Detection

- Image resizing and normalization
- CNN-based detector (ResNet-18)
- Trained on adversarial and natural adversarial images
- Detects hidden instructions and perturbation-based attacks

---

## 🛡️ Risk-Aware Decision Engine

Instead of simple classification, the system applies confidence-based thresholds:

| Confidence Level | Action | Meaning |
|------------------|--------|--------|
| High malicious confidence | 🔴 BLOCK | Reject input |
| Medium confidence | 🟠 WARNING | Human review recommended |
| High benign confidence | 🟢 ALLOW | Safe to proceed |

This reduces false positives while maintaining strong security guarantees.

---

## 📊 Model Evaluation (Offline Validation)

Metrics are computed on labeled validation datasets (standard ML practice).

### 🎧 Audio Model
- Precision: **0.64**
- Recall: **0.61**
- F1-score: **0.62**
- ROC–AUC: **0.67**

### 🖼️ Image Model
- Precision: **0.71**
- Recall: **0.57**
- F1-score: **0.58**
- ROC–AUC: **0.54**

Metrics are reported **per modality** for transparent performance analysis.

---

## ⚡ Performance

- Audio inference latency: **< 100 ms**
- Image inference latency: **< 200 ms**
- Suitable for real-time or near-real-time deployment

---

## 🖥️ Web Interface

- Built with **Flask**
- Upload audio or image files
- Displays:
  - Prediction
  - Confidence score
  - Security action (ALLOW / WARNING / BLOCK)
- Shows separate evaluation metrics for audio and image models

---

## 📂 Project Structure


Each modality is processed **independently**, ensuring robustness, modularity, and clear evaluation.

---

## 🎧 Audio Detection

- Supported formats: `.wav`, `.mp3`, `.m4a`
- Preprocessing: resampling, fixed-length normalization
- Feature extraction:
  - MFCC
  - Delta MFCC
  - Delta-Delta MFCC
- Lightweight neural network classifier (non-CNN)
- Optimized for low-latency inference

---

## 🖼️ Image Adversarial Detection

- Image resizing and normalization
- CNN-based detector (ResNet-18)
- Trained on adversarial and natural adversarial images
- Detects hidden instructions and perturbation-based attacks

---


---

---

## ⚡ Performance

- Audio inference latency: **< 100 ms**
- Image inference latency: **< 200 ms**
- Suitable for real-time or near-real-time deployment

---

## 🖥️ Web Interface

- Built with **Flask**
- Upload audio or image files
- Displays:
  - Prediction
  - Confidence score
  - Security action (ALLOW / WARNING / BLOCK)
- Shows separate evaluation metrics for audio and image models

---

## 📂 Project Structure


---

## 🛠️ Installation & Run

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py


