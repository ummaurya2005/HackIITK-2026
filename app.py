# from flask import Flask, render_template, request
# import os
# import torch
# import joblib
# import numpy as np
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# from src.model import AudioClassifier
# from src.audio_utils import load_audio
# from src.dataset import extract_features
# from src.inference import handle_decision
# from PIL import Image
# from torchvision import models, transforms
# import torch.nn as nn


# # -----------------------------
# # Setup
# # -----------------------------
# app = Flask(__name__)
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "audio_classifier.pt")
# SCALER_PATH = os.path.join(BASE_DIR, "saved_model", "scaler.pkl")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # -----------------------------
# # Load model & scaler ONCE
# # -----------------------------
# model = AudioClassifier()
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.to(device).eval()

# scaler = joblib.load(SCALER_PATH)

# # -----------------------------
# # Helper: Predict audio
# # -----------------------------
# def predict_audio(audio_path):
#     audio = load_audio(audio_path)
#     features = extract_features(audio)
#     features = scaler.transform([features])[0]

#     x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

#     with torch.no_grad():
#         probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

#     return probs[0], probs[1]  # ORIGINAL, PROMPT_INJECTION

# # -----------------------------
# # Load Image Adversarial Model
# # -----------------------------
# IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "image_adversarial_model.pt")

# image_model = models.resnet18(weights=None)
# image_model.fc = nn.Linear(image_model.fc.in_features, 2)

# checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=device)
# image_model.load_state_dict(checkpoint["model_state"])
# image_model.to(device).eval()

# image_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])
# def predict_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     image = image_transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         probs = torch.softmax(image_model(image), dim=1)[0]

#     clean_prob = probs[0].item()
#     adv_prob = probs[1].item()

#     # Decision logic
#     if adv_prob > 0.7:
#         action = "BLOCK"
#         label = "ADVERSARIAL"
#         confidence = adv_prob
#     elif adv_prob > 0.4:
#         action = "WARNING"
#         label = "SUSPICIOUS"
#         confidence = adv_prob
#     else:
#         action = "ALLOW"
#         label = "CLEAN"
#         confidence = clean_prob

#     return {
#         "label": label,
#         "confidence": round(confidence * 100, 2),
#         "action": action
#     }


# # -----------------------------
# # Compute metrics (offline demo)
# # -----------------------------
# def compute_metrics():
#     y_true = []
#     y_scores = []

#     for label, folder in [(0, "original"), (1, "injected")]:
#         folder_path = os.path.join(BASE_DIR, "data", folder)
#         if not os.path.exists(folder_path):
#             continue

#         for f in os.listdir(folder_path):
#             if f.endswith((".wav", ".mp3", ".m4a")):
#                 p0, p1 = predict_audio(os.path.join(folder_path, f))
#                 y_true.append(label)
#                 y_scores.append(p1)

#     y_pred = [1 if p > 0.4 else 0 for p in y_scores]

#     return {
#         "precision": round(precision_score(y_true, y_pred), 3),
#         "recall": round(recall_score(y_true, y_pred), 3),
#         "f1": round(f1_score(y_true, y_pred), 3),
#         "roc_auc": round(roc_auc_score(y_true, y_scores), 3)
#     }


# METRICS = compute_metrics()

# # -----------------------------
# # Routes
# # -----------------------------
# # @app.route("/", methods=["GET", "POST"])
# # def index():
# #     result = None

# #     if request.method == "POST":
# #         file = request.files["audio"]
# #         path = os.path.join("temp.wav")
# #         file.save(path)

# #         orig_prob, inj_prob = predict_audio(path)
# #         label = "ORIGINAL" if orig_prob > inj_prob else "PROMPT_INJECTION"
# #         confidence = max(orig_prob, inj_prob)

# #         result = {
# #             "label": label,
# #             "confidence": round(confidence * 100, 2)
# #         }
# # #         result = {
# # #         "prediction": final_label,
# # #         "confidence": round(confidence * 100, 2),
# # #         "action": action
# # # }


# #     return render_template("index.html", result=result, metrics=METRICS)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     result = None

#     if request.method == "POST":
#         file = request.files["audio"]
#         path = os.path.join("temp.wav")
#         file.save(path)

#         orig_prob, inj_prob = predict_audio(path)
#         label = "ORIGINAL" if orig_prob > inj_prob else "PROMPT_INJECTION"
#         confidence = max(orig_prob, inj_prob)

#         # 🔐 Security decision using inference logic
#         action = handle_decision(
#             audio_path=path,
#             prediction=label,
#             confidence=confidence
#         )

#         result = {
#             "label": label,
#             "confidence": round(confidence * 100, 2),
#             "action": action
#         }

#     return render_template("index.html", result=result, metrics=METRICS)

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request
import os
import torch
import joblib
import numpy as np

from src.model import AudioClassifier
from src.audio_utils import load_audio
from src.dataset import extract_features
from src.inference import handle_decision

from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

# -----------------------------
# Setup
# -----------------------------
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "audio_classifier.pt")
SCALER_PATH = os.path.join(BASE_DIR, "saved_model", "scaler.pkl")
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "image_adversarial_model.pt")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load AUDIO model & scaler
# -----------------------------
audio_model = AudioClassifier()
audio_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
audio_model.to(device).eval()

scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Offline validation metrics
# -----------------------------
AUDIO_METRICS = {
    "precision": 0.64,
    "recall": 0.61,
    "f1": 0.62,
    "roc_auc": 0.67
}

IMAGE_METRICS = {
    "precision": 0.50,
    "recall": 0.467,
    "f1": 0.483,
    "roc_auc": 0.537
}


# -----------------------------
# Audio prediction
# -----------------------------
def predict_audio(audio_path):
    audio = load_audio(audio_path)
    features = extract_features(audio)
    features = scaler.transform([features])[0]

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(audio_model(x), dim=1)[0].cpu().numpy()

    return probs[0], probs[1]  # ORIGINAL, PROMPT_INJECTION

# -----------------------------
# Load IMAGE adversarial model
# -----------------------------
image_model = models.resnet18(weights=None)
image_model.fc = nn.Linear(image_model.fc.in_features, 2)

checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=device)
image_model.load_state_dict(checkpoint["model_state"])
image_model.to(device).eval()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Image prediction
# -----------------------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(image_model(image), dim=1)[0]

    clean_prob = probs[0].item()
    adv_prob = probs[1].item()

    if adv_prob > 0.7:
        return {
            "label": "ADVERSARIAL",
            "confidence": round(adv_prob * 100, 2),
            "action": "BLOCK"
        }
    elif adv_prob > 0.4:
        return {
            "label": "SUSPICIOUS",
            "confidence": round(adv_prob * 100, 2),
            "action": "WARNING"
        }
    else:
        return {
            "label": "CLEAN",
            "confidence": round(clean_prob * 100, 2),
            "action": "ALLOW"
        }

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    audio_result = None
    image_result = None

    if request.method == "POST":

        # -------- AUDIO --------
        if "audio" in request.files and request.files["audio"].filename != "":
            audio_file = request.files["audio"]
            audio_path = os.path.join(BASE_DIR, "temp_audio.wav")
            audio_file.save(audio_path)

            orig_prob, inj_prob = predict_audio(audio_path)
            label = "ORIGINAL" if orig_prob > inj_prob else "PROMPT_INJECTION"
            confidence = max(orig_prob, inj_prob)

            action = handle_decision(
                audio_path=audio_path,
                prediction=label,
                confidence=confidence
            )

            audio_result = {
                "label": label,
                "confidence": round(confidence * 100, 2),
                "action": action
            }

        # -------- IMAGE --------
        if "image" in request.files and request.files["image"].filename != "":
            image_file = request.files["image"]
            image_path = os.path.join(BASE_DIR, "temp_image.jpg")
            image_file.save(image_path)

            image_result = predict_image(image_path)

    return render_template(
    "index.html",
    audio_result=audio_result,
    image_result=image_result,
    audio_metrics=AUDIO_METRICS,
    image_metrics=IMAGE_METRICS
)


if __name__ == "__main__":
    app.run(debug=True)
