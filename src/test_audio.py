# import os
# import torch
# import joblib
# from src.model import AudioClassifier
# from src.audio_utils import load_audio
# from src.dataset import extract_features
# from src.inference import handle_decision

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "audio_classifier.pt")
# SCALER_PATH = os.path.join(BASE_DIR, "saved_model", "scaler.pkl")

# def test_audio(audio_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = AudioClassifier()
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.to(device).eval()

#     scaler = joblib.load(SCALER_PATH)

#     audio = load_audio(audio_path)
#     features = extract_features(audio)
#     features = scaler.transform([features])[0]

#     x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

#     with torch.no_grad():
#         probs = torch.softmax(model(x), dim=1)[0]

#     print("ORIGINAL:", probs[0].item())
#     print("PROMPT_INJECTION:", probs[1].item())
#     action = handle_decision(
#             audio_path=audio_path,
#             prediction=final_label,
#             confidence=confidence
# )

# print(f"System Action : {action}")

# if __name__ == "__main__":
#     audio_file = os.path.join(BASE_DIR, "data", "original", "original (1).wav")
#     test_audio(audio_file)


import os
import torch
import joblib

from src.model import AudioClassifier
from src.audio_utils import load_audio
from src.features import extract_features
from src.inference import handle_decision

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "audio_classifier.pt")
SCALER_PATH = os.path.join(BASE_DIR, "saved_model", "scaler.pkl")

# --------------------------------------------------
# Thresholds (slightly biased toward ORIGINAL)
# --------------------------------------------------
PI_THRESH = 0.75       # block if high confidence
ORIG_THRESH = 0.55    # allow original more easily


def test_audio(audio_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------- Load model -----------------
    model = AudioClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    scaler = joblib.load(SCALER_PATH)

    # ----------------- Preprocess -----------------
    audio = load_audio(audio_path)
    features = extract_features(audio)
    features = scaler.transform([features])[0]

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    # ----------------- Inference -----------------
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    orig_prob = probs[0].item()
    inj_prob = probs[1].item()

    # ----------------- Decision Logic -----------------
    if inj_prob >= PI_THRESH:
        final_label = "PROMPT_INJECTION"
        confidence = inj_prob
    elif orig_prob >= ORIG_THRESH:
        final_label = "ORIGINAL"
        confidence = orig_prob
    else:
        final_label = "UNCERTAIN"
        confidence = max(orig_prob, inj_prob)

    # ----------------- Security Action -----------------
    action = handle_decision(
        audio_path=audio_path,
        prediction=final_label,
        confidence=confidence
    )

    # ----------------- Output -----------------
    print("=" * 55)
    print("Audio File :", audio_path)
    print(f"ORIGINAL           : {orig_prob:.4f}")
    print(f"PROMPT_INJECTION   : {inj_prob:.4f}")
    print("Prediction         :", final_label)
    print(f"Confidence         : {confidence * 100:.2f}%")
    print("System Action      :", action)
    print("=" * 55)

    return final_label, confidence, action


# --------------------------------------------------
# Run directly
# --------------------------------------------------
if __name__ == "__main__":
    audio_file = os.path.join(
        BASE_DIR,
        "data",
        "original",
        "original (1).wav"
    )

    test_audio(audio_file)
