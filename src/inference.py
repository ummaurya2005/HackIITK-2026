import os
import shutil
from datetime import datetime

# thresholds (tunable)
PI_BLOCK_THRESH = 0.75
ORIG_ALLOW_THRESH = 0.60

BASE_FLAG_DIR = "flagged_inputs"


def handle_decision(
    audio_path: str,
    prediction: str,
    confidence: float
):
    """
    Returns: action (ALLOW / WARNING / BLOCK)
    Stores flagged audio when needed
    """

    os.makedirs(BASE_FLAG_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(audio_path)

    # 🚫 BLOCK
    if prediction == "PROMPT_INJECTION" and confidence >= PI_BLOCK_THRESH:
        target_dir = os.path.join(BASE_FLAG_DIR, "blocked")
        os.makedirs(target_dir, exist_ok=True)

        shutil.copy(
            audio_path,
            os.path.join(target_dir, f"{timestamp}_{filename}")
        )

        return "BLOCK"

    # ⚠️ WARNING
    if prediction == "UNCERTAIN" or (
        prediction == "PROMPT_INJECTION" and confidence < PI_BLOCK_THRESH
    ):
        target_dir = os.path.join(BASE_FLAG_DIR, "warning")
        os.makedirs(target_dir, exist_ok=True)

        shutil.copy(
            audio_path,
            os.path.join(target_dir, f"{timestamp}_{filename}")
        )

        return "WARNING"

    # ✅ ALLOW
    return "ALLOW"
