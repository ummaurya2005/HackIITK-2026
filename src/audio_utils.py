import librosa
from src.config import SAMPLE_RATE, DURATION

def load_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio = librosa.util.fix_length(audio, size=SAMPLE_RATE * DURATION)
    return audio
