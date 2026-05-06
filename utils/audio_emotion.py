import numpy as np
import librosa

def _energy(y):
    return float(np.mean(y**2))

def _zcr(y):
    return float(np.mean(librosa.feature.zero_crossing_rate(y)))

def _centroid(y, sr):
    return float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

def _mfcc_mean(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return float(np.mean(mfcc))

def predict_audio_emotion(file_like, sr_target: int = 22050) -> dict:
    """
    Rule-based lightweight emotion estimate from a short audio clip.
    Returns dict with keys: label in {happy, sad, calm}, and features for debugging.
    """
    # Load audio from a file-like object (BytesIO) or path
    y, sr = librosa.load(file_like, sr=sr_target, mono=True)
    y = np.nan_to_num(y)

    e = _energy(y)
    z = _zcr(y)
    c = _centroid(y, sr)
    m = _mfcc_mean(y, sr)

    # Simple heuristic thresholds tuned for demo
    if e > 0.02 and c > 2000:
        label = "happy"
    elif e < 0.005 and c < 1500:
        label = "sad"
    else:
        label = "calm"

    return {
        "label": label,
        "features": {
            "energy": e,
            "zcr": z,
            "centroid": c,
            "mfcc_mean": m,
            "sr": sr
        }
    }
