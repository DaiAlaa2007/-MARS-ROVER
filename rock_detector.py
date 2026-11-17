import os
import numpy as np
import cv2
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

IMG_SIZE = 128
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rock_model.pkl")
_bundle = joblib.load(_MODEL_PATH)
_model = _bundle["model"]
_mineral_map = _bundle["mineral_map"]


def extract_features(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    feats = []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for ch in range(3):
        feats += [hsv[:, :, ch].mean(), hsv[:, :, ch].std()]
    for ch, rng in [(0, 180), (1, 256), (2, 256)]:
        h = cv2.calcHist([hsv], [ch], None, [8], [0, rng]).flatten()
        feats += list(h / h.sum())

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1, 3],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        feats.append(graycoprops(glcm, prop).mean())
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    feats += list(hist / hist.sum())

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area, per = cv2.contourArea(c), cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)
        circ = 4 * np.pi * area / (per ** 2) if per > 0 else 0
        feats += [area / IMG_SIZE**2, per / IMG_SIZE, w / h, circ]
    else:
        feats += [0, 0, 1, 0]
    return np.array(feats, dtype=np.float32)


def analyze_rock(image):
    """image: numpy array (BGR, من cv2.imread أو كاميرا OpenCV)."""
    if image is None or image.ndim != 3:
        return {"rock_type": "unknown", "possible_mineral": "unknown", "confidence": 0.0}
    feats = extract_features(image).reshape(1, -1)
    probs = _model.predict_proba(feats)[0]
    idx = int(probs.argmax())
    rock = str(_model.classes_[idx])
    return {
        "rock_type": rock,
        "possible_mineral": _mineral_map[rock],
        "confidence": round(float(probs[idx]) * 100, 1),
    }
