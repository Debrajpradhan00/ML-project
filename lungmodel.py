import cv2
import numpy as np
import joblib
from skimage.filters import gabor
from skimage.feature import graycomatrix, graycoprops

# ── Load model, scaler, and class names ─────────────────────
svm         = joblib.load("models/svm_model.pkl")
scaler      = joblib.load("models/scaler.pkl")
label_names = joblib.load("models/label_names.pkl")  # e.g. ["Normal", "Adenocarcinoma", ...]

print(f"✅ Model loaded — detects: {label_names}")


# ── PREPROCESSING ────────────────────────────────────────────
def normalize_ct_image(img: np.ndarray) -> np.ndarray:
    """Normalize images from ANY source/scanner."""
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip(img, p1, p99)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)

def enhance_gabor(img: np.ndarray) -> np.ndarray:
    img_float = img.astype(np.float32) / 255.0
    filt_real, filt_imag = gabor(img_float, frequency=0.6, theta=0)
    enhanced = np.sqrt(filt_real**2 + filt_imag**2).astype(np.float32)
    if enhanced.max() == enhanced.min():
        return np.zeros_like(img, dtype=np.uint8)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced.astype(np.uint8)

def segment_lung(img: np.ndarray):
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask    = (labels == largest).astype(np.uint8) * 255
    return cv2.bitwise_and(img, img, mask=mask), mask

def extract_glcm_features(img: np.ndarray) -> np.ndarray:
    img  = cv2.resize(img, (128, 128))
    img  = (img / 4).astype(np.uint8)
    glcm = graycomatrix(img, distances=[1, 2],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=64, symmetric=True, normed=True)
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        features.extend(graycoprops(glcm, prop).flatten())
    features += [np.mean(img), np.std(img), float(np.median(img)),
                 float(np.percentile(img, 25)), float(np.percentile(img, 75))]
    return np.array(features)


# ── MAIN PREDICT ─────────────────────────────────────────────
def predict_image(img_bytes: bytes) -> dict:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image.")

    # Full pipeline
    img          = normalize_ct_image(img)
    enhanced     = enhance_gabor(img)
    segmented, _ = segment_lung(enhanced)
    features     = extract_glcm_features(segmented)
    scaled       = scaler.transform([features])

    prediction    = svm.predict(scaled)[0]
    probabilities = svm.predict_proba(scaled)[0]

    label      = label_names[prediction]
    confidence = round(float(probabilities[prediction]) * 100, 2)

    # Low confidence guard — model is unsure
    if confidence < 55:
        label = "Uncertain"

    # Per-class probability dict  { "Normal": 12.5, "Adenocarcinoma": 78.3, ... }
    prob_dict = {
        label_names[i]: round(float(probabilities[i]) * 100, 2)
        for i in range(len(label_names))
    }

    return {
        "label":         label,
        "prediction":    int(prediction),
        "confidence":    confidence,
        "probabilities": prob_dict,
        # backward-compat keys
        "prob_normal":   prob_dict.get("Normal", 0),
        "prob_cancer":   round(100 - prob_dict.get("Normal", 0), 2),
    }