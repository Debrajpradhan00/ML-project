import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import gabor
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")

# ✅ FIX: main.py must NOT import from lungmodel
# lungmodel.py tries to load .pkl files at import time
# which causes FileNotFoundError before training is done

# ── EXACT folder names from your system ───────────────────────
DATA_DIRS = {
    "Normal":               "data.py/normal",
    "Adenocarcinoma":       "data.py/adenocarcinoma",
    "Large Cell Carcinoma": "data.py/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa",
    "Squamous Cell":        "data.py/squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa",
}

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


# ── LOAD IMAGES ───────────────────────────────────────────────
def load_images(folder):
    images = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
            else:
                print(f"   ⚠️  Could not read: {f}")
    return images


# ── PREPROCESSING ─────────────────────────────────────────────
def normalize_ct_image(img):
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    if p1 == p99:
        return np.zeros_like(img, dtype=np.uint8)
    img = np.clip(img, p1, p99)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)

def enhance_gabor(img):
    img_float = img.astype(np.float32) / 255.0
    filt_real, filt_imag = gabor(img_float, frequency=0.6, theta=0)
    enhanced = np.sqrt(filt_real**2 + filt_imag**2).astype(np.float32)
    if enhanced.max() == enhanced.min():
        return np.zeros_like(img, dtype=np.uint8)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced.astype(np.uint8)

def segment_lung(img):
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask    = (labels == largest).astype(np.uint8) * 255
    return cv2.bitwise_and(img, img, mask=mask), mask

def extract_glcm_features(img):
    img = cv2.resize(img, (128, 128))
    img = (img / 4).astype(np.uint8)
    glcm = graycomatrix(img, distances=[1, 2],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=64, symmetric=True, normed=True)
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        features.extend(graycoprops(glcm, prop).flatten())
    features += [np.mean(img), np.std(img), float(np.median(img)),
                 float(np.percentile(img, 25)), float(np.percentile(img, 75))]
    return np.array(features)

def process_images(images, label_name):
    features = []
    for i, img in enumerate(images):
        try:
            img      = normalize_ct_image(img)
            enhanced = enhance_gabor(img)
            seg, _   = segment_lung(enhanced)
            feat     = extract_glcm_features(seg)
            features.append(feat)
        except Exception as e:
            print(f"   ⚠️  Skipping image {i} in '{label_name}': {e}")
        if (i + 1) % 10 == 0:
            print(f"   Processed {i+1}/{len(images)}...")
    return features


# ── MAIN TRAINING ─────────────────────────────────────────────
def train_and_save():
    print("\n📂 Checking folders...")

    valid_classes = {}
    for name, folder in DATA_DIRS.items():
        if os.path.exists(folder):
            valid_classes[name] = folder
            print(f"   ✅ Found: {folder}")
        else:
            print(f"   ❌ Not found: {folder}")

    if len(valid_classes) < 2:
        print("\n❌ Need at least 2 valid folders. Check DATA_DIRS paths above.")
        return

    label_names  = list(valid_classes.keys())
    all_features = []
    all_labels   = []

    for label_idx, (label_name, folder) in enumerate(valid_classes.items()):
        images = load_images(folder)
        if len(images) == 0:
            print(f"⚠️  No images found in: {folder}")
            continue
        print(f"\n✅ {label_name}: {len(images)} images — extracting features...")
        feats = process_images(images, label_name)
        all_features.extend(feats)
        all_labels.extend([label_idx] * len(feats))

    if len(all_features) == 0:
        print("\n❌ No features extracted. Check your image folders.")
        return

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"\n✅ Total samples: {len(X)}")
    for i, name in enumerate(label_names):
        print(f"   [{i}] {name}: {np.sum(y == i)} samples")

    min_samples  = min(np.sum(y == i) for i in range(len(label_names)))
    use_stratify = min_samples >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if use_stratify else None
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print(f"\n   Train: {len(X_train)} | Test: {len(X_test)}")

    print(f"\n⏳ Training SVM on {len(label_names)} classes...")
    svm = SVC(
        kernel='rbf', C=10, gamma='scale',
        probability=True,
        decision_function_shape='ovr',
        random_state=42
    )
    svm.fit(X_train, y_train)

    y_pred   = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(label_names)), yticks=range(len(label_names)),
           xticklabels=label_names, yticklabels=label_names,
           ylabel='True Label', xlabel='Predicted Label',
           title=f'Confusion Matrix — {accuracy*100:.1f}% Accuracy')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
    plt.show()

    # ── Save all 3 model files ────────────────────────────────
    joblib.dump(svm,         "models/svm_model.pkl")
    joblib.dump(scaler,      "models/scaler.pkl")
    joblib.dump(label_names, "models/label_names.pkl")

    print("\n✅ Saved to models/:")
    print("   svm_model.pkl")
    print("   scaler.pkl")
    print("   label_names.pkl")
    print(f"\n🎉 Done! Model now detects: {label_names}")


# ── Entry point ───────────────────────────────────────────────
# ✅ FIX: Only run training here, never import lungmodel in main.py
if __name__ == "__main__":
    train_and_save()