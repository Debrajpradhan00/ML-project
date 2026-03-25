import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import gabor
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
import warnings
warnings.filterwarnings("ignore")


# ── PATHS ──────────────────────────────────────
NORMAL_DIR   = "data.py/normal"
ABNORMAL_DIR = "data.py/adenocarcinoma"
OUTPUT_DIR   = "output"

# ── LOAD IMAGES ────────────────────────────────
def load_images(folder):
    images, names = [], []
    for f in os.listdir(folder):
        if f.lower().endswith(".png"):
            img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                names.append(f)
    return images, names

normal_imgs,   normal_names   = load_images(NORMAL_DIR)
abnormal_imgs, abnormal_names = load_images(ABNORMAL_DIR)

print(f"✅ Normal images loaded:   {len(normal_imgs)}")
print(f"✅ Abnormal images loaded: {len(abnormal_imgs)}")

# ── DISPLAY ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(normal_imgs[0], cmap='gray')
axes[0].set_title(f"Normal: {normal_names[0]}")
axes[0].axis('off')

axes[1].imshow(abnormal_imgs[0], cmap='gray')
axes[1].set_title(f"Abnormal: {abnormal_names[0]}")
axes[1].axis('off')

plt.suptitle("Stage 1 — Raw CT Images", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/stage1_raw_images.png")
plt.show()

print("✅ Stage 1 done! Check your output folder.")



# ─────────────────────────────────────────
# STAGE 2 — IMAGE ENHANCEMENT (Gabor Filter)
# ─────────────────────────────────────────
from skimage.filters import gabor

def enhance_gabor(img):
    # Normalize input to float [0, 1] as skimage expects
    img_float = img.astype(np.float32) / 255.0

    filt_real, filt_imag = gabor(img_float, frequency=0.6, theta=0)
    enhanced = np.sqrt(filt_real**2 + filt_imag**2)

    # Convert to float32 explicitly before cv2.normalize
    enhanced = enhanced.astype(np.float32)

    # Guard against flat (all-zero) output
    if enhanced.max() == enhanced.min():
        return np.zeros_like(img, dtype=np.uint8)

    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced.astype(np.uint8)

# Enhance first normal and first abnormal image
enhanced_normal   = enhance_gabor(normal_imgs[0])
enhanced_abnormal = enhance_gabor(abnormal_imgs[0])

# Display original vs enhanced side by side
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0][0].imshow(normal_imgs[0],    cmap='gray')
axes[0][0].set_title("Normal — Original")
axes[0][0].axis('off')

axes[0][1].imshow(enhanced_normal,   cmap='gray')
axes[0][1].set_title("Normal — Gabor Enhanced")
axes[0][1].axis('off')

axes[1][0].imshow(abnormal_imgs[0],  cmap='gray')
axes[1][0].set_title("Abnormal — Original")
axes[1][0].axis('off')

axes[1][1].imshow(enhanced_abnormal, cmap='gray')
axes[1][1].set_title("Abnormal — Gabor Enhanced")
axes[1][1].axis('off')

plt.suptitle("Stage 2 — Gabor Enhanced CT Images", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/stage2_enhanced.png")
plt.show()

print("✅ Stage 2 done! Enhanced images saved to output folder.")


def segment_lung(img):
    # Otsu thresholding to create binary mask
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological ops to clean up noise and fill holes
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  kernel, iterations=2)

    # Keep only the largest connected region (main lung area)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask    = (labels == largest).astype(np.uint8) * 255

    segmented = cv2.bitwise_and(img, img, mask=mask)
    return segmented, mask

seg_normal,   mask_normal   = segment_lung(enhanced_normal)
seg_abnormal, mask_abnormal = segment_lung(enhanced_abnormal)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0][0].imshow(enhanced_normal,   cmap='gray'); axes[0][0].set_title("Normal — Enhanced");    axes[0][0].axis('off')
axes[0][1].imshow(mask_normal,       cmap='gray'); axes[0][1].set_title("Normal — Mask");        axes[0][1].axis('off')
axes[0][2].imshow(seg_normal,        cmap='gray'); axes[0][2].set_title("Normal — Segmented");   axes[0][2].axis('off')
axes[1][0].imshow(enhanced_abnormal, cmap='gray'); axes[1][0].set_title("Abnormal — Enhanced");  axes[1][0].axis('off')
axes[1][1].imshow(mask_abnormal,     cmap='gray'); axes[1][1].set_title("Abnormal — Mask");      axes[1][1].axis('off')
axes[1][2].imshow(seg_abnormal,      cmap='gray'); axes[1][2].set_title("Abnormal — Segmented"); axes[1][2].axis('off')
plt.suptitle("Stage 3 — Lung Segmentation", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/stage3_segmented.png")
plt.show()
print("✅ Stage 3 done! Segmented images saved to output folder.")


# STAGE 4 — FEATURE EXTRACTION (GLCM Texture)
from skimage.feature import graycomatrix, graycoprops
def extract_glcm_features(img):
    # Resize all images to same size for consistency
    img = cv2.resize(img, (128, 128))

    # Rescale to 0-63 (64 levels) for faster GLCM computation
    img = (img / 4).astype(np.uint8)

    distances = [1, 2]
    angles    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm      = graycomatrix(img, distances=distances, angles=angles,
                              levels=64, symmetric=True, normed=True)

    # Extract 6 GLCM properties × 8 combinations (2 distances × 4 angles) = 48 features
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        values = graycoprops(glcm, prop).flatten()
        features.extend(values)

    # Add 5 basic statistical features
    features.append(np.mean(img))
    features.append(np.std(img))
    features.append(float(np.median(img)))
    features.append(float(np.percentile(img, 25)))
    features.append(float(np.percentile(img, 75)))

    return np.array(features)  # 53 features total per image


def process_all_images(images):
    """Run enhance → segment → extract features for every image."""
    features = []
    for i, img in enumerate(images):
        enhanced     = enhance_gabor(img)
        segmented, _ = segment_lung(enhanced)
        feat         = extract_glcm_features(segmented)
        features.append(feat)
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(images)} images...")
    return np.array(features)

print("\n⏳ Extracting features from all images (may take 1-2 min)...")
X_normal   = process_all_images(normal_imgs)
X_abnormal = process_all_images(abnormal_imgs)

print(f"✅ Stage 4 done!")
print(f"   Normal feature matrix:   {X_normal.shape}")
print(f"   Abnormal feature matrix: {X_abnormal.shape}")

# STAGE 5 — DATASET PREPARATION
X = np.vstack([X_normal, X_abnormal])
y = np.array([0] * len(X_normal) + [1] * len(X_abnormal))  # 0 = normal, 1 = cancer

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (critical for SVM performance)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\n✅ Stage 5 done! Dataset ready.")
print(f"   Total samples : {len(X)}")
print(f"   Train samples : {len(X_train)}")
print(f"   Test  samples : {len(X_test)}")
print(f"   Features/image: {X.shape[1]}")


# STAGE 6 — CLASSIFICATION (SVM)
# ═══════════════════════════════════════════════
print("\n⏳ Training SVM classifier...")

svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm.fit(X_train, y_train)

y_pred   = svm.predict(X_test)
y_prob   = svm.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Stage 6 done! SVM trained.")
print(f"   Accuracy: {accuracy * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Adenocarcinoma"]))

import joblib
import os

os.makedirs("models", exist_ok=True)
joblib.dump(svm,    "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("✅ Model saved to models/")