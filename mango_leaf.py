# =======================
# DENSENET + VGG + PCA + RANDOM FOREST
# =======================

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, VGG19
from tensorflow.keras.applications.densenet import preprocess_input as dn_pre
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_pre
from tensorflow.keras.preprocessing.image import img_to_array

# --------------------------
# Ayarlar
# --------------------------
DATASET_DIR = "dataset"       
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
PCA_COMPONENTS = 250          
N_TREES = 500                 

np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------------------------
# Model Yükleme
# --------------------------
dn_model = DenseNet121(weights="imagenet", include_top=False, pooling="avg")
vgg_model = VGG19(weights="imagenet", include_top=False, pooling="avg")

# --------------------------
# Görüntü yükleme
# --------------------------
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img

# --------------------------
# Feature extraction
# --------------------------
def extract_features(paths):
    imgs = []

    for p in paths:
        img = load_image(p)
        img = img_to_array(img)
        imgs.append(img)

    imgs = np.array(imgs)

    # DenseNet features
    dn_in = dn_pre(imgs.copy())
    f_dn = dn_model.predict(dn_in, batch_size=BATCH_SIZE, verbose=0)

    # VGG19 features
    vg_in = vgg_pre(imgs.copy())
    f_vg = vgg_model.predict(vg_in, batch_size=BATCH_SIZE, verbose=0)

    # Concatenate
    features = np.concatenate([f_dn, f_vg], axis=1)
    return features

# --------------------------
# Dataset sınıf isimlerini oku
# --------------------------
classes = sorted(os.listdir(DATASET_DIR))

all_features = []
all_labels = []

print("\n=== FEATURE EXTRACTION BASLADI ===\n")

for cls in classes:
    cls_folder = os.path.join(DATASET_DIR, cls)
    img_paths = [
        os.path.join(cls_folder, f)
        for f in os.listdir(cls_folder)
        if f.lower().endswith(".jpg")
    ]

    print(f"Processing class {cls} -> {len(img_paths)} images")

    feats = extract_features(img_paths)

    all_features.append(feats)
    all_labels += [cls] * len(feats)

# Hepsini birleştir
X = np.vstack(all_features)
y = np.array(all_labels)

print("\nFeature shape (Dense+VGG):", X.shape)

# --------------------------
# LABEL ENCODING
# --------------------------
le = LabelEncoder()
y_enc = le.fit_transform(y)

# --------------------------
# SCALING
# --------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# PCA
# --------------------------
pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
X_pca = pca.fit_transform(X_scaled)

print("PCA output shape:", X_pca.shape)

# --------------------------
# TRAIN/TEST SPLIT (80/20)
# --------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_enc, test_size=0.2, shuffle=True, stratify=y_enc, random_state=SEED
)

# --------------------------
# RANDOM FOREST
# --------------------------
rf = RandomForestClassifier(n_estimators=N_TREES, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)

# --------------------------
# TEST EVALUATION
# --------------------------
y_pred = rf.predict(X_test)

print("\n=== CLASSIFICATION REPORT ===\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n=== CONFUSION MATRIX ===\n")
print(confusion_matrix(y_test, y_pred))

# --------------------------
# Save model
# --------------------------
joblib.dump({
    "rf": rf,
    "scaler": scaler,
    "pca": pca,
    "le": le
}, "mango_pca_rf_model.joblib")

print("\nModel saved -> mango_pca_rf_model.joblib\n")
