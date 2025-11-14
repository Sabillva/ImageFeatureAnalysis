# mango_dense_features_rf_gpu_fixed.py
import os, random
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, VGG19
from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_pre
from tensorflow.keras.preprocessing.image import img_to_array

# ------------- Ayarlar -------------
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32       # GPU ile batch olarak işleme
USE_VGG = True
FEATURES_PKL = "features_concat_gpu_fixed.pkl"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------- Model yükle (feature extractor) -------------
densenet = DenseNet121(weights='imagenet', include_top=False, pooling='avg')
if USE_VGG:
    vgg = VGG19(weights='imagenet', include_top=False, pooling='avg')

# ------------- Görüntü okuma -------------
def load_image(path, target_size=IMG_SIZE):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Cannot read image:", path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img

# ------------- Batch feature extraction -------------
def extract_features_batch(image_paths):
    X_dn, X_vg, feats = [], [], []
    for path in image_paths:
        img = load_image(path)
        x = img_to_array(img)
        if USE_VGG:
            X_dn.append(x.copy())
            X_vg.append(x.copy())
        else:
            X_dn.append(x)
    # DenseNet
    X_dn = np.array(X_dn)
    X_dn = densenet_pre(X_dn)
    f_dn = densenet.predict(X_dn, batch_size=BATCH_SIZE, verbose=0)
    # VGG
    if USE_VGG:
        X_vg = np.array(X_vg)
        X_vg = vgg_pre(X_vg)
        f_vg = vgg.predict(X_vg, batch_size=BATCH_SIZE, verbose=0)
        feats = np.concatenate([f_dn, f_vg], axis=1)
    else:
        feats = f_dn
    return feats

# ------------- Feature extraction train/test -------------
if os.path.exists(FEATURES_PKL):
    print("Loading saved features:", FEATURES_PKL)
    df = pd.read_pickle(FEATURES_PKL)
    X_train = np.vstack(df[df['set']=='train']['features'].values)
    y_train = df[df['set']=='train']['label'].values
    X_test = np.vstack(df[df['set']=='test']['features'].values)
    y_test = df[df['set']=='test']['label'].values
else:
    categories = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR,d))])
    train_rows, test_rows = [], []

    # -------- TRAIN FEATURES --------
    for cls in categories:
        cls_folder = os.path.join(TRAIN_DIR, cls)
        imgs = [os.path.join(cls_folder, f) for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg','.png'))]
        print(f"Processing TRAIN class {cls} -> {len(imgs)} images")
        feats = extract_features_batch(imgs)
        for f, img_path in zip(feats, imgs):
            train_rows.append({'features': f, 'label': cls, 'set': 'train'})

    # -------- TEST FEATURES --------
    for cls in categories:
        cls_folder = os.path.join(TEST_DIR, cls)
        imgs = [os.path.join(cls_folder, f) for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg','.png'))]
        print(f"Processing TEST class {cls} -> {len(imgs)} images")
        feats = extract_features_batch(imgs)
        for f, img_path in zip(feats, imgs):
            test_rows.append({'features': f, 'label': cls, 'set': 'test'})

    # DataFrame oluştur
    df_train = pd.DataFrame(train_rows)
    df_test = pd.DataFrame(test_rows)
    df = pd.concat([df_train, df_test], ignore_index=True)
    df.to_pickle(FEATURES_PKL)

    X_train = np.vstack(df[df['set']=='train']['features'].values)
    y_train = df[df['set']=='train']['label'].values
    X_test = np.vstack(df[df['set']=='test']['features'].values)
    y_test = df[df['set']=='test']['label'].values
    print("Saved features to", FEATURES_PKL)

# ------------- Encode labels -------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
print("Classes:", le.classes_)

# ------------- Scale -------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ------------- RandomForest ---------
rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
rf.fit(X_train_s, y_train_enc)

# ------------- Evaluate ---------
y_pred = rf.predict(X_test_s)
print("\nClassification report (hold-out):\n", classification_report(y_test_enc, y_pred, target_names=le.classes_))
print("\nConfusion matrix:\n", confusion_matrix(y_test_enc, y_pred))

# 10-fold CV
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train_enc, y_test_enc])
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
scores = cross_val_score(rf, scaler.transform(X_all), y_all, cv=skf, scoring='accuracy', n_jobs=-1)
print("10-fold CV accuracies:", np.round(scores,4))
print("Mean CV acc: {:.4f} ± {:.4f}".format(scores.mean(), scores.std()))

# Save model
joblib.dump({'rf': rf, 'scaler': scaler, 'le': le}, "rf_dense_vgg_model_gpu_fixed.joblib")
print("Saved RF model -> rf_dense_vgg_model_gpu_fixed.joblib")
