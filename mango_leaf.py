import os
import random
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from vmdpy import VMD
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from skimage.restoration import denoise_tv_chambolle

base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

categories = [
    'Anthracnose', 'Bacterial_Canker', 'Cutting_Weevil',
    'Die_Back', 'Gall_Midge', 'Healthy',
    'Powdery_Mildew', 'Sooty_Mould'
]

for category in categories:
    src_folder = os.path.join(base_dir, category)
    images = os.listdir(src_folder)
    random.shuffle(images)
    test_size = 50
    test_images = images[:test_size]
    train_images = images[test_size:]

    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

    for img in train_images:
        shutil.copy(os.path.join(src_folder, img),
                    os.path.join(train_dir, category, img))
    for img in test_images:
        shutil.copy(os.path.join(src_folder, img),
                    os.path.join(test_dir, category, img))

#----------------------------------------------------
train_dir = 'dataset/train'
categories = os.listdir(train_dir)
random_category = random.choice(categories)
category_path = os.path.join(train_dir, random_category)
random_image = random.choice(os.listdir(category_path))
image_path = os.path.join(category_path, random_image)

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(5,5))
plt.imshow(img_rgb)
plt.title(f"Kategori: {random_category}")
plt.axis('off')
plt.show()

#---------------------- Total Variation Filter ---------------------------
tv_denoised = denoise_tv_chambolle(img_rgb, weight=0.1, channel_axis=-1)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Orijinal Görüntü")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(tv_denoised)
plt.title("TV Filtresi Sonucu")
plt.axis('off')
plt.tight_layout()
plt.show(block=False)
plt.pause(8)
plt.close()

#--------------------- VMD (Varyasyonel Mod Ayrıştırma) ----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Her satır için 1D sinyal oluşturup modlara ayırmak
H, W = gray.shape
K = 3
modes_2d = np.zeros((K, H, W))

for i in range(H):
    row_signal = gray[i, :].astype(float)
    u_row, _, _ = VMD(row_signal, alpha=2000, tau=0, K=K, DC=0, init=1, tol=1e-7)
    for k in range(K):
        modes_2d[k, i, :] = u_row[k]

# Modları görsel olarak göstermek
plt.figure(figsize=(12,4))
for k in range(K):
    plt.subplot(1, K, k+1)
    plt.imshow(modes_2d[k], cmap='gray')
    plt.title(f"Mod {k+1}")
    plt.axis('off')
plt.suptitle("VMD ile Ayriştirilmiş Modlar (Yüksek-Orta-Düşük Frekans)")
plt.tight_layout()
plt.show()

#-------------------- Özellik Çıkarımı (Feature Extraction) ---------------------------
def extract_features_from_image(image_path):
    """Bir görüntüden TV filtresi + VMD tabanli özellikler çikarmak"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- TVF
    tv_filtered = denoise_tv_chambolle(img_rgb, weight=0.1, channel_axis=-1)
    gray_tv = cv2.cvtColor((tv_filtered * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # --- Grayscale sinyali oluşturmak
    signal = np.mean(gray_tv, axis=1)

    # --- VMD uygulamak
    u, _, _ = VMD(signal, alpha=2000, tau=0.0, K=3, DC=0, init=1, tol=1e-7)

    # --- Her mod için istatistiksel özellikler hesaplamak
    features = []
    for i in range(3):
        mode = u[i]
        mean_val = np.mean(mode)
        var_val = np.var(mode)
        energy_val = np.sum(mode ** 2)
        ent_val = entropy(np.abs(mode) + 1e-10)
        features.extend([mean_val, var_val, energy_val, ent_val])

    return features

#--------------- Tüm dataset için özellik tablosu -------------------
data = []
labels = []

for category in categories:
    category_path = os.path.join(train_dir, category)
    images = os.listdir(category_path)
    random.shuffle(images)
    for filename in images[:250]:  # 250 örnek alalim
        image_path = os.path.join(category_path, filename)
        features = extract_features_from_image(image_path)
        data.append(features)
        labels.append(category)

df = pd.DataFrame(data)
df['label'] = labels

print("Ozellik cikarimi tamamlandi.")
print(df.head())

#----------------------- Support Vector Machine (SVM) modeli ---------------------------
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)

print("\nSiniflandirma Sonuclari:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
