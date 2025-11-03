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

# Ana dataset klasörünü belirt
base_dir = 'dataset'

# Train ve test klasörlerini oluştur
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Hastalık türleri
categories = [
    'Anthracnose', 'Bacterial_Canker', 'Cutting_Weevil',
    'Die_Back', 'Gall_Midge', 'Healthy',
    'Powdery_Mildew', 'Sooty_Mould'
]

# Her kategori için dosyaları ayır
for category in categories:
    src_folder = os.path.join(base_dir, category)
    images = os.listdir(src_folder)
    random.shuffle(images)

  # 50 test, 450 train
    test_size = 50
    test_images = images[:test_size]
    train_images = images[test_size:]

 # Alt klasörleri oluştur
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

 # Train verilerini taşı
    for img in train_images:
        shutil.copy(os.path.join(src_folder, img),
                    os.path.join(train_dir, category, img))

 # Test verilerini taşı
    for img in test_images:
        shutil.copy(os.path.join(src_folder, img),
                    os.path.join(test_dir, category, img))

print("Dataset train/test olarak ayrildi")

#----------------------------------------------------
# Veri seti yolunu belirle
train_dir = 'dataset/train'

# Kategoriler
categories = os.listdir(train_dir)

# Rastgele bir kategori seç
random_category = random.choice(categories)
category_path = os.path.join(train_dir, random_category)

# Rastgele bir resim seç
random_image = random.choice(os.listdir(category_path))
image_path = os.path.join(category_path, random_image)

# Görüntüyü oku (OpenCV BGR formatında okur)
img = cv2.imread(image_path)

# RGB formatına çevir (matplotlib doğru renkleri gösterebilsin)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Görüntüyü göster
plt.figure(figsize=(5,5))
plt.imshow(img_rgb)
plt.title(f"Kategori: {random_category}")
plt.axis('off')
plt.show()

#-------------------- Total Variation Filter (Toplam Varyasyon Filtresi) ----------------------------------
from skimage.restoration import denoise_tv_chambolle

# Görüntü yolundan oku
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Orijinal ve filtrelenmiş görüntüyü karşılaştır
tv_denoised = denoise_tv_chambolle(img_rgb, weight=0.1, channel_axis=-1)

# Görüntüleri karşılaştırmalı göster
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Orijinal Görüntü")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(tv_denoised)
plt.title("Total Variation Filtresi Sonucu")
plt.axis('off')         
                        # Burada toplam varyasyon filtresi kullanarak
                        # mango yaprağı görüntüsündeki istenmeyen gürültüleri temizliyoruz.
                        # Diğer klasik filtrelerin aksine,
                        # TV filtresi kenar bilgilerini koruyarak
                        # doku detaylarının kaybolmamasını sağlar.

plt.tight_layout()
plt.show(block=False)  # pencereyi kapatmadan devam edebilmek için
plt.pause(15)           # 15 saniye açık kalsın
plt.close()

#--------------------- VMD (Varyasyonel Mod Ayrıştırma) ----------------------------
# Görüntüyü gri tonlamaya çevir
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Her satırın ortalamasını alarak 1D sinyal oluştur
signal = np.mean(gray, axis=1)

# VMD parametreleri
alpha = 2000       # düzenlilik parametresi
tau = 0.0          # zaman adımı (genelde 0)
K = 3              # mod sayısı (3 mod)
DC = 0             # DC bileşeni dahil etme
init = 1           # başlangıç yöntemi (rastgele)
tol = 1e-7         # hata toleransı

# VMD uygula
u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)

# Sonuçları çizdir
plt.figure(figsize=(12, 8))
plt.subplot(K+1, 1, 1)
plt.plot(signal)
plt.title("Orijinal Sinyal (Grayscale Satir Ortalamasi)")
plt.xlabel("Satir")
plt.ylabel("Yoğunluk")

for i in range(K):
    plt.subplot(K+1, 1, i+2)
    plt.plot(u[i])
    plt.title(f"Mod {i+1}")
    plt.xlabel("Satir")
    plt.ylabel("Yoğunluk")

plt.tight_layout()
plt.show()

# VMD (Variational Mode Decomposition)
# Bir sinyali (veya görüntüyü) farklı frekans bileşenlerine ayırır.
# Her bileşene “mod” denir — bu modlar, görüntüdeki doku, renk geçişi,
# leke deseni gibi detayları farklı seviyelerde temsil eder.
# Burada amacım, görüntüyü birkaç bileşene ayırarak, hastalık belirtilerini daha belirgin hale getirmek.

#-------------------- Özellik Çıkarımı (Feature Extraction) ---------------------------
def extract_features_from_image(image_path):
    """Bir görüntüden TV filtresi + VMD tabanli özellikler çikarir"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Total Variation Filter (gürültü azaltma)
    tv_filtered = denoise_tv_chambolle(img_rgb, weight=0.1, channel_axis=-1)
    gray_tv = cv2.cvtColor((tv_filtered * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # --- Grayscale sinyali oluştur
    signal = np.mean(gray_tv, axis=1)

    # --- VMD uygula
    alpha = 2000
    tau = 0.0
    K = 3
    DC = 0
    init = 1
    tol = 1e-7
    u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)

    # --- Her mod için istatistiksel özellikler hesapla
    features = []
    for i in range(K):
        mode = u[i]
        mean_val = np.mean(mode)
        var_val = np.var(mode)
        energy_val = np.sum(mode ** 2)
        ent_val = entropy(np.abs(mode) + 1e-10)
        features.extend([mean_val, var_val, energy_val, ent_val])

    return features

#-------------------------- Tüm dataset için özellik tablosu ---------------------------------------
train_dir = 'dataset/train'
categories = os.listdir(train_dir)

data = []
labels = []

for category in categories:
    category_path = os.path.join(train_dir, category)
    for filename in os.listdir(category_path)[:50]:  # her sınıftan 50 örnek (hızlı deneme için)
        image_path = os.path.join(category_path, filename)
        features = extract_features_from_image(image_path)
        data.append(features)
        labels.append(category)

df = pd.DataFrame(data)
df['label'] = labels

print("Ozellik cikarimi tamamlandi")
print(df.head())

#----------------------- Support Vector Machine (SVM) modeli ---------------------------
# Veriyi ayır
X = df.drop('label', axis=1)
y = df['label']

# Train-test böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Özellikleri ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM modeli
svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_train_scaled, y_train)

# Tahmin
y_pred = svm.predict(X_test_scaled)

# Değerlendirme
print("\n Siniflandirma Sonuclari:")
print(classification_report(y_test, y_pred))
print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))