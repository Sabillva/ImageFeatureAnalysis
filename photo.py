import numpy as np
import cv2
import os
from math import sqrt


base_dir = r"C:\Users\asus2\OneDrive\Desktop\Goruntu_Isleme_Ders_2\dataset"
emotions = ["smile", "fear", "upset", "suprised", "normal", "angry"]

# Haar Cascade ile yüz algılama yapılıyor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def energy(I):
    return np.sum(I ** 2)

def entropy(I):
    I_safe = I + 1e-12
    return -np.sum(I_safe * np.log2(I_safe))

def contrast(I):
    n, m = I.shape
    result = 0
    for i in range(n):
        for j in range(m):
            result += abs(i - j) ** 2 * I[i, j]
    return result

def correlation(I):
    n, m = I.shape
    mu_r = np.mean(np.sum(I, axis=1))
    mu_c = np.mean(np.sum(I, axis=0))
    sigma_r = np.std(np.sum(I, axis=1))
    sigma_c = np.std(np.sum(I, axis=0))
    result = 0
    for i in range(n):
        for j in range(m):
            result += ((i - mu_r) * (j - mu_c) * I[i, j]) / (sigma_r * sigma_c + 1e-12)
    return result

def max_probability(I):
    return np.max(I)

def homogeneity(I):
    n, m = I.shape
    result = 0
    for i in range(n):
        for j in range(m):
            result += I[i, j] / (1 + abs(i - j))
    return result

def dissimilarity(I):
    n, m = I.shape
    result = 0
    for i in range(n):
        for j in range(m):
            result += abs(i - j) * I[i, j]
    return result

def mean_value(I):
    return np.mean(I)


def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Foto okunmadi: {image_path}")
        return np.zeros(8)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Yüz tespiti yapılıyor
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        print(f"Yuz bulunamadi, tum goruntu kullanilacak: {image_path}")
        face_region = gray
    else:
        (x, y, w, h) = faces[0]
        face_region = gray[y:y + h, x:x + w]

    # Normalize ediliyor (0-1 arası)
    I = cv2.normalize(face_region.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    features = np.array([
        energy(I),
        entropy(I),
        contrast(I),
        correlation(I),
        max_probability(I),
        homogeneity(I),
        dissimilarity(I),
        mean_value(I)
    ])
    
    # Negatif veya NaN değerleri 0 yap
    features = np.nan_to_num(features, nan=0.0)
    features[features < 0] = 0.0

    # --- LOGARİTMİK ÖLÇEKLEME ---
    features = np.log1p(features)

    # --- NORMALİZASYON ---
    max_val = np.max(features)
    if max_val > 0:
        features = features / max_val
        
    return features


def euclidean_distance(f_test, f_train):
    return sqrt(np.sum((f_test - f_train) ** 2))


for emotion in emotions:
    folder_path = os.path.join(base_dir, emotion)
    if not os.path.exists(folder_path):
        print(f"Dosya bulunmadi: {folder_path}")
        continue

    
    image_paths = [os.path.join(folder_path, f"{emotion}{i}.jpg") for i in range(1, 11)]
    train_images = image_paths[:-1]
    test_image = image_paths[-1]

    print(f"\n========== {emotion.upper()} DUYGUSU ==========")
    print("\n=== Her bir training foto icin oznitelik degerleri ===\n")

    train_features = []
    for idx, img_path in enumerate(train_images):
        f = extract_features(img_path)
        train_features.append(f)
        print(f"F{idx+1} ({os.path.basename(img_path)}):")
        print(f"Energy={f[0]:.4f}, Entropy={f[1]:.4f}, Contrast={f[2]:.4f}, Correlation={f[3]:.4f}, "
              f"MaxProb={f[4]:.4f}, Homogeneity={f[5]:.4f}, Dissimilarity={f[6]:.4f}, Mean={f[7]:.4f}\n")

    test_features = extract_features(test_image)
    print(f"\nTest foto ({os.path.basename(test_image)}) icin oznitelik degerleri:")
    print(f"Energy={test_features[0]:.4f}, Entropy={test_features[1]:.4f}, Contrast={test_features[2]:.4f}, Correlation={test_features[3]:.4f}, "
          f"MaxProb={test_features[4]:.4f}, Homogeneity={test_features[5]:.4f}, Dissimilarity={test_features[6]:.4f}, Mean={test_features[7]:.4f}")

 
    print("\n=== D Mesafeleri (Oklid) ===")
    distances = []
    for idx, f in enumerate(train_features):
        D = euclidean_distance(test_features, f)
        distances.append((idx + 1, D))
        print(f"D{idx+1} (training image {idx+1}) = {D:.4f}")


    distances.sort(key=lambda x: x[1])
    print("\n=== D mesafeler siralanmis (en yakin -> en uzak) ===")
    for idx, (img_idx, dist) in enumerate(distances):
        print(f"{idx+1}. {emotion}{img_idx}.jpg -> D = {dist:.4f}")

    print(f"\nEn yakin goruntu: {emotion}{distances[0][0]}.jpg")
    print(f"Oklid mesafesi: {distances[0][1]:.4f}")
    print("========================================\n")