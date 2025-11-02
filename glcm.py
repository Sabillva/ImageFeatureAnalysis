import numpy as np
import cv2
import os
from math import sqrt
from skimage.feature import graycomatrix, graycoprops

base_dir = r"C:\Users\asus2\OneDrive\Desktop\Goruntu_Isleme_Ders_2\dataset"
emotions = ["smile", "fear", "upset", "suprised", "normal", "angry"]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

glcm_features = ["energy", "entropy", "contrast", "correlation",
                 "max_probability", "homogeneity", "dissimilarity", "mean_value"]

offsets = {
    0: (0, 1),
    45: (-1, 1),
    90: (-1, 0),
    135: (-1, -1)
}

def compute_glcm_features(I, offset):
    """Belirtilen offset yönünde GLCM ve 8 özniteliği hesaplıyoruz"""
    I = cv2.normalize(I, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dy, dx = offset

    glcm = graycomatrix(I, distances=[1], angles=[np.arctan2(-dy, dx)],
                        levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]

    entropy = -np.sum(glcm * np.log2(glcm + 1e-12))
    max_prob = np.max(glcm)
    mean_val = np.mean(glcm)

    return np.array([
        energy, entropy, contrast, correlation,
        max_prob, homogeneity, dissimilarity, mean_val
    ])

def extract_features(image_path):
    """Bir görüntüden yüzü algılıyoruz ve her offset yönü için özellikleri hesaplıyoruz"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Foto okunamadi: {image_path}")
        return {angle: np.zeros(8) for angle in offsets.keys()}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        print(f"Yuz bulunamadi, tum goruntu kullanilacak: {image_path}")
        face_region = gray
    else:
        (x, y, w, h) = faces[0]
        face_region = gray[y:y + h, x:x + w]

    feature_dict = {}
    print(f"\n--- {os.path.basename(image_path)} icin GLCM oznitelikleri ---")
    for angle, offset in offsets.items():
        f = compute_glcm_features(face_region, offset)
        f = np.nan_to_num(f, nan=0.0)
        f[f < 0] = 0.0
        f = np.log1p(f)
        if np.max(f) > 0:
            f = f / np.max(f)
        feature_dict[angle] = f

        print(f"[Offset {angle}°]")
        for i, val in enumerate(f):
            print(f"  {glcm_features[i]} = {val:.6f}")
        print()

    return feature_dict

def euclidean_distance(f_test, f_train):
    """İki özellik vektörü arasındaki Öklid mesafesini hesaplıyoruz"""
    return sqrt(np.sum((f_test - f_train) ** 2))

for emotion in emotions:
    folder_path = os.path.join(base_dir, emotion)
    if not os.path.exists(folder_path):
        print(f"Dosya bulunamadi: {folder_path}")
        continue

    image_paths = [os.path.join(folder_path, f"{emotion}{i}.jpg") for i in range(1, 11)]
    train_images = image_paths[:-1]
    test_image = image_paths[-1]

    print(f"\n========== {emotion.upper()} DUYGUSU ==========\n")

    # Train ve test özelliklerini hesaplıyoruz
    train_features = [extract_features(p) for p in train_images]
    test_features = extract_features(test_image)
    print(f"\nTest foto olarak ({os.path.basename(test_image)}) kabul edildi.\n")

    print("\n=== Oklid Mesafeleri (Her Offset icin Ayri Hesaplandi) ===")
    for angle in offsets.keys():
        print(f"\n--- Offset {angle}° ---")
        distances = []
        for idx, f_train in enumerate(train_features):
            D = euclidean_distance(test_features[angle], f_train[angle])
            distances.append((idx + 1, D))
            print(f"D{idx+1}_{angle}° = {D:.4f}")

        # Mesafeleri sıralıyoruz
        distances.sort(key=lambda x: x[1])

        print(f"\n[Offset {angle}° icin en yakin goruntuler (kucukten buyuge)]:")
        for rank, (img_idx, dist) in enumerate(distances, 1):
            print(f"{rank}. {emotion}{img_idx}.jpg -> D = {dist:.4f}")

        # En yakını belirtiyoruz
        print(f"\n>>> En yakin (Offset {angle}°): {emotion}{distances[0][0]}.jpg")
        print(f"Oklid mesafesi: {distances[0][1]:.4f}")
        print("-" * 50)

    print("\n========================================\n")
