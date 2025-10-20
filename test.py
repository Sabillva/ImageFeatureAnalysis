import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


# foto = cv2.imread("./normal1.jpg")

# def foto_negatifi(foto):
#     L = np.max(foto)
#     negatif_foto = L - foto
#     return negatif_foto

foto = cv2.imread("./normal10.jpg", 0)

# negative_foto = foto_negatifi(foto)

# yan_yana = np.hstack((foto, negative_foto))

print(foto.shape)

# cv2.imshow("fotograf", foto)
# cv2.waitKey(0)
# cv2.destroyAllWindows

# x = 775
# y = 275
# mavi_kanal = 0
# yesil_kanal = 1
# kirmizi_kanal = 2

# mavi_yogunluk = foto[x, y, mavi_kanal]
# yesil_yogunluk = foto[x, y, yesil_kanal]
# kirmizi_yogunluk = foto[x, y, kirmizi_kanal]

# print("mavi yogunluk:", mavi_yogunluk)
# print("yesil yogunluk:", yesil_yogunluk)
# print("kirmizi yogunluk:", kirmizi_yogunluk)

# minimum_yogunluk = np.min(foto)
# maximum_yogunluk = np.max(foto)

# print("Maximum yogunluk:", maximum_yogunluk)
# print("Minimum yogunluk", minimum_yogunluk)

# yogunluk = foto[x, y]
# print("yogunluk:", yogunluk)

crop = foto[700:100, 500:1000]
print(crop.shape)

# crop = foto[10:15, 10:15, 1]
# print(crop)

# kirmizi_kanali = foto[:, :, 2]

cv2.imshow("fotograf", crop)
# plt.imshow(foto, cmap="gray")
# plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows

# plt.imshow(yan_yana, cmap="gray")
# plt.show()