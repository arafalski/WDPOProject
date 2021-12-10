import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('data/09.jpg', cv2.IMREAD_COLOR)
# img_blur = cv2.medianBlur(img, 45)
img_blur = cv2.GaussianBlur(img, (55, 55), 0)

img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

lower = np.array([10, 110, 50])
upper = np.array([110, 255, 255])
mask1 = cv2.inRange(img_hsv, lower, upper)
mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, np.ones((51, 51), np.uint8))

lower2 = np.array([0, 30, 5])
upper2 = np.array([8, 255, 255])
mask2 = cv2.inRange(img_hsv, lower2, upper2)
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, np.ones((65, 65), np.uint8))

lower3 = np.array([150, 30, 50])
upper3 = np.array([255, 150, 255])
mask3 = cv2.inRange(img_hsv, lower3, upper3)

mask = mask1 / 255 + mask2 / 255 + mask3 / 255
mask[mask != 0] = 255

mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((65, 65), np.uint8))

img_color = img.copy()
B, G, R = img_color[:, :, 0], img_color[:, :, 1], img_color[:, :, 2]
B[mask == 0] = 0
G[mask == 0] = 0
R[mask == 0] = 0

H = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
S = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
V = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]

fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(10, 4))
ax11.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax11.set_title('Oryginalne zdjęcie')

ax12.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
ax12.set_title('Zdjęcie po przejściach')

fig1.tight_layout()


fig2, ((ax21, ax22, ax23),
       (ax24, ax25, ax26)) = plt.subplots(2, 3, figsize=(16, 8))

ax21.imshow(mask1, cmap='gray')
ax21.set_title('Maska - banany / pomarańcze')

ax22.imshow(mask2, cmap='gray')
ax22.set_title('Maska - jabłka')

ax23.imshow(mask3, cmap='gray')
ax23.set_title('Maska - refleksy')

ax24.imshow(mask, cmap='gray')
ax24.set_title('Suma masek')

fig2.tight_layout()


fig2, (axh, axs, axv) = plt.subplots(1, 3, figsize=(14, 4))
axh.imshow(H, cmap='gray')
axh.set_title('Kanał H')
axs.imshow(S, cmap='gray')
axs.set_title('Kanał S')
axv.imshow(V, cmap='gray')
axv.set_title('Kanał V')

plt.show()
