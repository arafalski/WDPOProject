import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_rects(contours):
    rects = []
    for cont in contours:
        x = [p[0][0] for p in cont]
        y = [p[0][1] for p in cont]
        rects.append([(np.min(x), np.min(y)),
                      (np.max(x), np.max(y))])
    return rects


img = cv2.imread('data/03.jpg', cv2.IMREAD_COLOR)
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

mask = cv2.morphologyEx(mask.astype(np.uint8),
                        cv2.MORPH_CLOSE, np.ones((65, 65), np.uint8))

H = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)[:, :, 0]
S = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)[:, :, 1]
V = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)[:, :, 2]

contours, hierarchy = cv2.findContours(mask.copy(),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
print(f'Num of objects: {len(contours)}')

img_color = img.copy()
cv2.drawContours(img_color, contours, -1, (0, 255, 0), 10)

rects = get_rects(contours)
for r in rects:
    cv2.rectangle(img_color, r[0], r[1], (0, 0, 255), 10)

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
