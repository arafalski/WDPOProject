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


def get_mean_color(img, mask):
    imgf32 = img.copy().astype(np.float32)
    B = imgf32[:, :, 0]
    G = imgf32[:, :, 1]
    R = imgf32[:, :, 2]
    B[mask == 0] = np.nan
    G[mask == 0] = np.nan
    R[mask == 0] = np.nan

    B_mean = np.nanmean(B)
    G_mean = np.nanmean(G)
    R_mean = np.nanmean(R)

    img[:, :] = np.array([B_mean, G_mean, R_mean])


def classify(h, s):
    if h > 20 and h < 120:
        return 'banana'

    if s > 190:
        return 'orange'

    return 'apple'


img = cv2.imread('data/09.jpg', cv2.IMREAD_COLOR)
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

contours, _ = cv2.findContours(mask.copy(),
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
print(f'Num of objects: {len(contours)}')

img_color = img.copy()
rects = get_rects(contours)
fruits = []
for (r, c) in zip(rects, contours):
    img_rect = img_color[r[0][1]:r[1][1] + 1, r[0][0]:r[1][0] + 1]
    img_rect_copy = img_rect.copy()

    for y in range(img_rect_copy.shape[0]):
        for x in range(img_rect_copy.shape[1]):
            if cv2.pointPolygonTest(c, (x + int(r[0][0]), y + int(r[0][1])), False) == -1:
                img_rect_copy[y, x] = 0

    mask_rect = mask[r[0][1]:r[1][1] + 1, r[0][0]:r[1][0] + 1]
    get_mean_color(img_rect_copy, mask_rect)

    img_rect_hsv = cv2.cvtColor(img_rect_copy, cv2.COLOR_BGR2HSV)
    H = int(np.mean(img_rect_hsv[:, :, 0]))
    S = int(np.mean(img_rect_hsv[:, :, 1]))
    fruits.append(classify(H, S))
    print(f'xmin = {r[0][0]}')
    print(f'H = {H}')
    print(f'S = {S}')
    print(classify(H, S))
    print()

fruits.sort()
print(f'Fruits: {fruits}')
################################################################
# images plotting
fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(10, 4))
ax11.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax11.set_title('Oryginalne zdjęcie')

# ax12.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
# ax12.set_title('Zdjęcie po przejściach')

fig1.tight_layout()


# fig2, ((ax21, ax22, ax23),
#        (ax24, ax25, ax26)) = plt.subplots(2, 3, figsize=(16, 8))

# ax21.imshow(mask1, cmap='gray')
# ax21.set_title('Maska - banany / pomarańcze')

# ax22.imshow(mask2, cmap='gray')
# ax22.set_title('Maska - jabłka')

# ax23.imshow(mask3, cmap='gray')
# ax23.set_title('Maska - refleksy')

# ax24.imshow(mask, cmap='gray')
# ax24.set_title('Suma masek')

# fig2.tight_layout()


# H = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)[:, :, 0]
# S = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)[:, :, 1]
# V = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)[:, :, 2]

# fig3, (axh, axs, axv) = plt.subplots(1, 3, figsize=(14, 4))
# axh.imshow(H, cmap='gray')
# axh.set_title('Kanał H')
# axs.imshow(S, cmap='gray')
# axs.set_title('Kanał S')
# axv.imshow(V, cmap='gray')
# axv.set_title('Kanał V')

# fig3.tight_layout()

plt.show()
