import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class Fruit(Enum):
    apple = 1
    banana = 2
    orange = 3


def generate_mask(img_hsv):
    # bananas and oranges
    lower = np.array([10, 110, 50])
    upper = np.array([110, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower, upper)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE,
                             np.ones((51, 51), dtype=np.uint8))

    # apples
    lower2 = np.array([0, 30, 5])
    upper2 = np.array([8, 255, 255])
    mask2 = cv2.inRange(img_hsv, lower2, upper2)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE,
                             np.ones((65, 65), dtype=np.uint8))

    # glare on apples
    lower3 = np.array([150, 30, 50])
    upper3 = np.array([255, 150, 255])
    mask3 = cv2.inRange(img_hsv, lower3, upper3)

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((65, 65), dtype=np.uint8))

    return mask


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
    B, G, R = cv2.split(imgf32)
    B[mask == 0] = np.nan
    G[mask == 0] = np.nan
    R[mask == 0] = np.nan

    B_mean = int(np.nanmean(B))
    G_mean = int(np.nanmean(G))
    R_mean = int(np.nanmean(R))

    return np.array([[[B_mean, G_mean, R_mean]]], dtype=np.uint8)


def classify(h, s):
    if h > 20 and h < 120:
        return Fruit.banana

    if s > 190:
        return Fruit.orange

    return Fruit.apple


img = cv2.imread('data/00.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, dsize=(3000, 4000))
img_blur = cv2.GaussianBlur(img, (55, 55), 0)

img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

mask = generate_mask(img_hsv)

contours, _ = cv2.findContours(mask,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

rects = get_rects(contours)
fruits = []
for (r, c) in zip(rects, contours):
    y, x = img_blur.shape[0], img_blur.shape[1]
    rect_mask = np.zeros((y, x), dtype=np.uint8)
    cv2.fillPoly(rect_mask, [c], 255)
    img_rect = img_blur.copy()
    img_rect[rect_mask == 0] = 0
    img_rect = img_rect[r[0][1]:r[1][1] + 1, r[0][0]:r[1][0] + 1]

    mask_rect = mask[r[0][1]:r[1][1] + 1, r[0][0]:r[1][0] + 1]
    mean_color = get_mean_color(img_rect, mask_rect)

    mean_color_hsv = cv2.cvtColor(mean_color, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(mean_color_hsv)

    fruits.append(classify(h, s))
    # print(f'xmin = {r[0][0]}')
    # print(f'H = {H}')
    # print(f'S = {S}')
    # print(classify(H, S))
    # print()

print(f'Num of apples: {fruits.count(Fruit.apple)}')
print(f'Num of oranges: {fruits.count(Fruit.orange)}')
print(f'Num of bananas: {fruits.count(Fruit.banana)}')
################################################################
# images plotting
fig1, ax1 = plt.subplots()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Oryginalne zdjęcie')

fig1.tight_layout()


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
