import cv2
import json
import click
import numpy as np

from glob import glob
from tqdm import tqdm

from enum import Enum
from typing import Dict


class Fruit(Enum):
    apple = 1
    banana = 2
    orange = 3


def generate_mask(img_hsv: np.ndarray) -> np.ndarray:
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


def detect_fruits(img_path: str) -> Dict[str, int]:
    """Fruit detection function, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each fruit.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
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
        if cv2.contourArea(c) < 20000:
            continue
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

    apple = fruits.count(Fruit.apple)
    banana = fruits.count(Fruit.banana)
    orange = fruits.count(Fruit.orange)

    return {'apple': apple, 'banana': banana, 'orange': orange}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory')
@click.option('-o', '--output_file_path', help='Path to output file')
def main(data_path, output_file_path):

    img_list = glob(f'{data_path}/*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(img_path)

        filename = img_path.split('/')[-1]

        results[filename] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
