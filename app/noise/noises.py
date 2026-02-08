import random

import cv2
import numpy
import numpy as np


def pixelated(image: numpy.ndarray, prob: float=0.5)-> numpy.ndarray:
    scale_factor: float = prob * 2

    # Получим размеры уменьшенного изображения
    img_with, img_heigh, _ = image.shape
    down_width = img_with // scale_factor
    down_height = img_heigh // scale_factor

    # Уменьшили изображение
    small_img = cv2.resize(image, (down_width, down_height))

    # Увеличили обратно с сохранением квадратов пикселей
    pixelated_img = cv2.resize(small_img, (img_with, img_heigh), interpolation=cv2.INTER_NEAREST)

    return pixelated_img

def gaus_noise(image: numpy.ndarray, prob: float=0.5) -> numpy.ndarray:
    quality: int = int(10 * prob)
    quality = 3 if quality < 3 else 25 if quality > 25 else quality
    skize = tuple(quality for _ in range(2))
    return cv2.GaussianBlur(image, ksize=skize, sigmaX=0)


def salt_a_paper(image: numpy.ndarray, prob: float=0.5) -> numpy.ndarray:
    row, col = image.shape[:2]
    num_salt = int(np.ceil(row * col * (prob / 100)))
    num_pepper = int(np.ceil(row * col * (prob / 100)))
    coords = [np.random.randint(0, i - 1, size=num_salt + num_pepper) for i in [row, col]]
    image[coords[0][:num_salt], coords[1][:num_salt]] = 255
    image[coords[0][num_salt:], coords[1][num_salt:]] = 0
    return image


def color_salt_paper(image: numpy.ndarray, prob: float=0.5) -> numpy.ndarray:
    s_vs_p = 0.5
    amount = prob / 10
    out = np.copy(image)
    num_colors = np.ceil(amount * image.size * s_vs_p)
    colors = np.random.randint(low=0, high=256, size=(int(num_colors), 3))
    indices = np.random.choice(image.size // 3, size=int(num_colors), replace=False)
    idx = np.unravel_index(indices, image.shape[:-1])
    out[idx] = colors
    return out


def quantize_image(image: numpy.ndarray, prob: float=0.5) -> numpy.ndarray:
    factor = prob * 10
    quantized_image = np.floor_divide(image, factor) * factor
    return quantized_image.astype(np.uint8)


def shot_noise(image: numpy.ndarray, prob: float=0.5) -> numpy.ndarray:
    rows, cols, _ = image.shape
    for x in range(rows):
        for y in range(cols):
            if random.random() < prob:
                image.astype(np.float64)[x, y] += random.gauss(0, 10)
    return image.clip(0, 255).astype(np.uint8)
