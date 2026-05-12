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
    quality = prob // 2 if prob < 7 else 7
    skize = tuple(quality for _ in range(2))
    return cv2.GaussianBlur(image, ksize=skize, sigmaX=0)


def salt_a_paper(image: numpy.ndarray, prob: float=0.5) -> numpy.ndarray:
    s_vs_p = 0.5  # Соотношение соли (белый) к перцу (черный)
    amount = min(prob / 10000, 0.3)
    out = np.copy(image)

    # Количество изменяемых пикселей
    num_pixels = int(amount * image.size)

    # Соль (белый шум)
    num_salt = int(num_pixels * s_vs_p)
    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    out[coords[0], coords[1]] = [255, 255, 255]  # Белый

    # Перец (черный шум)
    num_pepper = num_pixels - num_salt
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    out[coords[0], coords[1]] = [0, 0, 0]  # Черный

    return out


def color_salt_paper(image: np.ndarray, prob: float = 0.5) -> np.ndarray:
    s_vs_p = 0.5
    amount = prob / 10000 if prob <= 3 else 0.15
    out = np.copy(image)

    # image.size — это общее число элементов (H * W * 3)
    # image.size // 3 — число пикселей (H * W)
    num_pixels = int(np.ceil(amount * image.size // 3 * s_vs_p))

    # Генерируем случайные индексы пикселей (не элементов!)
    indices = np.random.choice(image.size // 3, size=num_pixels, replace=False)
    # Преобразуем плоские индексы в координаты (y, x)
    idx = np.unravel_index(indices, image.shape[:-1])

    # Генерируем случайные цвета для каждого пикселя: (num_pixels, 3)
    colors = np.random.randint(0, 256, size=(num_pixels, 3))

    # Присваиваем цвета по координатам
    out[idx] = colors

    return out


def quantize_image(image: numpy.ndarray, prob: float=0.5) -> numpy.ndarray:
    if prob <= 10:
        factor = (prob * 10) // 5
    else:
        factor = 100 // 5
    quantized_image = np.floor_divide(image, factor) * factor
    return quantized_image.astype(np.uint8)


def shot_noise(image: numpy.ndarray, prob: float=0.5) -> numpy.ndarray:
    rows, cols, _ = image.shape
    for x in range(rows):
        for y in range(cols):
            if random.random() < prob:
                image.astype(np.float64)[x, y] += random.gauss(0, 10)
    return image.clip(0, 255).astype(np.uint8)
