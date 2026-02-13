import cv2
import numpy
import numpy as np


def adjust_brightness(image: numpy.ndarray, percent: float = 3) -> numpy.ndarray:
    percent_change = np.random.uniform(-percent, percent)
    value = int(percent_change * 255)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(int) + value, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_rgb_channels(image: numpy.ndarray, percent: float = 3) -> numpy.ndarray:
    b_percent = np.random.uniform(-percent, percent)
    g_percent = np.random.uniform(-percent, percent)
    r_percent = np.random.uniform(-percent, percent)

    b, g, r = cv2.split(image)

    b_adjusted = np.clip(b.astype(float) * (1 + b_percent / 10), 0, 255).astype(np.uint8)
    g_adjusted = np.clip(g.astype(float) * (1 + g_percent / 10), 0, 255).astype(np.uint8)
    r_adjusted = np.clip(r.astype(float) * (1 + r_percent / 10), 0, 255).astype(np.uint8)
    return cv2.merge([b_adjusted, g_adjusted, r_adjusted])


def adjust_contrast(image: numpy.ndarray, percent: float = 3) -> numpy.ndarray:
    percent_change = np.random.uniform(-percent, percent)
    factor = 1 + (percent_change / 10)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(float) * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def enhance_sharpness(image: numpy.ndarray, percent: float = 70) -> numpy.ndarray:
    percent = np.random.uniform(0, percent)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    enhanced = cv2.filter2D(image, -1, kernel)
    return cv2.addWeighted(image, 1 - percent / 10, enhanced, percent / 10, 0)
