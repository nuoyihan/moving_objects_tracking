import cv2
import numpy as np

def enhance_contrast(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame

def apply_gaussian_blur(frame):
    return cv2.GaussianBlur(frame, (5, 5), 0)

def apply_morphological_operations(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

