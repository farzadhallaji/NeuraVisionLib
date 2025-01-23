from utils.general import set_seed
import numpy as np
import random
import cv2

def random_flip(data, seed):
    set_seed(seed)
    """Randomly flips the image horizontally or vertically."""
    if random.random() > 0.5:
        data = np.flip(data, axis=1)  # Horizontal flip
    if random.random() > 0.5:
        data = np.flip(data, axis=0)  # Vertical flip
    return data

def random_rotation(data, params, seed):
    set_seed(seed)
    """Randomly rotates the image within a specified angle."""
    angle = params.get("angle", 15)  # Default rotation angle
    random_angle = random.uniform(-angle, angle)
    center = (data.shape[1] // 2, data.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, random_angle, 1.0)
    data = cv2.warpAffine(data, rotation_matrix, (data.shape[1], data.shape[0]))
    return data

def color_jitter(data, params, seed):
    set_seed(seed)
    """Applies color jitter to adjust brightness, contrast, saturation, and hue."""
    brightness = params.get("brightness", 0.0)
    contrast = params.get("contrast", 0.0)
    saturation = params.get("saturation", 0.0)
    hue = params.get("hue", 0.0)

    # Apply brightness
    if brightness > 0:
        factor = 1 + random.uniform(-brightness, brightness)
        data = np.clip(data * factor, 0, 1)

    # Apply contrast
    if contrast > 0:
        factor = 1 + random.uniform(-contrast, contrast)
        mean = np.mean(data, axis=(0, 1), keepdims=True)
        data = np.clip((data - mean) * factor + mean, 0, 1)

    # Apply saturation (only for RGB images)
    if saturation > 0 and data.shape[-1] == 3:
        hsv = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1 + random.uniform(-saturation, saturation)), 0, 1)
        data = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Apply hue (only for RGB images)
    if hue > 0 and data.shape[-1] == 3:
        hsv = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-hue, hue)) % 1.0
        data = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return data