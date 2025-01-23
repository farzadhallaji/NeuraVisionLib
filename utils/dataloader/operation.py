import numpy as np
import random
import cv2
from utils.general import set_seed


def parse_operation_config(operation, required_keys=None, defaults=None):
    params = operation.get("parameters", {})
    if required_keys:
        for key in required_keys:
            if key not in params:
                raise ValueError(f"Missing required parameter '{key}' for operation '{operation['name']}'.")

    if defaults:
        for key, value in defaults.items():
            params.setdefault(key, value)

    return params


def resize(data, operation):
    """Resize data to specified dimensions."""
    params = parse_operation_config(operation, required_keys=["width", "height"])
    width, height = params["width"], params["height"]
    return cv2.resize(data, (width, height), interpolation=cv2.INTER_LINEAR)



def normalize(data, operation):
    """Normalize data using specified method (zscore or minmax)."""
    defaults = {"method": "zscore", "range": [0, 1], "mean": 0, "std": 1}
    params = parse_operation_config(operation, defaults=defaults)

    if params["method"] == "zscore":
        mean, std = np.array(params["mean"], dtype=np.float32), np.array(params["std"], dtype=np.float32)
        return (data - mean) / std
    elif params["method"] == "minmax":
        min_val, max_val = params["range"]
        return (data - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unsupported normalization method: {params['method']}")



def binarize(data, operation):
    """Binarize data based on a threshold."""
    threshold = operation["parameters"]["threshold"]
    return (data > threshold).astype(np.float32)


def sliding_window(data, window_size, stride):
    """Helper function to generate sliding window patches."""
    patches = []
    h, w = data.shape
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            patch = data[y:y + window_size, x:x + window_size]
            patches.append(patch)
    return np.stack(patches)


def apply_sliding_window(data, operation):
    """Extract patches from data using sliding window."""
    params = operation["parameters"]
    window_size = params["window_size"]
    stride = params["stride"]
    return sliding_window(data, window_size, stride)


def random_flip(data, operation, seed=None):
    """Randomly flips the image horizontally or vertically."""
    if seed is not None:
        set_seed(seed)

    params = parse_operation_config(operation)  # No required params for random_flip
    if random.random() > 0.5:
        data = np.flip(data, axis=1)  # Horizontal flip
    if random.random() > 0.5:
        data = np.flip(data, axis=0)  # Vertical flip
    return data


def random_rotation(data, operation, seed=None):
    """Randomly rotates the image within a specified angle."""
    if seed is not None:
        set_seed(seed)
    params = operation["parameters"]
    angle = params.get("angle", 15)  # Default rotation angle
    random_angle = random.uniform(-angle, angle)
    center = (data.shape[1] // 2, data.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, random_angle, 1.0)
    return cv2.warpAffine(data, rotation_matrix, (data.shape[1], data.shape[0]))


def color_jitter(data, operation, seed=None):
    """Applies color jitter to adjust brightness, contrast, saturation, and hue."""
    if seed is not None:
        set_seed(seed)
    params = operation["parameters"]

    # Retrieve parameters
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


# Map of operations for easier dynamic calls
OPERATION_MAP = {
    "resize": resize,
    "normalization": normalize,
    "binarization": binarize,
    "sliding_window": apply_sliding_window,
    "random_flip": random_flip,
    "random_rotation": random_rotation,
    "color_jitter": color_jitter,
}
