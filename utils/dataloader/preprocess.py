import cv2
import numpy as np

def resize(data, operation):
    """Resize data to specified dimensions."""
    params = operation["parameters"]
    width = params["width"]
    height = params["height"]
    return cv2.resize(data, (width, height), interpolation=cv2.INTER_LINEAR)


def normalize(data, operation):
    """Normalize data using specified method (zscore or minmax)."""
    params = operation["parameters"]
    method = params.get("method", "zscore")
    if method == "zscore":
        mean = np.array(params["mean"], dtype=np.float32)
        std = np.array(params["std"], dtype=np.float32)
        return (data - mean) / std
    elif method == "minmax":
        min_val, max_val = params["range"]
        return (data - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def binarize(data, operation):
    """Binarize data based on a threshold."""
    threshold = operation["parameters"]["threshold"]
    return (data > threshold).astype(np.float32)

def sliding_window(data, window_size, stride):
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