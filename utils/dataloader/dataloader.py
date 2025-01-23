import os
import rasterio
import cv2
import numpy as np
import h5py
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File format to function mapping
FILE_READERS = {
    ".tiff": "read_tiff",
    ".tif": "read_tif",
    ".png": "read_png",
    ".jpg": "read_jpg",
    ".jpeg": "read_jpg", 
    ".h5": "read_h5",
}

def read_tiff(file_path):
    """Reads a .tiff file using rasterio."""
    try:
        logger.info(f"Reading .tiff file: {file_path}")
        with rasterio.open(file_path) as src:
            return src.read().astype(np.float32)
    except Exception as e:
        logger.exception(f"Error reading .tiff file at {file_path}: {e}")
        raise

def read_tif(file_path):
    """Reads a .tif file using rasterio."""
    try:
        logger.info(f"Reading .tif file: {file_path}")
        with rasterio.open(file_path) as src:
            return src.read(1).astype(np.uint8)
    except Exception as e:
        raise IOError(f"Error reading .tif file at {file_path}: {e}")

def read_png(file_path):
    """Reads a .png file using OpenCV."""
    try:
        logger.info(f"Reading .png file: {file_path}")
        return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        raise IOError(f"Error reading .png file at {file_path}: {e}")

def read_jpg(file_path):
    """Reads a .jpg or .jpeg file using OpenCV."""
    try:
        logger.info(f"Reading .jpg file: {file_path}")
        return cv2.imread(file_path, cv2.IMREAD_COLOR)
    except Exception as e:
        raise IOError(f"Error reading .jpg file at {file_path}: {e}")

def read_h5(file_path, key="data"):
    """Reads an .h5 file using h5py."""
    try:
        logger.info(f"Reading .h5 file: {file_path}")
        with h5py.File(file_path, "r") as f:
            if key not in f:
                raise KeyError(f"Key '{key}' not found in {file_path}")
            return np.array(f[key])
    except Exception as e:
        raise IOError(f"Error reading .h5 file at {file_path}: {e}")

def read_file(file_path):
    """Reads a file based on its extension using the appropriate function."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    file_format = os.path.splitext(file_path)[-1].lower()
    reader_func_name = FILE_READERS.get(file_format)

    if reader_func_name is None:
        logger.error(f"Unsupported file format: {file_format}")
        raise ValueError(f"Unsupported file format: {file_format}")

    try:
        reader_func = globals()[reader_func_name]
        return reader_func(file_path)
    except Exception as e:
        logger.exception(f"Error reading file {file_path}: {e}")
        raise





