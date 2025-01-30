import os
import rasterio
import cv2
import numpy as np
import h5py
import json
import pandas as pd
import logging
import inspect

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
    ".csv": "read_csv",
    ".json": "read_json",
    ".txt": "read_txt",
    ".npy": "read_npy",
    ".npz": "read_npz"
}

# --------------------- #
# FILE READER FUNCTIONS #
# --------------------- #

def read_tiff(file_path, band=None):
    """Reads a .tiff file using rasterio."""
    try:
        logger.info(f"Reading .tiff file: {file_path}")
        with rasterio.open(file_path) as src:
            return src.read() if band is None else src.read(band).astype(np.float32)
    except Exception as e:
        logger.exception(f"Error reading .tiff file at {file_path}: {e}")
        raise IOError(f"Error reading .tiff file at {file_path}: {e}")

def read_tif(file_path, band=None):
    """Reads a .tif file using rasterio."""
    return read_tiff(file_path, band=band if band else 1)

def read_png(file_path):
    """Reads a .png file using OpenCV."""
    try:
        logger.info(f"Reading .png file: {file_path}")
        return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        logger.exception(f"Error reading .png file: {e}")
        raise IOError(f"Error reading .png file: {e}")

def read_jpg(file_path):
    """Reads a .jpg or .jpeg file using OpenCV."""
    try:
        logger.info(f"Reading .jpg file: {file_path}")
        return cv2.imread(file_path, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.exception(f"Error reading .jpg file: {e}")
        raise IOError(f"Error reading .jpg file: {e}")

def read_h5(file_path, key=None):
    """Reads an .h5 file using h5py."""
    try:
        logger.info(f"Reading .h5 file: {file_path}")
        with h5py.File(file_path, "r") as f:
            available_keys = list(f.keys())
            if not available_keys:
                raise IOError(f"No datasets found in {file_path}")

            selected_key = key if key and key in available_keys else available_keys[0]
            logger.info(f"Using dataset key: {selected_key}")
            return np.array(f[selected_key])
    except Exception as e:
        logger.exception(f"Error reading .h5 file: {e}")
        raise IOError(f"Error reading .h5 file: {e}")

def read_csv(file_path, delimiter=",", **kwargs):
    """Reads a .csv file using pandas."""
    try:
        logger.info(f"Reading .csv file: {file_path}")
        return pd.read_csv(file_path, delimiter=delimiter, **kwargs)
    except Exception as e:
        logger.exception(f"Error reading .csv file: {e}")
        raise IOError(f"Error reading .csv file: {e}")

def read_json(file_path, encoding="utf-8", **kwargs):
    """Reads a .json file."""
    try:
        logger.info(f"Reading .json file: {file_path}")
        with open(file_path, "r", encoding=encoding) as f:
            return json.load(f)
    except Exception as e:
        logger.exception(f"Error reading .json file: {e}")
        raise IOError(f"Error reading .json file: {e}")

def read_txt(file_path, encoding="utf-8", **kwargs):
    """Reads a .txt file."""
    try:
        logger.info(f"Reading .txt file: {file_path}")
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.exception(f"Error reading .txt file: {e}")
        raise IOError(f"Error reading .txt file: {e}")

def read_npy(file_path):
    """Reads a .npy file using numpy."""
    try:
        logger.info(f"Reading .npy file: {file_path}")
        return np.load(file_path)
    except Exception as e:
        logger.exception(f"Error reading .npy file: {e}")
        raise IOError(f"Error reading .npy file: {e}")

def read_npz(file_path):
    """Reads a .npz file using numpy."""
    try:
        logger.info(f"Reading .npz file: {file_path}")
        return dict(np.load(file_path))
    except Exception as e:
        logger.exception(f"Error reading .npz file: {e}")
        raise IOError(f"Error reading .npz file: {e}")

# ------------------------- #
# DYNAMIC FILE READING LOGIC #
# ------------------------- #

def get_reader_function_args(file_format):
    """
    Retrieves the arguments (excluding file_path) for a given file reader function.

    :param file_format: The file extension (e.g., ".csv", ".json").
    :return: Dictionary of function argument names and default values.
    :raises: ValueError if the file format is unsupported.
    """
    reader_func_name = FILE_READERS.get(file_format.lower())

    if reader_func_name is None:
        logger.error(f"Unsupported file format: {file_format}")
        raise ValueError(f"Unsupported file format: {file_format}")

    reader_func = globals().get(reader_func_name)

    if reader_func is None:
        logger.error(f"Reader function '{reader_func_name}' not found.")
        raise ValueError(f"Reader function '{reader_func_name}' is not defined.")

    # Get function signature (excluding "file_path")
    signature = inspect.signature(reader_func)
    arg_info = {
        param.name: param.default if param.default is not inspect.Parameter.empty else None
        for param in signature.parameters.values()
        if param.name != "file_path"
    }
    return arg_info

def read_file(file_path, **kwargs):
    """
    Reads a file dynamically using the correct reader function.

    :param file_path: Path to the file.
    :param kwargs: Additional arguments for the reader function.
    :return: Data read from the file.
    :raises: FileNotFoundError, ValueError, IOError.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    file_format = os.path.splitext(file_path)[-1].lower()
    reader_func_name = FILE_READERS.get(file_format)

    if reader_func_name is None:
        logger.error(f"Unsupported file format: {file_format}")
        raise ValueError(f"Unsupported file format: {file_format}")

    try:
        reader_func = globals().get(reader_func_name)
        if reader_func is None:
            logger.error(f"Reader function '{reader_func_name}' not found.")
            raise ValueError(f"Reader function '{reader_func_name}' is not defined.")

        # Get expected arguments dynamically
        expected_args = get_reader_function_args(file_format)

        # Filter kwargs to only include relevant arguments
        valid_kwargs = {k: v for k, v in kwargs.items() if k in expected_args}

        return reader_func(file_path, **valid_kwargs)

    except Exception as e:
        logger.exception(f"Error reading file {file_path}: {e}")
        raise
