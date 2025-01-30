import os
import rasterio
import cv2
import numpy as np
import h5py
import json
import pandas as pd
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
    ".csv": "read_csv",
    ".json": "read_json",
    ".txt": "read_txt",
    ".npy": "read_npy",
    ".npz": "read_npz"
}

def read_tiff(file_path):
    """
    Reads a .tiff file using rasterio.

    :param file_path: Path to the .tiff file
    :return: Numpy array of the file's contents
    :raises: IOError if the file cannot be read
    """
    try:
        logger.info(f"Reading .tiff file: {file_path}")
        with rasterio.open(file_path) as src:
            return src.read().astype(np.float32)
    except Exception as e:
        logger.exception(f"Error reading .tiff file at {file_path}: {e}")
        raise IOError(f"Error reading .tiff file at {file_path}: {e}")

def read_tif(file_path):
    """
    Reads a .tif file using rasterio.

    :param file_path: Path to the .tif file
    :return: Numpy array of the file's contents
    :raises: IOError if the file cannot be read
    """
    try:
        logger.info(f"Reading .tif file: {file_path}")
        with rasterio.open(file_path) as src:
            return src.read(1).astype(np.uint8)
    except Exception as e:
        logger.exception(f"Error reading .tif file at {file_path}: {e}")
        raise IOError(f"Error reading .tif file at {file_path}: {e}")

def read_png(file_path):
    """
    Reads a .png file using OpenCV.

    :param file_path: Path to the .png file
    :return: Numpy array of the file's contents
    :raises: IOError if the file cannot be read
    """
    try:
        logger.info(f"Reading .png file: {file_path}")
        return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        logger.exception(f"Error reading .png file at {file_path}: {e}")
        raise IOError(f"Error reading .png file at {file_path}: {e}")

def read_jpg(file_path):
    """
    Reads a .jpg or .jpeg file using OpenCV.

    :param file_path: Path to the .jpg or .jpeg file
    :return: Numpy array of the file's contents
    :raises: IOError if the file cannot be read
    """
    try:
        logger.info(f"Reading .jpg file: {file_path}")
        return cv2.imread(file_path, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.exception(f"Error reading .jpg file at {file_path}: {e}")
        raise IOError(f"Error reading .jpg file at {file_path}: {e}")

def read_h5(file_path, key="data"):
    """
    Reads an .h5 file using h5py.

    :param file_path: Path to the .h5 file
    :param key: Dataset key in the HDF5 file to read
    :return: Numpy array of the data under the specified key
    :raises: IOError if the file cannot be read or the key is missing
    """
    try:
        logger.info(f"Reading .h5 file: {file_path}")
        with h5py.File(file_path, "r") as f:
            if key not in f:
                logger.error(f"Key '{key}' not found in {file_path}")
                raise KeyError(f"Key '{key}' not found in {file_path}")
            return np.array(f[key])
    except Exception as e:
        logger.exception(f"Error reading .h5 file at {file_path}: {e}")
        raise IOError(f"Error reading .h5 file at {file_path}: {e}")

def read_csv(file_path, **kwargs):
    """
    Reads a .csv file using pandas.

    :param file_path: Path to the .csv file
    :param kwargs: Additional arguments for pandas.read_csv
    :return: Pandas DataFrame
    :raises: IOError if the file cannot be read
    """
    try:
        logger.info(f"Reading .csv file: {file_path}")
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        logger.exception(f"Error reading .csv file at {file_path}: {e}")
        raise IOError(f"Error reading .csv file at {file_path}: {e}")

def read_json(file_path, **kwargs):
    """
    Reads a .json file using the built-in json module.

    :param file_path: Path to the .json file
    :param kwargs: Additional arguments for json.load
    :return: Parsed JSON data
    :raises: IOError if the file cannot be read
    """
    try:
        logger.info(f"Reading .json file: {file_path}")
        with open(file_path, "r", **kwargs) as f:
            return json.load(f)
    except Exception as e:
        logger.exception(f"Error reading .json file at {file_path}: {e}")
        raise IOError(f"Error reading .json file at {file_path}: {e}")

def read_txt(file_path, **kwargs):
    """
    Reads a .txt file using the built-in open function.

    :param file_path: Path to the .txt file
    :param kwargs: Additional arguments for open
    :return: File contents as a string
    :raises: IOError if the file cannot be read
    """
    try:
        logger.info(f"Reading .txt file: {file_path}")
        with open(file_path, "r", **kwargs) as f:
            return f.read()
    except Exception as e:
        logger.exception(f"Error reading .txt file at {file_path}: {e}")
        raise IOError(f"Error reading .txt file at {file_path}: {e}")

def read_npy(file_path):
    """
    Reads a .npy file using numpy.

    :param file_path: Path to the .npy file
    :return: Numpy array
    :raises: IOError if the file cannot be read
    """
    try:
        logger.info(f"Reading .npy file: {file_path}")
        return np.load(file_path)
    except Exception as e:
        logger.exception(f"Error reading .npy file at {file_path}: {e}")
        raise IOError(f"Error reading .npy file at {file_path}: {e}")

def read_npz(file_path):
    """
    Reads a .npz file using numpy.

    :param file_path: Path to the .npz file
    :return: Dictionary of arrays contained in the file
    :raises: IOError if the file cannot be read
    """
    try:
        logger.info(f"Reading .npz file: {file_path}")
        return dict(np.load(file_path))
    except Exception as e:
        logger.exception(f"Error reading .npz file at {file_path}: {e}")
        raise IOError(f"Error reading .npz file at {file_path}: {e}")

def read_file(file_path, **kwargs):
    """
    Reads a file based on its extension using the appropriate function.

    :param file_path: Path to the file to be read
    :param kwargs: Additional arguments to pass to the reader function
    :return: Data read from the file
    :raises FileNotFoundError: If the file does not exist
    :raises ValueError: If the file format is unsupported
    :raises IOError: If the file cannot be read
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

        # Pass kwargs dynamically to the reader function
        return reader_func(file_path, **kwargs)
    except Exception as e:
        logger.exception(f"Error reading file {file_path}: {e}")
        raise
