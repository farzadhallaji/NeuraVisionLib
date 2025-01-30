import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from utils.dataloader.file_reader import read_file
from utils.general import load_config, set_seed, load_module_function_from_config
from sklearn.model_selection import train_test_split
from utils.dataloader.operation import OPERATION_MAP  
import matplotlib.pyplot as plt


SPLIT_KEYS = ['train', 'val', 'test']


class ConfigurableDataset(Dataset):
    def __init__(self, config, split):
        self.split_name = split
        self.config = config

        self.dataset_name = config["dataset"]["name"]
        self.dataset_path = config["dataset"]["path"]

        self.read_data_file = config["dataset"]["read_data_file"]
        self.split = config["dataset"]["split"]

        self.preprocessing = config.get("preprocessing", {})
        self.augmentation = config.get("augmentation", {})
        
        self.read_data_file = config.get("read_data_file", {})
        if self.read_data_file.get("custom_read_file", {}).get("enabled", False):
            f = self.read_data_file.get("custom_read_file", {}).get("function", {})
            self.read_file = load_module_function_from_config(f)
        else:
            self.read_file = read_file

        self._initialize_split()
        
        self.samples = self._load_samples()

        # self._validate_config()



    def _initialize_split(self):
        self.split_ratio = {}
        self.split_dir = {}
        self.data_dir = {}

        for sp in SPLIT_KEYS:
            d_split = self.split[sp]
            if isinstance(d_split, str):
                # Case: Split directory specified as a relative path
                self.split_dir[sp] = d_split
                self.split_ratio[sp] = 1.0
                self.data_dir[sp] = os.path.join(self.dataset_path, d_split)
            elif isinstance(d_split, float):
                # Case: Split ratio provided (data is in a single folder)
                self.split_dir[sp] = self.dataset_path
                self.data_dir[sp] = self.dataset_path
                self.split_ratio[sp] = d_split
            else:
                raise ValueError(f"Invalid split configuration for '{sp}': {d_split}")

        # Check if specified directories exist
        for sp, dir_path in self.data_dir.items():
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Dataset split directory does not exist: {dir_path}")

        # Load additional split configuration options
        self.split_use_seed = self.split.get("use_seed", False)
        self.split_seed = self.split.get("seed", None)
        self.split_shuffle = self.split.get("shuffle", False)
        self.split_stratify = self.split.get("shuffle", None)
