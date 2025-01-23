import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.dataloader.file_reader import read_file
from utils.general import load_config, set_seed
from sklearn.model_selection import train_test_split
import torch
import random
from utils.dataloader.operation import OPERATION_MAP  
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




SPLIT_KEYS = ['train', 'val', 'test']


class BaseDataset(Dataset):
    def __init__(self, config, split):
        self.split_name = split
        self.config = config

        self.dataset_name = config["dataset"]["name"]
        self.dataset_path = config["dataset"]["path"]

        self.read_data_file = config["dataset"]["read_data_file"]
        self.split = config["dataset"]["split"]

        self.preprocessing = config.get("preprocessing", {})
        self.augmentation = config.get("augmentation", {})
        
        if self.read_data_file.get("custom_read_file", False):
            self.read_file = self.custom_read_file
        else:
            self.read_file = read_file
        self.read_file_kargs = self.read_data_file.get("args",{})

        self._initialize_split()
        
        self.samples = self._load_samples()

        self._validate_config()

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

    def _get_all_files(self):
        if not os.path.exists(self.split_dir[self.split_name]):
            raise FileNotFoundError(f"Dataset directory does not exist: {self.split_dir[self.split_name]}")
        
        inputs = sorted(
            [os.path.join(self.split_dir[self.split_name], f) for f in os.listdir(self.split_dir[self.split_name]) if f.endswith(self.read_data_file["file_format"]["input"])]
        )
        targets = sorted(
            [os.path.join(self.split_dir[self.split_name], f) for f in os.listdir(self.split_dir[self.split_name]) if f.endswith(self.read_data_file["file_format"]["target"])]
        )

        if len(inputs) != len(targets):
            raise ValueError("Mismatch between input and target files.")

        return inputs, targets


    def _apply_split(self, inputs, targets):
        """Apply ordered splitting for train, val, and test sets."""
        if self.split_use_seed:
            set_seed(self.split_seed)

        # Check if a custom split function is provided in the config
        if self.split.get("custom_train_test_split", False):
            # Call the user-defined custom train-test split function
            return self.custom_train_test_split(inputs, targets, split_name=self.split_name, config=self.split)

        # Default splitting logic
        train_ratio = self.split_ratio.get("train", 0.0)
        val_ratio = self.split_ratio.get("val", 0.0)
        test_ratio = self.split_ratio.get("test", 0.0)

        if train_ratio + val_ratio + test_ratio > 1.0:
            raise ValueError("Split ratios for train, val, and test must sum to 1.0 or less.")

        # Compute adjusted test size
        test_size = val_ratio + test_ratio
        val_size = val_ratio

        # Use the default split function
        train_data, val_data, test_data = self._split_data(
            inputs, targets,
            test_size=test_size,
            val_size=val_size,
            random_state=self.split_seed,
            shuffle=self.split_shuffle,
            stratify=self.split_stratify
        )

        # Return the appropriate split
        if self.split_name == "train":
            return train_data
        elif self.split_name == "val":
            return val_data
        elif self.split_name == "test":
            return test_data
        else:
            raise ValueError(f"Unknown split type: {self.split_name}")


    def _split_data(self, inputs, targets, test_size, val_size=0.0, random_state=None, shuffle=True, stratify=None):
        """
        Wrapper for splitting data into train, val, and test sets with optional stratification.
        """
        def split(inputs, targets, test_size, stratify):
            return train_test_split(inputs, targets, test_size=test_size, random_state=random_state, stratify=stratify)

        if shuffle:
            combined = list(zip(inputs, targets))
            random.seed(random_state)
            random.shuffle(combined)
            inputs, targets = zip(*combined)

        if val_size > 0:
            train_val_inputs, test_inputs, train_val_targets, test_targets = split(inputs, targets, test_size, stratify)
            val_ratio_adjusted = val_size / (1.0 - test_size)
            train_inputs, val_inputs, train_targets, val_targets = split(train_val_inputs, train_val_targets, val_ratio_adjusted, stratify)
        else:
            train_inputs, test_inputs, train_targets, test_targets = split(inputs, targets, test_size, stratify)
            val_inputs, val_targets = [], []

        return (list(zip(train_inputs, train_targets)),
                list(zip(val_inputs, val_targets)),
                list(zip(test_inputs, test_targets)))


    def _load_samples(self):
        inputs, targets = self._get_all_files()
        samples = self._apply_split(inputs, targets)
        if not samples:
            raise ValueError(f"No samples found for split: {self.split_name}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        input_path, target_path = self.samples[index]
        input_data = self.read_file(input_path, **self.read_file_kargs)
        target_data = self.read_file(target_path, **self.read_file_kargs)

        for operation in self.preprocessing.get("operations", []) + self.augmentation.get("operations", []):
            if operation["enabled"]:
                if operation["apply_to"] == "input":
                    input_data = self._apply_operation(input_data, operation)
                elif operation["apply_to"] == "target":
                    target_data = self._apply_operation(target_data, operation)
                elif operation["apply_to"] == "both":
                    input_data = self._apply_operation(input_data, operation)
                    target_data = self._apply_operation(target_data, operation)
        
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.float32)

        return input_tensor, target_tensor
    
    def _apply_operation(self, data, operation):
        operation_name = operation["name"]
        operation_func = OPERATION_MAP.get(operation_name)
        if operation_func:
            return operation_func(data, operation)
        elif operation_name.startswith("custom"):
            custom_method = getattr(self, f"_{operation_name}", None)
            if custom_method:
                return custom_method(data, operation)
            raise ValueError(f"Custom operation '{operation_name}' is not implemented.")
        else:
            raise ValueError(f"Unsupported operation: {operation_name}")
    
    def custom_read_file(self, file_path, **kwargs):
        print("Custom read file function called with the following parameters:")
        print(f"File Path: {file_path}")
        for key, value in kwargs.items():
            print(f"{key}: {value}")
        raise NotImplementedError("custom_read_file function is not implemented. Please define your custom logic.")
    
    def custom_train_test_split(self, inputs, targets, split_name, config):
        """
        Custom train-test split logic defined by the user.

        Args:
            inputs (list): List of input file paths.
            targets (list): List of target file paths.
            split_name (str): The current split ("train", "val", or "test").
            config (dict): The dataset configuration dictionary.

        Returns:
            list: List of tuples containing input-target pairs for the specified split.
        """
        print("Custom split logic called!")
        print(f"Split Name: {split_name}")
        print(f"Config: {config}")

        raise NotImplementedError(
            f"A custom split logic is enabled but not implemented. "
            f"Override the `custom_train_test_split` method in your dataset class."
        )


    def custom_preprocess_input(self, **kwargs):
        raise NotImplementedError("custom_preprocess_input function is not implemented. Please define your custom logic.")

    def custom_preprocess_target(self, **kwargs):
        raise NotImplementedError("custom_preprocess_target function is not implemented. Please define your custom logic.")

    def custom_preprocess_both(self, **kwargs):
        raise NotImplementedError("custom_preprocess_both function is not implemented. Please define your custom logic.")


    def custom_augmentations_input(self, **kwargs):
        raise NotImplementedError("custom_augmentations_input function is not implemented. Please define your custom logic.")

    def custom_augmentations_target(self, **kwargs):
        raise NotImplementedError("custom_augmentations_target function is not implemented. Please define your custom logic.")

    def custom_augmentations_both(self, **kwargs):
        raise NotImplementedError("custom_augmentations_both function is not implemented. Please define your custom logic.")
    
    def _validate_config(self):
        """Validate the dataset configuration."""
        required_keys = ["dataset", "split", "read_data_file"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Validate dataset path
        if not os.path.exists(self.config["dataset"]["path"]):
            raise FileNotFoundError(f"Dataset path does not exist: {self.config['dataset']['path']}")

        # Validate read_data_file
        read_file_config = self.config["dataset"]["read_data_file"]
        if "file_format" not in read_file_config:
            raise ValueError("Missing 'file_format' in read_data_file configuration.")
        if not isinstance(read_file_config["file_format"], dict):
            raise ValueError("'file_format' must be a dictionary.")


    def visualize_sample(self, index=None, save_path=None):
        """Visualize a sample from the dataset."""
        if index is None:
            index = random.randint(0, len(self.samples) - 1)
        input_path, target_path = self.samples[index]
        input_data = self.read_file(input_path, **self.read_file_kargs)
        target_data = self.read_file(target_path, **self.read_file_kargs)
        
        plt.subplot(1, 2, 1)
        plt.title("Input")
        plt.imshow(input_data, cmap="gray" if len(input_data.shape) == 2 else None)
        plt.subplot(1, 2, 2)
        plt.title("Target")
        plt.imshow(target_data, cmap="gray" if len(target_data.shape) == 2 else None)
        if save_path:
            plt.savefig(save_path)
        plt.show()



class DataLoaderFactory:
    def __init__(self, config_path):
        self.config = load_config(config_path)

    def get_all_dataloaders(self):
        """Get DataLoader objects for all splits (train, val, test)."""
        loaders = {}
        for split in SPLIT_KEYS:
            dataset = BaseDataset(self.config, split)
            loader_config = self.config["loader"]
            loaders[split] = DataLoader(
                dataset=dataset,
                batch_size=loader_config.get("batch_size", 16),
                shuffle=loader_config.get("shuffle", True) and split == "train",
                num_workers=loader_config.get("num_workers", 4),
                drop_last=loader_config.get("drop_last", False),
                persistent_workers=loader_config.get("persistent_workers", True),
                prefetch_factor=loader_config.get("prefetch_factor", 2),
            )
        return loaders



