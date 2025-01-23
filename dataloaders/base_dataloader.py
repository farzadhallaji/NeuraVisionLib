import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.dataloader.dataloader import read_file
from utils.dataloader.preprocess import resize, normalize, binarize, apply_sliding_window
from utils.dataloader.augmentation import color_jitter, random_flip, random_rotation
from utils.general import load_config
from sklearn.model_selection import train_test_split

SPLIT_KEYS = ['train', 'val', 'test']


class BaseDataset(Dataset):
    def __init__(self, config, split):
        self.split = split

        self.config = config
        self.dataset_name = config["dataset"]["name"]
        self.dataset_path = config["dataset"]["path"]
        self._initialize_split(config)
        self.file_format = config["dataset"]["file_format"]
        self.preprocessing = config.get("preprocessing", {}).get("operations", [])
        self.augmentations = config.get("augmentation", {}).get("operations", [])
        enable_custom_read_file = config.get("dataset", {}).get("custom_read_file", {}).get("enabled", False)
        if enable_custom_read_file:
            self.read_file = self.custom_read_file
        else:
            self.read_file = read_file  
        
        for sp in SPLIT_KEYS:
            if not os.path.exists(self.data_dir[sp]):
                raise FileNotFoundError(f"Dataset split directory does not exist: {self.data_dir}")

        self.samples = self._load_samples()

        print(f"Loaded {len(self.samples)} samples for split: {self.split}")

    def _initialize_split(self, config):
        split_config = config["dataset"]["split"]
        self.split_ratio = {}
        self.split_dir = {}
        self.data_dir = {}
        for sp in SPLIT_KEYS:
            d_split = split_config[sp]
            if isinstance(d_split, str):
                self.split_dir[sp] = d_split
                self.split_ratio[sp] = 1.0
                self.data_dir[sp] = os.path.join(self.dataset_path, self.split_dir[sp])
            elif isinstance(d_split, float):
                self.split_dir[sp] = self.dataset_path
                self.data_dir[sp] = self.dataset_path
                self.split_ratio[sp] = d_split
            else:
                raise ValueError(f"Invalid split configuration for '{self.split}': {d_split}")

        self.split_use_seed = split_config.get("use_seed", False)
        self.split_seed = split_config.get("seed", None)
        self.split_shuffle = split_config.get("shuffle", True)

    def _get_all_files(self):
        if not os.path.exists(self.split_dir[self.split]):
            raise FileNotFoundError(f"Dataset directory does not exist: {self.split_dir[self.split]}")
        
        inputs = sorted(
            [os.path.join(self.split_dir[self.split], f) for f in os.listdir(self.split_dir[self.split]) if f.endswith(self.file_format["input"])]
        )
        targets = sorted(
            [os.path.join(self.split_dir[self.split], f) for f in os.listdir(self.split_dir[self.split]) if f.endswith(self.file_format["target"])]
        )

        if len(inputs) != len(targets):
            raise ValueError("Mismatch between input and target files.")

        return inputs, targets


    def _apply_split(self, inputs, targets):
        """Apply ordered splitting for train, val, and test sets."""
        
        # Case 1: Directory-based splitting
        if isinstance(self.split_dir[self.split], str):
            return list(zip(inputs, targets))

        # Case 2: Ratio-based splitting
        total_samples = len(inputs)
        train_ratio = self.split_ratio.get("train", 0.0)
        val_ratio = self.split_ratio.get("val", 0.0)
        test_ratio = self.split_ratio.get("test", 0.0)

        if train_ratio + val_ratio + test_ratio > 1.0:
            raise ValueError("Split ratios for train, val, and test must sum to 1.0 or less.")

        # Compute adjusted ratios
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.0

        # Split into train and temp (temp contains val+test)
        train_inputs, temp_inputs, train_targets, temp_targets = train_test_split(
            inputs, targets, test_size=(val_ratio + test_ratio), random_state=self.split_seed, shuffle=self.split_shuffle
        )

        # Further split temp into val and test
        if val_ratio + test_ratio > 0:
            val_inputs, test_inputs, val_targets, test_targets = train_test_split(
                temp_inputs, temp_targets, test_size=(1 - val_ratio_adjusted), random_state=self.split_seed, shuffle=self.split_shuffle
            )
        else:
            val_inputs, val_targets, test_inputs, test_targets = [], [], [], []

        # Return the appropriate split
        if self.split == "train":
            return list(zip(train_inputs, train_targets))
        elif self.split == "val":
            return list(zip(val_inputs, val_targets))
        elif self.split == "test":
            return list(zip(test_inputs, test_targets))
        else:
            raise ValueError(f"Unknown split type: {self.split}")


    def _load_samples(self):
        inputs, targets = self._get_all_files()
        samples = self._apply_split(inputs, targets)
        if not samples:
            raise ValueError(f"No samples found for split: {self.split}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        input_path, target_path = self.samples[index]
        input_data = self.read_file(input_path)
        target_data = self.read_file(target_path)

        for operation in self.preprocessing:
            if operation["enabled"]:
                if operation["apply_to"] == "input":
                    input_data = self._apply_operation(input_data, operation)
                elif operation["apply_to"] == "target":
                    target_data = self._apply_operation(target_data, operation)
                elif operation["apply_to"] == "both":
                    input_data = self._apply_operation(input_data, operation)
                    target_data = self._apply_operation(target_data, operation)
        
        for operation in self.augmentations:
            if operation["enabled"]:
                input_data = self._apply_operation(input_data, operation)
                target_data = self._apply_operation(target_data, operation)

        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.float32)

        return input_tensor, target_tensor

    def _apply_operation(self, data, operation):
        operation_name = operation["name"]
        if operation_name == "resize":
            return resize(data, operation)
        elif operation_name == "normalization":
            return normalize(data, operation)
        elif operation_name == "binarization":
            return binarize(data, operation)
        elif operation_name == "sliding_window":
            return apply_sliding_window(data, operation)
        elif operation_name == "custom_preprocess_input":
            return self._custom_preprocess_input(data, operation)
        elif operation_name == "custom_preprocess_target":
            return self._custom_preprocess_target(data, operation)
        elif operation_name == "custom_preprocess_both":
            return self._custom_preprocess_both(data, operation)
        
        elif operation_name == "random_flip":
            return normalize(data, operation)
        elif operation_name == "random_rotation":
            return binarize(data, operation)
        elif operation_name == "color_jitter":
            return binarize(data, operation)
        elif operation_name == "custom_augmentations_input":
            return self._custom_augmentations_input(data, operation)
        elif operation_name == "custom_augmentations_target":
            return self._custom_augmentations_target(data, operation)
        elif operation_name == "custom_augmentations_both":
            return self._custom_augmentations_both(data, operation)
        else:
            raise ValueError(f"Unsupported operation: {operation_name}")
    
    
    def custom_read_file(self, file_path, **kwargs):
        print("Custom read file function called with the following parameters:")
        print(f"File Path: {file_path}")
        for key, value in kwargs.items():
            print(f"{key}: {value}")
        raise NotImplementedError("custom_read_file function is not implemented. Please define your custom logic.")
    
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


class DataLoaderFactory:
    def __init__(self, config_path):
        self.config = load_config(config_path)

    def get_dataloader(self, split):
        dataset = BaseDataset(self.config, split)

        loader_config = self.config["loader"]
        return DataLoader(
            dataset=dataset,
            batch_size=loader_config.get("batch_size", 16),
            shuffle=loader_config.get("shuffle", True) and split == "train",
            num_workers=loader_config.get("num_workers", 4),
            drop_last=loader_config.get("drop_last", False),
            persistent_workers=loader_config.get("persistent_workers", True),
            prefetch_factor=loader_config.get("prefetch_factor", 2),
        )


