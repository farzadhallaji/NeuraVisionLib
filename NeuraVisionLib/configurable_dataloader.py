import os
import torch
from torch.utils.data import Dataset, DataLoader
from NeuraVisionLib.utils.dataloader.file_reader import read_file, get_reader_function_args
from NeuraVisionLib.utils.general import set_seed, load_custom_function, load_config
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



SPLIT_KEYS = ['train', 'val', 'test']


class ConfigurableDataset(Dataset):
    def __init__(self, config, split):
        self._validate_config()

        self.split_name = split
        self.config = config

        self.dataset_name = config["dataset"]["name"]
        self.dataset_type = config["dataset"]["type"]
        self.dataset_paths = config["dataset"]["paths"]
        self.file_format = config["dataset"]["file_format"] 
        self.custom_read_cfg = config["dataset"].get("custom_read_data_file", None)

        self.metadata_cfg = config["metadata"]

        self.split_cfg = config["split"]

        self._load_functions()

        self._initialize_split()
        self.custom_split = config["split"].get("custom_split", None)

        self.preprocessing = config.get("preprocessing", {})
        self.augmentation = config.get("augmentation", {})

        self.samples = self._load_samples()


    def _initialize_split(self):
        self.split_ratio = {}
        self.split_dirs = {}
        for sp in SPLIT_KEYS:
            d_split = self.split_cfg[sp]
            if isinstance(d_split, str):
                self.split_dirs[sp] = [os.path.join(path, d_split) for path in self.dataset_paths]
                self.split_ratio[sp] = 1.0
            elif isinstance(d_split, float):
                self.split_dirs[sp] = self.dataset_paths
                self.split_ratio[sp] = d_split
            else:
                raise ValueError(f"Invalid split configuration for '{sp}': {d_split}")
        for sp, dir_paths in self.split_dirs.items():
            for dir_path in dir_paths:
                if not os.path.exists(dir_path):
                    raise FileNotFoundError(f"Dataset split directory does not exist: {sp} -> {dir_path}")

        self.split_random_state = self.split_cfg.get("random_state", None)
        self.split_stratify = self.split_cfg.get("stratify", False)
        self.split_shuffle = self.split_cfg.get("shuffle", None)

    def _get_all_files(self):
        inputs, targets = [], []  
        for pth in self.split_dirs[self.split_name]: 
            if not os.path.exists(pth):
                raise FileNotFoundError(f"Dataset directory does not exist: {self.split_name} -> {pth}")
            curr_inputs = sorted(
                [os.path.join(pth, f) for f in os.listdir(pth) if f.endswith(self.file_format["input"])]
            )
            curr_targets = sorted(
                [os.path.join(pth, f) for f in os.listdir(pth) if f.endswith(self.file_format["target"])]
            )
            if len(curr_inputs) != len(curr_targets):
                raise ValueError(f"Mismatch between input and target files in directory: {self.split_name} -> {pth}")
            inputs.extend(curr_inputs)
            targets.extend(curr_targets)
        return inputs, targets
    
    def default_split_data(self, inputs, targets, test_size, val_size=0.0, random_state=None, shuffle=True, stratify=None):
        """
        Wrapper for splitting data into train, val, and test sets with optional stratification.
        """
        if not inputs or not targets:
            raise ValueError(f"No data found for split: {self.split_name}")

        def split(inputs, targets, test_size, stratify):
            return train_test_split(inputs, targets, test_size=test_size, random_state=random_state, stratify=stratify)

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

    def _apply_split(self, inputs, targets):
        """ordered splitting: 1.train, 2.val, 3.test"""
        if self.custom_split:
            if self.custom_split.get('enabled', True):
                custom_train_test_split = load_custom_function(self.custom_split.get('function', None))
                return custom_train_test_split(inputs, targets, split_name=self.split_name, config=self.split_cfg)

        train_ratio = self.split_ratio.get("train", 0.0)
        val_ratio = self.split_ratio.get("val", 0.0)
        test_ratio = self.split_ratio.get("test", 0.0)
        split_method = self.split_cfg['method'] 
        if split_method =='ratio':
            if train_ratio + val_ratio + test_ratio > 1.0:
                raise ValueError("Split ratios for train, val, and test must sum to 1.0 or less.")
            test_size = val_ratio + test_ratio
            val_size = val_ratio

            train_data, val_data, test_data = self.default_split_data(
                inputs, targets,
                test_size=test_size,
                val_size=val_size,
                random_state=self.split_random_state,
                shuffle=self.split_shuffle,
                stratify=self.split_stratify
            )
            if self.split_name == "train":
                return train_data
            elif self.split_name == "val":
                return val_data
            elif self.split_name == "test":
                return test_data
            else:
                raise ValueError(f"Unknown split type: {self.split_name}")

        elif split_method == 'folder':
            if self.split_shuffle:
                set_seed(self.split_random_state) 
                combined = list(zip(inputs, targets))
                random.shuffle(combined)
                inputs, targets = zip(*combined)  
            
            return list(zip(inputs, targets))

        else:
            raise ValueError(f"Unknown split method '{split_method}'. Expected 'ratio' or 'folder'.")


    def _load_samples(self):
        inputs, targets = self._get_all_files()
        samples = self._apply_split(inputs, targets)
        if not samples:
            raise ValueError(f"No samples found for split: {self.split_name}")
        return samples
    
    def _validate_config(self):
        """Validates the dataset configuration and raises errors for missing or incorrect settings."""
        
        required_keys = ["dataset", "split", "loader"]
        
        # Ensure required keys exist
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key in config: {key}")
        
        # Validate dataset settings
        dataset_cfg = self.config["dataset"]
        
        if "name" not in dataset_cfg or not isinstance(dataset_cfg["name"], str):
            raise ValueError("Dataset name must be defined as a string.")
        
        valid_types = ["2d", "3d", "4d", "text", "tabular", "pointcloud", "custom"]
        if dataset_cfg.get("type") not in valid_types:
            raise ValueError(f"Invalid dataset type '{dataset_cfg.get('type')}'. Must be one of {valid_types}.")
        
        if "paths" not in dataset_cfg or not isinstance(dataset_cfg["paths"], list) or len(dataset_cfg["paths"]) == 0:
            raise ValueError("Dataset paths must be a non-empty list of valid directories.")
        
        # Validate file formats
        if "file_format" not in dataset_cfg or not isinstance(dataset_cfg["file_format"], dict):
            raise ValueError("Missing 'file_format' section in dataset configuration.")
        
        for key in ["input", "target"]:
            if key not in dataset_cfg["file_format"] or not isinstance(dataset_cfg["file_format"][key], str):
                raise ValueError(f"File format '{key}' must be defined as a string in 'file_format'.")

        # Validate metadata settings (optional)
        metadata_cfg = self.config.get("metadata", {})
        
        if metadata_cfg.get("enabled", False):
            if "file_path" not in metadata_cfg or not isinstance(metadata_cfg["file_path"], str):
                raise ValueError("Metadata file path must be defined when metadata is enabled.")
            
            for col_key in ["key_column", "target_column"]:
                if col_key not in metadata_cfg or not isinstance(metadata_cfg[col_key], str):
                    raise ValueError(f"Metadata {col_key} must be a valid column name.")

        # Validate split settings
        split_cfg = self.config["split"]
        valid_methods = ["folder", "ratio", "custom"]

        if "method" not in split_cfg or split_cfg["method"] not in valid_methods:
            raise ValueError(f"Invalid split method '{split_cfg.get('method')}'. Must be one of {valid_methods}.")

        if split_cfg["method"] == "ratio":
            total_ratio = sum(split_cfg.get(key, 0) for key in SPLIT_KEYS)
            if total_ratio > 1.0:
                raise ValueError("Train, validation, and test split ratios must sum to 1.0 or less.")
        
        elif split_cfg["method"] == "custom":
            if "custom_split" not in split_cfg or not isinstance(split_cfg["custom_split"], dict):
                raise ValueError("Custom split must be defined as a dictionary when method is 'custom'.")
            
            if not split_cfg["custom_split"].get("enabled", False):
                raise ValueError("Custom split must be enabled when method is 'custom'.")
            
            if "function" not in split_cfg["custom_split"]:
                raise ValueError("Missing function definition in custom split configuration.")

        # Validate preprocessing operations
        preprocessing_cfg = self.config.get("preprocessing", {})
        
        for key in ["input_operations", "target_operations", "both_operations"]:
            if key in preprocessing_cfg and isinstance(preprocessing_cfg[key], list):
                for operation in preprocessing_cfg[key]:
                    if not isinstance(operation, dict) or "name" not in operation or "enabled" not in operation:
                        raise ValueError(f"Invalid preprocessing operation format in {key}.")
        
        # Validate augmentation settings
        augmentation_cfg = self.config.get("augmentation", {})
        
        for key in ["input_operations", "target_operations", "both_operations"]:
            if key in augmentation_cfg and isinstance(augmentation_cfg[key], list):
                for operation in augmentation_cfg[key]:
                    if not isinstance(operation, dict) or "name" not in operation or "enabled" not in operation:
                        raise ValueError(f"Invalid augmentation operation format in {key}.")

        # Validate dataloader settings
        loader_cfg = self.config["loader"]
        
        if "batch_size" not in loader_cfg or not isinstance(loader_cfg["batch_size"], int) or loader_cfg["batch_size"] <= 0:
            raise ValueError("Batch size must be a positive integer.")
        
        if "num_workers" not in loader_cfg or not isinstance(loader_cfg["num_workers"], int) or loader_cfg["num_workers"] < 0:
            raise ValueError("Number of workers must be a non-negative integer.")

        if "shuffle" not in loader_cfg or not isinstance(loader_cfg["shuffle"], bool):
            raise ValueError("Shuffle must be a boolean value.")

        if loader_cfg.get("use_seed", False):
            if "seed" not in loader_cfg or not isinstance(loader_cfg["seed"], int):
                raise ValueError("Seed value must be defined as an integer when 'use_seed' is enabled.")

        logger.info("Configuration validation passed successfully.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):

        input_path, target_path = self.samples[index]
        _read_file = read_file
        file_read_args = {}

        if self.custom_read_cfg and self.custom_read_cfg.get('enabled', False):
            custom_fun_cfg = self.custom_read_cfg['function']
            _read_file = load_custom_function(custom_fun_cfg)
            file_read_args = custom_fun_cfg.get("args", {})
        else:
            file_format_input = self.file_format["input"]
            file_format_target = self.file_format["target"]
            file_read_args = {
                "input": get_reader_function_args(file_format_input),
                "target": get_reader_function_args(file_format_target)
            }

        input_data = _read_file(input_path, **file_read_args.get("input", {}))
        target_data = _read_file(target_path, **file_read_args.get("target", {}))
        input_data, target_data = self.apply_operations(input_data, target_data)

        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.float32)

        return input_tensor, target_tensor

    def apply_operations(self, input_data, target_data):
        for operation in self.preprocessing.get("input_operations", []) + self.augmentation.get("input_operations", []):
            if operation["enabled"]:
                input_data = self._apply_operation(input_data, operation)

        for operation in self.preprocessing.get("target_operations", []) + self.augmentation.get("target_operations", []):
            if operation["enabled"]:
                target_data = self._apply_operation(target_data, operation)

        for operation in self.preprocessing.get("both_operations", []) + self.augmentation.get("both_operations", []):
            if operation["enabled"]:
                input_data = self._apply_operation(input_data, operation)
                target_data = self._apply_operation(target_data, operation)

        return input_data, target_data

    def visualize_sample(self, index=None, save_path=None):
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

    def _load_functions(self):
        """
        Loads all functions specified in 'functions_to_load_first' from the config.

        Stores them in a dictionary for easy access later.
        """
        self.loaded_functions = {}

        functions_list = self.config.get("functions_to_load_first", [])
        
        for func_cfg in functions_list:
            func_details = func_cfg.get("function", {})

            function_name = func_details.get("function_name")
            module_path = func_details.get("module_path")
            function_args = func_details.get("args", {})

            if not function_name or not module_path:
                raise ValueError(f"Invalid function configuration: {func_cfg}")

            # Load the function dynamically
            loaded_function = load_custom_function({
                "name": function_name,
                "module_path": module_path
            })

            # Store function and its args in a dictionary
            self.loaded_functions[function_name] = {
                "function": loaded_function,
                "args": function_args
            }

            logger.info(f"Loaded function: {function_name} from {module_path}")




# collate_fn(batch)	Custom function to merge samples into a batch	❌ No (default works)
# worker_init_fn(worker_id)	Ensures each worker gets a different random seed	❌ No (for multiprocessing)
# get_class_distribution()	Returns class distribution (useful for stratification)	❌ No (for debugging)





class DataLoaderFactory:
    def __init__(self, config_path):
        self.config = load_config(config_path)

    def get_all_dataloaders(self):
        """Get DataLoader objects for all splits (train, val, test)."""
        loaders = {}
        for split in SPLIT_KEYS:
            dataset = ConfigurableDataset(self.config, split)
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




###### from line 127 yaml