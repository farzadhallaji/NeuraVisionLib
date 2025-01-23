import yaml
import importlib
import logging
import random
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning for deterministic runs

def dynamic_import(module_path, function_name):
    try:
        module = importlib.import_module(module_path.replace("/", ".").rstrip(".py"))
        return getattr(module, function_name)
    except Exception as e:
        logger.exception(f"Error importing {function_name} from {module_path}: {e}")
        raise

