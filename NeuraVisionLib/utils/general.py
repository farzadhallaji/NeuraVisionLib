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


def load_custom_function(custom_function_config):
    """
    Dynamically import a custom function based on the configuration.

    Args:
        custom_function_config (dict): Dictionary containing function details.

    Returns:
        callable: The imported function.
        dict: The additional arguments for the function.
    """
    if not custom_function_config or not custom_function_config.get("name") or not custom_function_config.get("module_path"):
        raise ValueError("Custom function configuration is incomplete.")
    
    # Import the module and function
    module = importlib.import_module(custom_function_config["module_path"])
    function = getattr(module, custom_function_config["name"])
    
    # Extract additional arguments
    args = custom_function_config.get("args", {})
    return function, args


def load_module_function_from_config(config):
    """
    Dynamically loads a function based on a configuration dictionary.

    :param config: Dictionary containing module_path, function_name, and optional args.
    :return: Loaded function object with args partially applied if provided.
    :raises ImportError: If the module or function cannot be loaded.
    """
    module_path = config.get("module_path")
    function_name = config.get("name")
    args = config.get("args", {})

    if not module_path or not function_name:
        raise ValueError("'module_path' and 'name' are required in the config.")

    try:
        module = importlib.import_module(module_path)
        function = getattr(module, function_name)
        logger.info(f"Successfully loaded function '{function_name}' from module '{module_path}'")

        # If args are provided, return a partially applied function
        if args:
            from functools import partial
            function = partial(function, **args)

        return function
    except ImportError as e:
        logger.error(f"Error importing module '{module_path}': {e}")
        raise ImportError(f"Module '{module_path}' could not be imported: {e}")
    except AttributeError as e:
        logger.error(f"Function '{function_name}' not found in module '{module_path}': {e}")
        raise AttributeError(f"Function '{function_name}' not found in module '{module_path}': {e}")
