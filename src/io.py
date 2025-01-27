import glob
import logging
import yaml

def load_config(path: str) -> dict:
    """Load yaml configuration file from path"""
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError as e:
        logging.error(f"config file not found at {path}")
        raise e
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file at {path}: {e}")
        raise e
    except Exception as e:
        logging.error(f"unexpected error occured while loading config file at {path}: {e}")
        raise e
