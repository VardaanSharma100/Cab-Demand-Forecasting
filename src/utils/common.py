import joblib 
import os
import logging
import sys

def setup_logger():
    logging.basicConfig(
        level=logging.info,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def save_object(obj,file_path):
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    joblib.dump(obj,file_path)

def load_object(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at{file_path}")
    return joblib.load(file_path)