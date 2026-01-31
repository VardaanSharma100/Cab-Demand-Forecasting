import joblib 
import os
import logging
import sys

def setup_logger():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(project_root, 'execution.log')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'logs')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path),logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def save_object(obj,file_path):
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    joblib.dump(obj,file_path)

def load_object(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at{file_path}")
    return joblib.load(file_path)