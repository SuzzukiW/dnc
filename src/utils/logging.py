import logging
import os
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Create a logger with optional file output
    
    Args:
        name (str): Name of the logger
        log_file (str, optional): Path to log file
        level (logging level, optional): Logging level
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        # Ensure directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def save_metrics(metrics, file_path):
    """
    Save training metrics to a JSON file
    
    Args:
        metrics (dict): Dictionary of training metrics
        file_path (str): Path to save metrics
    """
    import json
    
    # Ensure directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    import numpy as np
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            metrics_serializable[key] = {
                k: [float(v) if isinstance(v, np.ndarray) else v for v in val]
                for k, val in value.items()
            }
        else:
            metrics_serializable[key] = [float(v) if isinstance(v, np.ndarray) else v for v in value]
    
    with open(file_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)
