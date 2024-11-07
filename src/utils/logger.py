# src/utils/logger.py

import logging
import sys
from pathlib import Path

def setup_logger(name, log_file, level=logging.INFO):
    """Setup logger with specified name and log file
    
    Args:
        name (str): Name of the logger
        log_file (str or Path): Path to log file
        level (int): Logging level
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Convert log_file to Path if it's a string
    log_file = Path(log_file)
    
    # Create directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Setup stream handler (console output)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers if any
    logger.handlers = []
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

def get_logger(name):
    """Get existing logger by name
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)