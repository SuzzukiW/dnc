# src/utils/logger.py

import logging

def get_logger(name, level=logging.INFO):
    """
    Initializes and returns a logger with the specified name and level.
    
    Args:
        name (str): The name of the logger.
        level (int, optional): The logging level. Defaults to logging.INFO.
    
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(ch)
    
    return logger
