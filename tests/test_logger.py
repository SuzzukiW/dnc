# tests/test_logger.py

import os
import tempfile
import shutil
from src.utils.logger import Logger

def test_logger():
    """Test basic logger functionality"""
    # Create temporary directory for logs
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize logger
        logger = Logger(temp_dir)
        
        # Test step logging
        step_data = {
            'queue_length': {'tl1': 5, 'tl2': 3},
            'waiting_time': {'tl1': 10.5, 'tl2': 8.2},
            'avg_speed': 15.0,
            'total_co2': 100.0,
            'epsilon': 0.95,
            'loss': 0.5
        }
        logger.log_step(step_data)
        
        # Test episode logging
        episode_data = {
            'tl1': 100.0,
            'tl2': 150.0
        }
        metrics = logger.log_episode(0, episode_data)
        
        # Verify metrics were logged
        assert os.path.exists(logger.metrics_file)
        assert os.path.exists(logger.episode_file)
        
        # Test getting metrics
        all_metrics = logger.get_metrics()
        assert len(all_metrics['episode']) == 1
        assert all_metrics['total_reward'][0] == 250.0  # 100 + 150
        
        print("Logger test passed!")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    test_logger()