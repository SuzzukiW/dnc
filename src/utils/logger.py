# src/utils/logger.py

import os
import csv
import json
import datetime
from typing import Dict, List, Union
import numpy as np

class Logger:
    """Logger for training metrics and progress"""
    
    def __init__(self, log_dir: str):
        """
        Initialize the logger
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log files
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = os.path.join(log_dir, f'metrics_{timestamp}.csv')
        self.episode_file = os.path.join(log_dir, f'episodes_{timestamp}.csv')
        
        # Initialize metrics storage
        self.episode_metrics = []
        self.step_metrics = []
        
        # Create CSV headers
        self._create_csv_files()
        
    def _create_csv_files(self):
        """Create CSV files with headers"""
        # Metrics file headers
        metrics_headers = [
            'episode',
            'total_reward',
            'avg_waiting_time',
            'avg_queue_length',
            'avg_speed',
            'total_co2',
            'epsilon',
            'loss'
        ]
        
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metrics_headers)
            
        # Episode file headers
        episode_headers = [
            'episode',
            'timestamp',
            'total_steps',
            'total_reward',
            'traffic_light_id',
            'avg_queue_length',
            'max_queue_length',
            'avg_waiting_time',
            'max_waiting_time'
        ]
        
        with open(self.episode_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(episode_headers)
    
    def log_step(self, step_data: Dict):
        """
        Log data for a single step
        
        Args:
            step_data: Dictionary containing step metrics
        """
        self.step_metrics.append(step_data)
        
    def log_episode(self, episode: int, episode_data: Dict[str, float]):
        """
        Log data for a complete episode
        
        Args:
            episode: Episode number
            episode_data: Dictionary containing episode metrics for each traffic light
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate aggregate metrics
        metrics = {
            'episode': episode,
            'timestamp': timestamp,
            'total_steps': len(self.step_metrics),
            'total_reward': sum(episode_data.values()),
        }
        
        # Add metrics for each traffic light
        for tl_id, reward in episode_data.items():
            tl_metrics = {
                'traffic_light_id': tl_id,
                'avg_queue_length': np.mean([step['queue_length'][tl_id] 
                                           for step in self.step_metrics if tl_id in step.get('queue_length', {})]),
                'max_queue_length': np.max([step['queue_length'][tl_id] 
                                          for step in self.step_metrics if tl_id in step.get('queue_length', {})]),
                'avg_waiting_time': np.mean([step['waiting_time'][tl_id] 
                                           for step in self.step_metrics if tl_id in step.get('waiting_time', {})]),
                'max_waiting_time': np.max([step['waiting_time'][tl_id] 
                                          for step in self.step_metrics if tl_id in step.get('waiting_time', {})])
            }
            metrics.update(tl_metrics)
        
        # Save to episode file
        with open(self.episode_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writerow(metrics)
        
        # Calculate and save aggregate metrics
        agg_metrics = {
            'episode': episode,
            'total_reward': metrics['total_reward'],
            'avg_waiting_time': np.mean([m['avg_waiting_time'] for m in [metrics]]),
            'avg_queue_length': np.mean([m['avg_queue_length'] for m in [metrics]]),
            'avg_speed': np.mean([step.get('avg_speed', 0) for step in self.step_metrics]),
            'total_co2': sum([step.get('total_co2', 0) for step in self.step_metrics]),
            'epsilon': self.step_metrics[-1].get('epsilon', 0) if self.step_metrics else 0,
            'loss': np.mean([step.get('loss', 0) for step in self.step_metrics])
        }
        
        # Save to metrics file
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(agg_metrics.keys()))
            writer.writerow(agg_metrics)
        
        # Clear step metrics for next episode
        self.step_metrics = []
        
        return metrics
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """
        Get all logged metrics
        
        Returns:
            Dictionary containing lists of metrics
        """
        metrics = {
            'episode': [],
            'total_reward': [],
            'avg_waiting_time': [],
            'avg_queue_length': [],
            'avg_speed': [],
            'total_co2': [],
            'epsilon': [],
            'loss': []
        }
        
        with open(self.metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in metrics:
                    metrics[key].append(float(row[key]))
        
        return metrics
    
    def save_config(self, config: Dict):
        """
        Save configuration to JSON file
        
        Args:
            config: Configuration dictionary
        """
        config_file = os.path.join(self.log_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def log_message(self, message: str):
        """
        Log a message with timestamp
        
        Args:
            message: Message to log
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = os.path.join(self.log_dir, 'training.log')
        
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
