import os
import numpy as np
from typing import Dict, List, Optional

class StateSharing:
    """
    Scenario class for hierarchical state sharing experiments
    """
    def __init__(
        self,
        net_file: str,
        route_file: str,
        use_gui: bool = False,
        num_episodes: int = 100,
        episode_length: int = 3600,
        delta_time: int = 5,
    ):
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.delta_time = delta_time

        # Initialize experiment metrics
        self.metrics = {
            'episode_rewards': [],
            'episode_waiting_times': [],
            'episode_throughput': [],
            'episode_queue_lengths': [],
            'regional_performance': {},
            'coordination_metrics': []
        }

    def preprocess_state(self, state: Dict) -> np.ndarray:
        """
        Advanced state preprocessing with feature extraction
        """
        # Extract raw state components
        own_state = np.array(state['own_state']).flatten()
        
        # Feature engineering
        # 1. Queue length features
        queue_features = own_state[:len(own_state)//3]
        queue_mean = np.mean(queue_features)
        queue_std = np.std(queue_features)
        
        # 2. Waiting time features
        wait_features = own_state[len(own_state)//3:2*len(own_state)//3]
        wait_mean = np.mean(wait_features)
        wait_std = np.std(wait_features)
        
        # 3. Speed features
        speed_features = own_state[2*len(own_state)//3:-1]
        speed_mean = np.mean(speed_features)
        speed_std = np.std(speed_features)
        
        # 4. Current phase
        current_phase = own_state[-1]
        
        # Combine engineered features
        engineered_state = np.array([
            queue_mean, queue_std,
            wait_mean, wait_std,
            speed_mean, speed_std,
            current_phase
        ])
        
        # Normalize with robust scaling
        def robust_scale(x):
            median = np.median(x)
            iqr = np.percentile(x, 75) - np.percentile(x, 25)
            return np.clip((x - median) / (iqr + 1e-8), -3, 3)
        
        normalized_state = robust_scale(engineered_state)
        
        return normalized_state

    def calculate_reward(
        self,
        state: Dict,
        action: int,
        next_state: Dict,
        info: Dict
    ) -> float:
        """
        Advanced reward calculation focusing on wait time reduction and throughput
        """
        # Extract key metrics with more precise extraction
        waiting_time = info.get('waiting_time', 0)
        throughput = info.get('throughput', 0)
        queue_length = info.get('queue_length', 0)
        
        # Aggressive wait time penalty
        wait_time_penalty = -2.0 * (waiting_time / 100.0) ** 2  # Quadratic penalty
        
        # Throughput reward with exponential scaling
        throughput_reward = np.log1p(throughput) * 1.5
        
        # Queue length penalty
        queue_penalty = -1.5 * (queue_length / 10.0)
        
        # Combine rewards with carefully tuned weights
        total_reward = (
            wait_time_penalty +   # Strong negative for wait times
            throughput_reward +   # Positive for throughput
            queue_penalty         # Penalty for queue buildup
        )
        
        return total_reward

    def log_metrics(
        self,
        episode: int,
        states: Dict[str, Dict],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        info: Dict
    ):
        """
        Simplified metrics logging matching Baseline
        """
        # Aggregate metrics across all agents
        total_waiting_time = 0
        total_throughput = 0
        total_vehicles = 0
        
        for tl_id, state in states.items():
            # Extract metrics from state or info
            waiting_time = info.get('waiting_time', 0)
            throughput = info.get('throughput', 0)
            
            total_waiting_time += waiting_time
            total_throughput += throughput
            total_vehicles += 1
        
        # Store metrics for this episode
        self.metrics['episode_rewards'].append(np.mean(list(rewards.values())))
        self.metrics['episode_waiting_times'].append(total_waiting_time / max(1, total_vehicles))
        self.metrics['episode_throughput'].append(total_throughput)

    def get_results(self) -> Dict:
        """
        Return experiment results
        """
        return {
            'rewards': self.metrics['episode_rewards'],
            'waiting_times': self.metrics['episode_waiting_times'],
            'throughput': self.metrics['episode_throughput'],
            'queue_lengths': self.metrics['episode_queue_lengths'],
            'regional_performance': self.metrics['regional_performance'],
            'coordination': self.metrics['coordination_metrics']
        }