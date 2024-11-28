# experiments/train/train_multi_agent_ppo.py

import os
import sys
import yaml
import torch
import numpy as np
import logging
import argparse
import traci
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import traceback
import pandas as pd
from evaluation_sets.metrics import (
    average_waiting_time, 
    total_throughput, 
    average_speed, 
    max_waiting_time
)

from src.environment.multi_agent_sumo_env_ppo import MultiAgentSumoEnvironmentPPO
from src.agents.ppo_agent import PPOAgent

class MultiAgentPPOTrainer:
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Simplify to CPU for consistent performance
        self.device = torch.device("cpu")
        
        # Get episode and step limits from config
        self.num_episodes = self.config['train']['num_episodes']
        self.max_steps = self.config['train']['max_steps_per_episode']
        
        # Optimize environment settings
        self.config['env']['sumo_step_length'] = 1.0  # Match baseline step length
        self.env = MultiAgentSumoEnvironmentPPO(self.config)
        
        # Pre-initialize agents
        self.agents = {
            agent_id: PPOAgent(
                state_dim=self.env.observation_space_size,
                action_dim=self.env.action_space_size,
                config=self.config,
                agent_id=agent_id
            )
            for agent_id in self.env.agent_ids
        }
        
        # Disable logging for performance
        logging.basicConfig(level=logging.ERROR)
        
        # Initialize metrics tracking with dynamic lists
        self.metrics = {
            'episode': [],
            'step': [],
            'total_waiting_time': [],
            'total_throughput': [],
            'average_speed': [],
            'vehicles_in_network': [],
            'completed_trips': []
        }
        self.episode_summaries = []
        
        # Track vehicles
        self.vehicles_seen = set()
        self.vehicles_completed = set()
        
    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(self.config.get('logging', {}).get('log_dir', 'logs'), timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        
    def _calculate_metrics(self, episode_rewards: dict) -> dict:
        """Calculate traffic metrics for the episode."""
        metrics = {}
        
        # Calculate average reward across all agents (safely handle empty case)
        metrics['avg_reward'] = sum(episode_rewards.values()) / max(1, len(episode_rewards))
        
        # Get all vehicles in the network
        current_vehicles = set(traci.vehicle.getIDList())
        self.vehicles_seen.update(current_vehicles)
        
        # Prepare vehicle data for metrics calculation
        vehicle_data = []
        for vid in current_vehicles:
            vehicle_data.append({
                'waiting_time': traci.vehicle.getWaitingTime(vid),
                'speed': traci.vehicle.getSpeed(vid),
                'distance': traci.vehicle.getDistance(vid)
            })
        
        # Use imported metrics functions
        metrics['mean_waiting_time (s)'] = average_waiting_time(vehicle_data)
        metrics['total_throughput (vehicles)'] = total_throughput(vehicle_data)
        metrics['mean_speed (km/h)'] = average_speed(vehicle_data)
        metrics['max_waiting_time (s)'] = max_waiting_time(vehicle_data)
        
        # Track completed vehicles
        new_completed = self.vehicles_seen - current_vehicles - self.vehicles_completed
        self.vehicles_completed.update(new_completed)
        
        # Additional metrics
        metrics['completed_trips'] = len(self.vehicles_completed)
        metrics['total_vehicles'] = len(self.vehicles_seen)
        metrics['completion_rate (%)'] = (len(self.vehicles_completed) / len(self.vehicles_seen) * 100) \
            if self.vehicles_seen else 0
        
        # Store metrics in lists
        self.metrics['episode'].append(self.current_episode)
        self.metrics['step'].append(self.current_step)
        self.metrics['total_waiting_time'].append(metrics['mean_waiting_time (s)'])
        self.metrics['total_throughput'].append(metrics['total_throughput (vehicles)'])
        self.metrics['average_speed'].append(metrics['mean_speed (km/h)'])
        self.metrics['vehicles_in_network'].append(len(current_vehicles))
        self.metrics['completed_trips'].append(metrics['completed_trips'])
        
        self.current_step += 1
        
        # Store episode summary
        if self.current_step >= self.max_steps:
            self.episode_summaries.append(metrics)
            
        return metrics
        
    def train(self):
        print("\nStarting optimized training...")
        print(f"Number of episodes: {self.num_episodes}")
        print(f"Maximum steps per episode: {self.max_steps}\n")
        
        try:
            # Main training loop with progress bar
            for episode in range(self.num_episodes):
                print(f"\nEpisode {episode + 1}/{self.num_episodes}")
                
                # Reset environment and tracking
                self.vehicles_seen = set()
                self.vehicles_completed = set()
                self.current_episode = episode
                self.current_step = 0
                states, _ = self.env.reset()
                
                # Episode progress bar
                pbar = tqdm(total=self.max_steps, desc='Training Progress', 
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} steps')
                
                # Run episode
                episode_rewards = {agent_id: 0.0 for agent_id in self.agents}  # Initialize with zeros
                done = False
                
                for step in range(self.max_steps):
                    if done:
                        # Update progress bar with remaining steps like baseline
                        remaining_steps = self.max_steps - step
                        pbar.update(remaining_steps)
                        print(f"\nAll vehicles completed their trips - Episode ended at step {step}")
                        break
                    
                    # Get actions from all agents
                    actions = {}
                    action_values = {}
                    action_log_probs = {}
                    for agent_id in self.agents:
                        action, value, log_prob = self.agents[agent_id].select_action(states[agent_id])
                        actions[agent_id] = action
                        action_values[agent_id] = value
                        action_log_probs[agent_id] = log_prob
                    
                    # Environment step
                    next_states, rewards, dones, _ = self.env.step(actions)
                    
                    # Collect metrics AFTER environment step, exactly like baseline
                    metrics = self._calculate_metrics(episode_rewards)
                    
                    # Store transitions
                    for agent_id in self.agents:
                        episode_rewards[agent_id] += rewards[agent_id]
                        self.agents[agent_id].store_transition(
                            state=states[agent_id],
                            action=actions[agent_id],
                            reward=rewards[agent_id],
                            value=action_values[agent_id],
                            log_prob=action_log_probs[agent_id],
                            mask=float(not dones[agent_id])
                        )
                    
                    # Update states
                    states = next_states
                    done = all(dones.values())
                    
                    # Update progress bar
                    pbar.update(1)
                    
                # Train agents
                for agent_id in self.agents:
                    self.agents[agent_id].train()
                
                # Print metrics for this episode
                episode_metrics = self._calculate_metrics(episode_rewards)
                print(f"\nEpisode {episode + 1} Metrics:")
                print(f"avg_waiting_time: {episode_metrics['mean_waiting_time (s)']:.2f}")
                print(f"total_throughput: {episode_metrics['total_throughput (vehicles)']:.0f}")
                print(f"avg_speed: {episode_metrics['mean_speed (km/h)']:.2f}")
                print(f"max_waiting_time: {episode_metrics['max_waiting_time (s)']:.2f}")
                
                pbar.close()
                
            # Compute final summary across all episodes
            if self.episode_summaries:
                print("\nFinal Training Summary:")
                final_summary = {
                    'avg_waiting_time': np.mean([ep['mean_waiting_time (s)'] for ep in self.episode_summaries]),
                    'total_throughput': np.mean([ep['total_throughput (vehicles)'] for ep in self.episode_summaries]),
                    'avg_speed': np.mean([ep['mean_speed (km/h)'] for ep in self.episode_summaries]),
                    'max_waiting_time': np.max([ep['max_waiting_time (s)'] for ep in self.episode_summaries])
                }
                
                # Print final summary
                for metric, value in final_summary.items():
                    print(f"{metric}: {value:.2f}")
        
        except Exception as e:
            print(f"Training error: {e}")
            traceback.print_exc()
        finally:
            self.env.close()
    
def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Optimized Multi-Agent PPO Training")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    trainer = MultiAgentPPOTrainer(args.config)
    trainer.train()

if __name__ == "__main__":
    main()