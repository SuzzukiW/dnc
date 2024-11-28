# experiments/train/train_hierarchical.py

import os
import sys
import yaml
import json
import logging
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import project-specific modules
from src.environment.multi_agent_sumo_env_hierarchical import HierarchicalTrafficEnv
from experiments.scenarios.communication.hierarchical import HierarchicalScenarios
from src.agents.dqn_agent import DQNAgent

# Import metrics for state sharing
from evaluation_sets.metrics import (
    average_waiting_time, 
    total_throughput, 
    average_speed, 
    max_waiting_time
)

from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import time
import traci
from tqdm import tqdm

class HierarchicalTrainer:
    """
    Trainer for hierarchical multi-agent traffic control system
    Manages both local intersection agents and regional coordinators
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the hierarchical trainer
        
        Args:
            config_path: Path to configuration file
        """
        # Validate config path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration with error handling
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error loading configuration: {e}")
            raise
        
        # Validate configuration
        if not isinstance(self.config, dict):
            raise ValueError("Invalid configuration format. Must be a dictionary.")
        
        # Set up logging
        self._setup_logging()
        
        # Provide default values for missing configuration keys
        default_config = {
            'simulation': {
                'net_file': None,
                'trip_file': None,
                'use_gui': False,
                'max_steps': 1000,
                'delta_time': 5,
                'step_length': 1.0,
                'num_episodes': 100
            },
            'hierarchy': {
                'num_regions': 1
            },
            'training': {
                'save_frequency': 10,
                'log_dir': 'logs',
                'checkpoint_dir': 'checkpoints'
            }
        }
        
        # Merge default config with loaded config
        from copy import deepcopy
        def merge_configs(default, user):
            merged = deepcopy(default)
            for key, value in user.items():
                if isinstance(value, dict) and key in merged:
                    merged[key] = merge_configs(merged[key], value)
                else:
                    merged[key] = value
            return merged
        
        self.config = merge_configs(default_config, self.config)
        
        # Convert simulation config to environment config
        env_config = {
            'environment': {
                'sumo_net_file': self.config['simulation']['net_file'],
                'sumo_route_file': self.config['simulation']['trip_file'],
                'gui': self.config['simulation']['use_gui'],
                'max_episode_steps': self.config['simulation']['max_steps'],
                'delta_time': int(self.config['simulation']['delta_time']),
                'step_length': float(self.config['simulation']['step_length'])
            },
            'hierarchy': self.config['hierarchy'],
            'agent': self.config.get('agent', {}),
            'training': {
                'reward_weights': self.config.get('training', {}).get('reward_weights', {}),
                'coordination_weights': self.config.get('training', {}).get('coordination_weights', {}),
                'eval_frequency': self.config.get('training', {}).get('eval_frequency', 10),
                'save_frequency': self.config.get('training', {}).get('save_frequency', 10)
            }
        }
        
        # Initialize environment
        self.env = HierarchicalTrafficEnv(env_config)
        
        # Initialize agents
        self.local_agents = self._init_local_agents()
        self.coordinators = self._init_coordinators()
        
        # Initialize metrics
        self.metrics = {
            'episode': [],
            'step': [],
            'total_waiting_time': [],
            'total_throughput': [],
            'average_speed': [],
            'vehicles_in_network': [],
            'completed_trips': []
        }
        
        # Set up checkpoint directory
        self.checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Create scenario config with environment structure
        scenario_config = {
            'environment': {
                'sumo_net_file': self.config['simulation']['net_file'],
                'sumo_route_file': self.config['simulation']['trip_file']
            },
            'hierarchy': self.config['hierarchy']
        }
        
        # Save temporary config for scenarios
        temp_config_path = os.path.join(os.path.dirname(config_path), 'temp_scenario_config.yaml')
        with open(temp_config_path, 'w') as f:
            yaml.dump(scenario_config, f)
        
        # Load scenarios with correct config structure
        self.scenario_generator = HierarchicalScenarios(temp_config_path)
        self.scenarios = self.scenario_generator.generate_scenarios()
        
        # Clean up temporary config
        os.remove(temp_config_path)
        
    def _setup_logging(self):
        """Configure logging for training"""
        log_dir = self.config['training']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def _init_local_agents(self):
        """
        Initialize local agents for each traffic light
        
        Returns:
            Dict[str, DQNAgent]: Dictionary of local agents
        """
        local_agents = {}
        
        # Dynamically determine state and action sizes
        state_size = self.config['agent']['state_size']
        action_size = self.config['agent']['action_size']
        
        # Create local agents for each traffic light
        for tl_id in self.env.traffic_lights:
            agent_config = {
                'gamma': self.config['agent'].get('gamma', 0.95),
                'learning_rate': self.config['agent'].get('learning_rate', 0.001),
                'epsilon_start': self.config['agent'].get('epsilon_start', 1.0),
                'epsilon_min': self.config['agent'].get('epsilon_min', 0.05),
                'epsilon_decay': self.config['agent'].get('epsilon_decay', 0.995),
                'batch_size': self.config['agent'].get('batch_size', 256),
                'memory_size': self.config['agent'].get('memory_size', 500000)
            }
            
            local_agents[tl_id] = DQNAgent(state_size, action_size, agent_config)
        
        return local_agents
    
    def _init_coordinators(self):
        """
        Initialize regional coordinators
        
        Returns:
            Dict[str, DQNAgent]: Dictionary of regional coordinators
        """
        coordinators = {}
        
        # Dynamically determine coordinator state and action sizes
        coordinator_state_size = self.config['agent']['coordinator_state_size']
        coordinator_action_size = self.config['agent']['action_size']
        
        # Determine number of regions
        num_regions = self.config['hierarchy']['num_regions']
        
        # Create coordinators for each region
        for region_id in range(num_regions):
            coordinator_config = {
                'gamma': self.config['agent'].get('gamma', 0.95),
                'learning_rate': self.config['agent'].get('learning_rate', 0.001),
                'epsilon_start': self.config['agent'].get('epsilon_start', 1.0),
                'epsilon_min': self.config['agent'].get('epsilon_min', 0.05),
                'epsilon_decay': self.config['agent'].get('epsilon_decay', 0.995),
                'batch_size': self.config['agent'].get('batch_size', 256),
                'memory_size': self.config['agent'].get('memory_size', 500000)
            }
            
            coordinators[region_id] = DQNAgent(
                coordinator_state_size, 
                coordinator_action_size, 
                coordinator_config
            )
        
        return coordinators
    
    def _get_coordinator_actions(self, states: Dict[str, np.ndarray]) -> Dict[int, int]:
        """
        Get actions from regional coordinators
        
        Args:
            states: Dictionary of local agent states
            
        Returns:
            Dictionary mapping region IDs to coordinator actions
        """
        coordinator_actions = {}
        
        # Get coordinator states for each region
        for region_id in range(self.config['hierarchy']['num_regions']):
            state = self.env.get_coordinator_state(region_id)
            action = self.coordinators[region_id].act(state)
            coordinator_actions[region_id] = action
            
        return coordinator_actions
        
    def _get_local_actions(self, states: Dict[str, np.ndarray], 
                          coordinator_actions: Dict[int, int]) -> Dict[str, int]:
        """
        Get actions from local agents with coordinator guidance
        
        Args:
            states: Dictionary of local agent states
            coordinator_actions: Dictionary of coordinator actions
            
        Returns:
            Dictionary mapping intersection IDs to actions
        """
        actions = {}
        
        # Get actions for each traffic light
        for tl_id in states:
            # Get the region this traffic light belongs to
            region_id = None
            for r_id, tls in self.env.regions.items():
                if tl_id in tls:
                    region_id = r_id
                    break
                    
            if region_id is None:
                print(f"Warning: Traffic light {tl_id} not found in any region")
                continue
                
            # Get coordinator guidance
            coordinator_guidance = coordinator_actions[region_id]
            
            # Augment state with coordinator guidance
            augmented_state = self._augment_state(states[tl_id], coordinator_guidance)
            
            # Get action from local agent
            action = self.local_agents[tl_id].act(augmented_state)
            actions[tl_id] = action
            
        return actions
    
    def _augment_state(self, state: np.ndarray, coordinator_guidance: int) -> np.ndarray:
        """Combine local state with coordinator guidance"""
        # Convert coordinator guidance to one-hot encoding
        guidance_one_hot = np.zeros(self.config['agent']['action_size'])
        guidance_one_hot[coordinator_guidance] = 1
        
        # Concatenate with local state
        return np.concatenate([state, guidance_one_hot])
    
    def _store_experience(self, states: Dict[str, np.ndarray], 
                          actions: Dict[str, int],
                          rewards: Dict[str, float],
                          next_states: Dict[str, np.ndarray],
                          dones: Dict[str, bool]):
        """Store experience in memory"""
        # Store local experiences
        for tl_id in states:
            self.local_agents[tl_id].remember(
                states[tl_id],
                actions[tl_id],
                rewards[tl_id],
                next_states[tl_id],
                dones[tl_id]
            )
            
    def _update_local_agents(self):
        """
        Update local intersection agents
        
        Returns:
            List of training losses
        """
        losses = []
        for tl_id, agent in self.local_agents.items():
            loss = agent.replay()
            if loss is not None:
                losses.append(loss)
        return losses
    
    def _update_coordinators(self):
        """
        Update regional coordinators
        
        Returns:
            List of training losses
        """
        losses = []
        for region_id, coordinator in self.coordinators.items():
            loss = coordinator.replay()
            if loss is not None:
                losses.append(loss)
        return losses
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'episode_{episode}')
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save local agents
        for tl_id, agent in self.local_agents.items():
            agent.save(os.path.join(checkpoint_path, f'local_{tl_id}.pth'))
            
        # Save coordinators
        for region_id, coordinator in self.coordinators.items():
            coordinator.save(os.path.join(checkpoint_path, f'coordinator_{region_id}.pth'))
            
        # Save metrics
        with open(os.path.join(checkpoint_path, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)
                
        logging.info(f"Saved checkpoint: episode_{episode}")
        
    def _plot_metrics(self):
        """Generate and save training visualization plots"""
        plot_dir = os.path.join(self.config['training']['log_dir'], 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        metrics_to_plot = [
            'episode',
            'step',
            'total_waiting_time',
            'total_throughput',
            'average_speed',
            'vehicles_in_network',
            'completed_trips'
        ]
        
        for metric in metrics_to_plot:
            if metric in self.metrics:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics[metric])
                plt.title(f'Training Progress: {metric}')
                plt.xlabel('Episode')
                plt.ylabel(metric)
                plt.grid(True)
                plt.savefig(os.path.join(plot_dir, f'{metric}.png'))
                plt.close()
                
    def _calculate_metrics(self, episode, vehicle_data):
        """
        Calculate metrics for the current episode
        
        Args:
            episode (int): Current episode number
            vehicle_data (list): List of vehicle data from the simulation
        
        Returns:
            dict: Calculated metrics for the episode
        """
        # Start timing the metrics calculation
        episode_start_time = time.time()
        
        # Calculate metrics using evaluation_sets metrics
        metrics = {
            'episode': episode,
            'total_waiting_time': average_waiting_time(vehicle_data),
            'total_throughput': total_throughput(vehicle_data),
            'average_speed': average_speed(vehicle_data),
            'max_waiting_time': max_waiting_time(vehicle_data),
            'vehicles_in_network': len(vehicle_data),
            'completed_trips': traci.simulation.getArrivedNumber(),
            'processing_time': time.time() - episode_start_time
        }
        
        # Log metrics for debugging
        logging.debug(f"Episode {episode} Metrics: {metrics}")
        
        return metrics

    def train(self):
        """
        Train the hierarchical multi-agent traffic control system
        
        Incorporates metrics tracking and logging using the new metrics functions
        """
        # Initialize tracking variables
        all_episode_metrics = []
        
        # Create a progress bar for episodes
        episodes_progress = tqdm(
            range(self.config['simulation']['num_episodes']), 
            desc="Training Progress", 
            unit="episode",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        # Iterate through episodes
        for episode in episodes_progress:
            # Reset environment and get initial states
            states = self.env.reset()
            
            # Initialize episode tracking
            episode_start_time = time.time()
            episode_rewards = defaultdict(float)
            done = {agent: False for agent in states.keys()}
            step = 0
            max_steps = self.config['simulation']['max_steps']
            
            # Create a progress bar for steps within the episode
            steps_progress = tqdm(
                total=max_steps, 
                desc=f"Episode {episode+1}", 
                unit="step",
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
            
            # Training loop for this episode
            while not all(done.values()) and step < max_steps:
                # Prepare actions for each agent
                actions = {}
                for agent_id, agent in self.local_agents.items():
                    if not done.get(agent_id, False):
                        actions[agent_id] = agent.act(states[agent_id])
                
                # Apply actions and get next states
                next_states, rewards, done, _ = self.env.step(actions)
                
                # Update rewards
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward
                
                # Store experiences for each agent
                for agent_id, agent in self.local_agents.items():
                    if not done.get(agent_id, False):
                        agent.remember(
                            states[agent_id], 
                            actions[agent_id], 
                            rewards[agent_id], 
                            next_states[agent_id], 
                            done.get(agent_id, False)
                        )
                
                # Perform experience replay and learning
                for agent in self.local_agents.values():
                    agent.replay()
                
                # Update states
                states = next_states
                step += 1
                
                # Update step progress bar
                steps_progress.update(1)
                
                # Optional: break if all agents are done
                if all(done.values()):
                    break
            
            # Close step progress bar
            steps_progress.close()
            
            # Calculate episode metrics
            vehicle_data = [
                {
                    'id': vid, 
                    'waiting_time': traci.vehicle.getWaitingTime(vid),
                    'speed': traci.vehicle.getSpeed(vid),
                    'route': traci.vehicle.getRoute(vid)
                }
                for vid in traci.vehicle.getIDList()
            ]
            episode_metrics = self._calculate_metrics(episode, vehicle_data)
            episode_metrics['total_rewards'] = sum(episode_rewards.values())
            episode_metrics['total_steps'] = step
            
            # Print current episode metrics
            print(f"\nEpisode {episode + 1} Metrics:")
            print(f"Average Waiting Time: {episode_metrics['total_waiting_time']:.2f} seconds")
            print(f"Total Throughput: {episode_metrics['total_throughput']:.2f} vehicles")
            print(f"Average Speed: {episode_metrics['average_speed']:.2f} km/h")
            print(f"Max Waiting Time: {episode_metrics['max_waiting_time']:.2f} seconds")
            print(f"Vehicles in Network: {episode_metrics['vehicles_in_network']}")
            print(f"Completed Trips: {episode_metrics['completed_trips']}")

            # Save metrics for final summary
            all_episode_metrics.append(episode_metrics)

            # Update episodes progress bar with metrics
            episodes_progress.set_postfix({
                'Waiting Time': f"{episode_metrics['total_waiting_time']:.2f}", 
                'Throughput': f"{episode_metrics['total_throughput']:.2f}",
                'Avg Speed': f"{episode_metrics['average_speed']:.2f}"
            })
            
            # Optional: save models periodically
            if episode % self.config['training'].get('save_frequency', 10) == 0:
                self._save_checkpoint(episode)
            
            # Optional: early stopping or additional logic
            if self._should_stop_training(all_episode_metrics):
                break
        
        # Close episodes progress bar
        episodes_progress.close()
        
        # Print final metrics after all episodes
        if all_episode_metrics:
            print("\nFinal Training Results - Averaged over all episodes:")
            print(f"Average Waiting Time: {np.mean([m['total_waiting_time'] for m in all_episode_metrics]):.2f} ± {np.std([m['total_waiting_time'] for m in all_episode_metrics]):.2f} seconds")
            print(f"Average Total Throughput: {np.mean([m['total_throughput'] for m in all_episode_metrics]):.2f} ± {np.std([m['total_throughput'] for m in all_episode_metrics]):.2f} vehicles")
            print(f"Average Speed: {np.mean([m['average_speed'] for m in all_episode_metrics]):.2f} ± {np.std([m['average_speed'] for m in all_episode_metrics]):.2f} km/h")
            print(f"Average Max Waiting Time: {np.mean([m['max_waiting_time'] for m in all_episode_metrics]):.2f} ± {np.std([m['max_waiting_time'] for m in all_episode_metrics]):.2f} seconds")
        
        return all_episode_metrics
    
    def _should_stop_training(self, all_episode_metrics):
        """
        Determine if training should stop early
        
        Args:
            all_episode_metrics (List[Dict]): Metrics from all episodes
        
        Returns:
            bool: Whether to stop training
        """
        # Example early stopping criteria
        if len(all_episode_metrics) > 10:
            recent_avg_waiting_time = sum(
                episode['total_waiting_time'] for episode in all_episode_metrics[-10:]
            ) / 10
            
            # Stop if waiting time is not improving
            if recent_avg_waiting_time > 1000:  # Adjust threshold as needed
                logging.info("Early stopping due to high waiting times.")
                return True
        
        return False
    
    def save_final_model(self):
        """Save final model"""
        final_model_path = os.path.join(self.checkpoint_dir, 'final_model')
        os.makedirs(final_model_path, exist_ok=True)
        
        # Save local agents
        for tl_id, agent in self.local_agents.items():
            agent.save(os.path.join(final_model_path, f'local_{tl_id}.pth'))
            
        # Save coordinators
        for region_id, coordinator in self.coordinators.items():
            coordinator.save(os.path.join(final_model_path, f'coordinator_{region_id}.pth'))
            
        logging.info("Saved final model.")
        
    def save_metrics(self):
        """Save metrics to file"""
        metrics_file = os.path.join(self.checkpoint_dir, 'training_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
        logging.info("Saved training metrics.")

def main():
    """
    Main function to run hierarchical training
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Hierarchical Traffic Signal Control Training')
    parser.add_argument('-c', '--config', 
                        type=str, 
                        default='config/hierarchical_training_config.yaml', 
                        help='Path to the configuration file')
    parser.add_argument('-l', '--log_level', 
                        type=str, 
                        default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Log configuration file path
    logging.info(f"Using configuration file: {args.config}")
    
    try:
        # Initialize trainer
        trainer = HierarchicalTrainer(args.config)
        
        # Run training
        trainer.train()
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    
    # Optional: Save final model and metrics
    trainer.save_final_model()
    trainer.save_metrics()
    
    logging.info("Training completed successfully.")

if __name__ == '__main__':
    main()