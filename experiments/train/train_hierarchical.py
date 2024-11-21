#!/usr/bin/env python3
# experiments/train/train_hierarchical.py

import os
import sys
from pathlib import Path
import argparse
import yaml
import torch
import numpy as np
from datetime import datetime
import logging
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import traci

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def get_state_space_size(config):
    """Get the size of state space.
    
    Args:
        config (dict): Environment configuration
        
    Returns:
        int: Size of state space
    """
    # Default state space includes:
    # - Queue length for each lane
    # - Waiting time for each lane
    # - Average speed for each lane
    max_lanes = config.get('max_lanes', 20)  # Maximum number of lanes per intersection
    return max_lanes * 3  # 3 features per lane

def get_action_space_size(config):
    """Get the size of action space.
    
    Args:
        config (dict): Environment configuration
        
    Returns:
        int: Size of action space
    """
    # Default action space is the number of phases
    return config.get('num_phases', 4)  # Default to 4 phases

def json_serialize(obj):
    """Helper function to serialize NumPy types for JSON"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                       np.int16, np.int32, np.int64, np.uint8,
                       np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def load_config(config_path):
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

from src.environment.multi_agent_sumo_env_hierarchical import MultiAgentSumoEnvironmentHierarchical
from experiments.scenarios.communication.hierarchical import HierarchicalManager
from src.utils.logging import setup_logger

def train_hierarchical(config_path):
    """
    Train hierarchical multi-agent system with enhanced coordination and learning
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Set default values for training if not present
    if 'training' not in config:
        config['training'] = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'target_update': 10,
            'max_steps': 50000,
            'eval_interval': 1000,
            'max_episode_steps': 3600
        }
        
    train_config = config['training']
    env_config = config['environment']
    log_config = config.get('logging', {
        'log_dir': 'logs',
        'metrics_interval': 5,
        'save_model_interval': 10,
        'training_log_interval': 100
    })
    
    # Setup logging
    log_dir = Path(log_config['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_level = getattr(logging, log_config.get('log_level', 'INFO').upper())
    logger = setup_logger('hierarchical_training', 
                         log_file=str(log_dir / 'training.log'),
                         level=log_level)
    
    # Initialize environment
    filtered_env_config = {'config': env_config}
    env = MultiAgentSumoEnvironmentHierarchical(**filtered_env_config)
    
    # Get actual state size from environment
    sample_state, _ = env.reset()  # Unpack the tuple returned by reset()
    first_agent_id = next(iter(sample_state.keys()))
    state_size = sample_state[first_agent_id].shape[0]  # Get size of first agent's state
    
    # Initialize learning parameters
    agent_config = config['agent']
    lr = float(agent_config['learning_rate'])
    lr_decay = agent_config.get('lr_decay', True)
    lr_decay_rate = float(agent_config.get('lr_decay_rate', 0.995))
    lr_min = float(agent_config.get('lr_min', 1e-5))
    
    epsilon = float(agent_config['epsilon_start'])
    epsilon_min = float(agent_config['epsilon_min'])
    epsilon_decay = float(agent_config['epsilon_decay'])
    
    # Initialize hierarchical manager with updated parameters
    manager = HierarchicalManager(
        state_size=state_size,
        action_size=get_action_space_size(env_config),
        net_file=env_config['net_file'],
        config=config
    )
    
    # Training metrics
    metrics = {
        'episode_rewards': [],
        'avg_waiting_times': [],
        'coordination_rates': [],
        'learning_rates': [],
        'epsilons': []
    }
    
    # Training loop
    total_steps = 0
    episode = 0
    states, _ = env.reset()
    
    # Register all agents with the manager
    for agent_id in states.keys():
        manager.add_agent(agent_id)
    
    while total_steps < train_config['max_steps']:
        episode += 1
        episode_rewards = defaultdict(float)
        episode_steps = 0
        done = False
        coordination_count = 0
        total_interactions = 0
        
        while not done and episode_steps < train_config.get('max_episode_steps', 1000):
            # Convert states to tensors
            states_tensor = {agent_id: torch.FloatTensor(state).unsqueeze(0) 
                           for agent_id, state in states.items()}
            
            # Update agent positions and regions
            agent_positions = env.get_agent_positions()
            manager.update_regions(agent_positions)
            
            # Get actions using current epsilon for exploration
            for agent_id, agent in manager.agents.items():
                agent.epsilon = epsilon  # Update epsilon for each agent
            actions = manager.step(states_tensor, training=True)
            
            # Take environment step
            next_states, rewards, dones, truncated, info = env.step(actions)
            
            # Process communications and update coordination metrics
            coord_info = manager.process_communications()
            if coord_info:
                coordination_count += coord_info['coordinated_agents']
                total_interactions += coord_info['total_interactions']
            
            # Update rewards and states
            for agent_id, reward in rewards.items():
                episode_rewards[agent_id] += reward
            
            states = next_states
            episode_steps += 1
            done = all(dones.values())
            
            # Update learning rate and epsilon
            if lr_decay:
                lr = max(lr * lr_decay_rate, lr_min)
                manager.update_learning_rate(lr)
            
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            
            total_steps += 1
            
            # Log metrics periodically
            if total_steps % log_config.get('metrics_interval', 100) == 0:
                avg_reward = sum(episode_rewards.values()) / len(episode_rewards)
                avg_waiting_time = np.mean([info[aid].get('waiting_time', 0) 
                                          for aid in info]) if info else 0
                coord_rate = (coordination_count / total_interactions 
                            if total_interactions > 0 else 0)
                
                metrics['episode_rewards'].append(avg_reward)
                metrics['avg_waiting_times'].append(avg_waiting_time)
                metrics['coordination_rates'].append(coord_rate)
                metrics['learning_rates'].append(lr)
                metrics['epsilons'].append(epsilon)
                
                logger.info(f"\nStep {total_steps} Metrics:")
                logger.info(f"Episode: {episode}")
                logger.info(f"Average Reward: {avg_reward:.4f}")
                logger.info(f"Average Waiting Time: {avg_waiting_time:.2f}s")
                logger.info(f"Coordination Rate: {coord_rate:.4f}")
                logger.info(f"Learning Rate: {lr:.6f}")
                logger.info(f"Epsilon: {epsilon:.4f}")
            
            # Save model periodically
            if total_steps % log_config.get('save_model_interval', 5000) == 0:
                manager.save_model(str(log_dir / f'model_step_{total_steps}.pt'))
                
                # Save training metrics
                with open(log_dir / 'training_metrics.json', 'w') as f:
                    json.dump(metrics, f, default=json_serialize)
        
        # End of episode logging
        if episode % log_config.get('training_log_interval', 1) == 0:
            logger.info(f"\nEpisode {episode} completed:")
            logger.info(f"Steps: {episode_steps}")
            logger.info(f"Total Steps: {total_steps}")
            logger.info(f"Average Episode Reward: {sum(episode_rewards.values()) / len(episode_rewards):.4f}")
    
    # Save final model and metrics
    manager.save_model(str(log_dir / 'model_final.pt'))
    with open(log_dir / 'final_metrics.json', 'w') as f:
        json.dump(metrics, f, default=json_serialize)
    
    return metrics

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train hierarchical traffic control system')
    parser.add_argument('--config', type=str, 
                      default=str(project_root / 'config' / 'hierarchical_config.yaml'),
                      help='Path to config file')
    args = parser.parse_args()
    train_hierarchical(args.config)

if __name__ == "__main__":
    main()