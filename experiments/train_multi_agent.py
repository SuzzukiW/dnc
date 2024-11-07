# experiments/train_multi_agent.py

import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
from datetime import datetime
import logging
import json
from collections import defaultdict
import matplotlib.pyplot as plt

from src.agents import MultiAgentDQN
from src.environment import MultiAgentSumoEnvironment
from src.utils import setup_logger

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_metrics(metrics, save_path):
    """Plot and save training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot average rewards
    axes[0, 0].plot(metrics['mean_rewards'])
    axes[0, 0].set_title('Mean Reward per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    
    # Plot global rewards
    axes[0, 1].plot(metrics['global_rewards'])
    axes[0, 1].set_title('Global Reward per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Global Reward')
    
    # Plot average waiting times
    axes[1, 0].plot(metrics['mean_waiting_times'])
    axes[1, 0].set_title('Mean Waiting Time per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Waiting Time (s)')
    
    # Plot average queue lengths
    axes[1, 1].plot(metrics['mean_queue_lengths'])
    axes[1, 1].set_title('Mean Queue Length per Episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Queue Length')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_multi_agent(env_config, agent_config, num_episodes=1000):
    """Train multiple cooperative DQN agents"""
    
    # Initialize environment
    env = MultiAgentSumoEnvironment(
        net_file=env_config['net_file'],
        route_file=env_config['route_file'],
        out_csv_name=env_config['out_csv_name'],
        use_gui=env_config['use_gui'],
        num_seconds=env_config['num_seconds'],
        delta_time=env_config['delta_time'],
        yellow_time=env_config['yellow_time'],
        min_green=env_config['min_green'],
        max_green=env_config['max_green'],
        neighbor_distance=env_config['neighbor_distance']
    )
    
    # Get traffic light IDs and neighbor map
    traffic_lights = env.traffic_lights
    neighbor_map = env.get_neighbor_map()
    
    # Initialize multi-agent system
    state_size = env.observation_spaces[traffic_lights[0]].shape[0]
    action_size = env.action_spaces[traffic_lights[0]].n
    multi_agent_system = MultiAgentDQN(
        state_size=state_size,
        action_size=action_size,
        agent_ids=traffic_lights,
        neighbor_map=neighbor_map,
        config=agent_config
    )
    
    # Setup logging with proper directory creation
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path("experiments/logs") / f"multi_agent_{timestamp}"
    model_dir = Path("experiments/models") / f"multi_agent_{timestamp}"
    
    # Create directories and their parents
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure final models directory exists
    (model_dir / "final_models").mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("multi_agent_training", log_dir / "training.log")
    
    # Save configurations
    with open(log_dir / "env_config.yaml", 'w') as f:
        yaml.dump(env_config, f)
    with open(log_dir / "agent_config.yaml", 'w') as f:
        yaml.dump(agent_config, f)
    
    # Training metrics
    metrics = {
        'mean_rewards': [],
        'global_rewards': [],
        'mean_waiting_times': [],
        'mean_queue_lengths': [],
        'agent_rewards': defaultdict(list)
    }
    
    # Training loop
    try:
        for episode in range(num_episodes):
            states, _ = env.reset()
            episode_rewards = defaultdict(float)
            episode_steps = 0
            done = False
            
            while not done:
                # Get actions for all agents
                actions = multi_agent_system.act(states)
                
                # Execute actions
                next_states, rewards, done, _, info = env.step(actions)
                
                # Update agents
                losses = multi_agent_system.step(
                    states, actions, rewards, next_states, 
                    {agent_id: done for agent_id in traffic_lights},
                    global_reward=info['global_reward']
                )
                
                # Update metrics
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward
                
                states = next_states
                episode_steps += 1
                
                # Log step information
                if episode_steps % 100 == 0:
                    logger.info(f"Episode {episode + 1}, Step {episode_steps}")
                    logger.info(f"Global reward: {info['global_reward']:.2f}")
                    logger.info(f"Average waiting time: {info['total_waiting_time']:.2f}")
                    
            # Calculate episode metrics
            mean_episode_reward = np.mean([r for r in episode_rewards.values()])
            
            # Update metrics
            metrics['mean_rewards'].append(mean_episode_reward)
            metrics['global_rewards'].append(info['global_reward'])
            metrics['mean_waiting_times'].append(info['total_waiting_time'])
            
            for agent_id, reward in episode_rewards.items():
                metrics['agent_rewards'][agent_id].append(reward)
            
            # Log episode results
            logger.info(f"Episode {episode + 1}/{num_episodes} completed")
            logger.info(f"Mean episode reward: {mean_episode_reward:.2f}")
            logger.info(f"Global reward: {info['global_reward']:.2f}")
            logger.info(f"Total waiting time: {info['total_waiting_time']:.2f}")
            logger.info("-" * 50)
            
            # Save models periodically
            if (episode + 1) % 100 == 0:
                # Create directory before saving
                save_dir = model_dir / f"episode_{episode + 1}"
                save_dir.mkdir(parents=True, exist_ok=True)
                multi_agent_system.save_agents(save_dir)
                
                # Plot and save metrics
                plot_metrics(metrics, log_dir / f"metrics_episode_{episode + 1}.png")
                
                # Save metrics
                metrics_path = log_dir / f"metrics_episode_{episode + 1}.json"
                with open(metrics_path, 'w') as f:
                    json.dump({
                        'mean_rewards': metrics['mean_rewards'],
                        'global_rewards': metrics['global_rewards'],
                        'mean_waiting_times': metrics['mean_waiting_times'],
                        'mean_queue_lengths': metrics['mean_queue_lengths'],
                        'agent_rewards': {k: v for k, v in metrics['agent_rewards'].items()}
                    }, f)
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        # Ensure directory exists before saving final models
        final_model_dir = model_dir / "final_models"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        multi_agent_system.save_agents(final_model_dir)
        
        plot_metrics(metrics, log_dir / "final_metrics.png")
        
        with open(log_dir / "final_metrics.json", 'w') as f:
            json.dump({
                'mean_rewards': metrics['mean_rewards'],
                'global_rewards': metrics['global_rewards'],
                'mean_waiting_times': metrics['mean_waiting_times'],
                'mean_queue_lengths': metrics['mean_queue_lengths'],
                'agent_rewards': {k: v for k, v in metrics['agent_rewards'].items()}
            }, f)
        
        env.close()
        
    return metrics

def main():
    # Load configurations
    env_config = load_config('config/env_config.yaml')
    agent_config = load_config('config/agent_config.yaml')
    
    # Add multi-agent specific configurations
    env_config.update({
        'neighbor_distance': 100,  # meters
        'yellow_time': 2,
        'min_green': 5,
        'max_green': 50
    })
    
    agent_config.update({
        'memory_size': 100000,
        'communication_mode': 'shared_experience',  # Can be 'none', 'shared_experience', 'full_state', 'hierarchical'
        'reward_type': 'hybrid'  # Can be 'local', 'global', 'hybrid', 'emissions'
    })
    
    # Train agents
    metrics = train_multi_agent(env_config, agent_config)
    
    print("Training completed!")

if __name__ == "__main__":
    main()