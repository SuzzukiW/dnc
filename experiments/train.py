# experiments/train.py

import os
import sys
import yaml
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.agents import DQNAgent
from src.environment import SUMOEnvironment
from src.utils.logger import setup_logger

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_dqn(env_config, agent_config, num_episodes=1000):
    """Train DQN agent on SUMO environment"""
    
    # Initialize environment
    env = SUMOEnvironment(
        net_file=env_config['net_file'],
        route_file=env_config['route_file'],
        out_csv_name=env_config['out_csv_name'],
        use_gui=env_config['use_gui'],
        num_seconds=env_config['num_seconds'],
        delta_time=env_config['delta_time']
    )
    
    # Initialize agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, agent_config)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path("experiments/logs") / timestamp
    model_dir = Path("experiments/models") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("dqn_training", log_dir / "training.log")
    
    # Save configurations
    with open(log_dir / "env_config.yaml", 'w') as f:
        yaml.dump(env_config, f)
    with open(log_dir / "agent_config.yaml", 'w') as f:
        yaml.dump(agent_config, f)
    
    # Training metrics
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'average_waiting_times': [],
        'episode_losses': []
    }
    
    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_losses = []
        step = 0
        done = False
        
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                episode_losses.append(loss)
            
            state = next_state
            episode_reward += reward
            step += 1
            
            # Update target network every 100 steps
            if step % 100 == 0:
                agent.update_target_network()
        
        # Log episode metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(step)
        metrics['average_waiting_times'].append(-episode_reward/step)  # Convert reward to waiting time
        metrics['episode_losses'].append(np.mean(episode_losses) if episode_losses else 0)
        
        # Log progress
        logger.info(f"Episode {episode+1}/{num_episodes}")
        logger.info(f"Reward: {episode_reward:.2f}")
        logger.info(f"Average Loss: {metrics['episode_losses'][-1]:.4f}")
        logger.info(f"Epsilon: {agent.epsilon:.4f}")
        
        # Save model periodically
        if (episode + 1) % 100 == 0:
            model_path = model_dir / f"dqn_model_episode_{episode+1}.pt"
            agent.save(model_path)
            
            # Save metrics
            metrics_path = log_dir / f"metrics_episode_{episode+1}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
    
    # Save final model and metrics
    agent.save(model_dir / "dqn_model_final.pt")
    with open(log_dir / "metrics_final.json", 'w') as f:
        json.dump(metrics, f)
    
    env.close()
    return metrics

def main():
    # Load configurations
    env_config = load_config('config/env_config.yaml')
    agent_config = load_config('config/agent_config.yaml')
    
    # Train agent
    metrics = train_dqn(env_config, agent_config)
    
    print("Training completed!")

if __name__ == "__main__":
    main()