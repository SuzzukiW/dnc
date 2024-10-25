# experiments/train.py

import os
import sys
import yaml
import torch
import numpy as np
import random
from pathlib import Path
from datetime import datetime

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.environment.sumo_env import SumoEnvironment
from src.agents.dqn_agent import DQNAgent
from src.utils.logger import Logger

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train():
    """Main training loop"""
    # Set random seed
    set_seed(42)
    
    # Load configurations
    config_dir = os.path.join(project_root, 'config')
    with open(os.path.join(config_dir, 'env_config.yaml'), 'r') as f:
        env_config = yaml.safe_load(f)
    with open(os.path.join(config_dir, 'agent_config.yaml'), 'r') as f:
        agent_config = yaml.safe_load(f)
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(project_root, 'logs', f'training_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize logger
    logger = Logger(log_dir)
    logger.save_config({**env_config, **agent_config})  # Save configurations
    
    # Create environment
    env = SumoEnvironment(
        config_file=os.path.join(project_root, env_config['sumo']['config_file']),
        use_gui=env_config['sumo']['gui'],
        num_seconds=env_config['sumo']['simulation_steps']
    )
    
    # Create agents for each traffic light
    agents = {}
    for tl_id in env.traffic_lights:
        state_dim = env.observation_spaces[tl_id].shape[0]
        action_dim = env.action_spaces[tl_id].n
        
        agents[tl_id] = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **agent_config['dqn']
        )
    
    logger.log_message(f"Starting training with {len(agents)} agents")
    
    try:
        # Training loop
        for episode in range(agent_config['training']['episodes']):
            states = env.reset()[0]  # Get initial states (updated for gymnasium)
            episode_rewards = {tl_id: 0 for tl_id in env.traffic_lights}
            
            for step in range(agent_config['training']['max_steps']):
                # Select actions
                actions = {
                    tl_id: agents[tl_id].select_action(states[tl_id])
                    for tl_id in env.traffic_lights
                }
                
                # Execute actions
                next_states, rewards, dones, _, _ = env.step(actions)  # Updated for gymnasium
                
                # Store experiences and train
                step_metrics = {
                    'queue_length': {},
                    'waiting_time': {},
                    'loss': 0
                }
                
                for tl_id in env.traffic_lights:
                    # Store experience
                    agents[tl_id].remember(
                        states[tl_id],
                        actions[tl_id],
                        rewards[tl_id],
                        next_states[tl_id],
                        dones[tl_id]
                    )
                    
                    # Train agent
                    loss = agents[tl_id].train()
                    episode_rewards[tl_id] += rewards[tl_id]
                    
                    # Collect metrics
                    step_metrics['loss'] += loss
                    step_metrics['queue_length'][tl_id] = len(env.traffic_lights)  # Placeholder
                    step_metrics['waiting_time'][tl_id] = -rewards[tl_id]  # Using negative reward as waiting time
                
                # Log step metrics
                step_metrics['loss'] /= len(agents)  # Average loss across agents
                step_metrics['epsilon'] = agents[list(agents.keys())[0]].epsilon
                logger.log_step(step_metrics)
                
                # Update states
                states = next_states
                
                # Check if episode is done
                if any(dones.values()):
                    break
            
            # Log episode results
            logger.log_episode(episode, episode_rewards)
            
            # Save models periodically
            if episode % agent_config['training']['save_interval'] == 0:
                save_dir = os.path.join(log_dir, f'episode_{episode}')
                os.makedirs(save_dir, exist_ok=True)
                
                for tl_id, agent in agents.items():
                    save_path = os.path.join(save_dir, f'agent_{tl_id}.pt')
                    agent.save(save_path)
                
                logger.log_message(f"Saved models at episode {episode}")
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = sum(episode_rewards.values()) / len(episode_rewards)
                print(f"Episode {episode}/{agent_config['training']['episodes']}, "
                      f"Average Reward: {avg_reward:.2f}, "
                      f"Epsilon: {agents[list(agents.keys())[0]].epsilon:.3f}")
    
    except Exception as e:
        logger.log_message(f"Error during training: {str(e)}")
        raise e
    
    finally:
        # Close environment
        env.close()
        logger.log_message("Training completed")

if __name__ == '__main__':
    train()