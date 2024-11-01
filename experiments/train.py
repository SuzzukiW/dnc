# experiments/train.py

import os
import sys
import yaml
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.environment.sumo_env import SUMOEnvironment
from src.agents.dqn_agent import DQNAgent
from src.utils.logger import Logger
from src.utils.data_collector import DataCollector
from src.utils.replay_buffer import ReplayBuffer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_training(env_config, agent_config):
    # Initialize environment
    env = SUMOEnvironment(
        net_file=env_config['net_file'],
        route_file=env_config['route_file'],
        use_gui=env_config.get('use_gui', False),
        num_seconds=env_config.get('num_seconds', 3600),
        yellow_time=env_config.get('yellow_time', 3),
        min_green=env_config.get('min_green', 5),
        max_green=env_config.get('max_green', 50)
    )
    
    # Get traffic light IDs and create agents
    tl_ids = env.get_traffic_light_ids()
    agents = {}
    for tl_id in tl_ids:
        agents[tl_id] = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            learning_rate=agent_config['learning_rate'],
            gamma=agent_config['gamma'],
            epsilon=agent_config['epsilon_start'],
            epsilon_min=agent_config['epsilon_min'],
            epsilon_decay=agent_config['epsilon_decay'],
            memory_size=agent_config['memory_size'],
            batch_size=agent_config['batch_size']
        )
    
    return env, agents

def train(env_config_path='config/env_config.yaml', 
          agent_config_path='config/agent_config.yaml',
          num_episodes=1000):
    
    # Load configurations
    env_config = load_config(env_config_path)
    agent_config = load_config(agent_config_path)
    
    # Setup environment and agents
    env, agents = setup_training(env_config, agent_config)
    
    # Initialize logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = Logger(f"logs/training_{timestamp}")
    
    # Initialize data collector
    data_collector = DataCollector(
        log_dir=f"logs/training_{timestamp}",
        metrics=['episode_reward', 'average_waiting_time', 'throughput']
    )
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Collect actions from all agents
            actions = {}
            for tl_id, agent in agents.items():
                tl_state = state[tl_id]
                actions[tl_id] = agent.get_action(tl_state)
            
            # Execute actions and get next states
            next_state, reward, done, info = env.step(actions)
            
            # Store experiences in replay buffers and train agents
            for tl_id, agent in agents.items():
                agent.store_transition(
                    state[tl_id],
                    actions[tl_id],
                    reward[tl_id],
                    next_state[tl_id],
                    done
                )
                
                if len(agent.memory) > agent.batch_size:
                    agent.train()
            
            state = next_state
            episode_reward += sum(reward.values())
        
        # Collect episode data
        metrics = {
            'episode': episode,
            'episode_reward': episode_reward,
            'average_waiting_time': info.get('average_waiting_time', 0),
            'throughput': info.get('throughput', 0),
            'epsilon': list(agents.values())[0].epsilon  # Track exploration rate
        }
        
        # Log metrics
        data_collector.collect(metrics)
        logger.log_episode(metrics)
        
        # Save model periodically
        if episode % 100 == 0:
            for tl_id, agent in agents.items():
                save_path = f"models/agent_{tl_id}_episode_{episode}.pth"
                agent.save_model(save_path)
        
        print(f"Episode {episode}/{num_episodes} - "
              f"Reward: {episode_reward:.2f} - "
              f"Avg Wait Time: {metrics['average_waiting_time']:.2f} - "
              f"Throughput: {metrics['throughput']}")
    
    # Final save
    for tl_id, agent in agents.items():
        save_path = f"models/agent_{tl_id}_final.pth"
        agent.save_model(save_path)
    
    env.close()
    logger.close()

if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Start training
    train(num_episodes=1000)