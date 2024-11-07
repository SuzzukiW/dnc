# quick_tests/test_single_agent.py

import os
import sys
from pathlib import Path
import yaml
import torch

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.agents import DQNAgent
from src.environment import MultiAgentSumoEnvironment
from src.utils.logger import setup_logger

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_single_agent_training(env_config, agent_config, num_episodes=10):
    """Run a quick test training for single traffic light"""
    
    # Initialize environment with modified config for single agent
    env = MultiAgentSumoEnvironment(
        net_file=env_config['net_file'],
        route_file=env_config['route_file'],
        out_csv_name="data/test-1/single_agent_test.csv",
        use_gui=True,  # Set to True to visualize training
        num_seconds=1000,  # Reduced simulation time for testing
        delta_time=5,
        yellow_time=2,
        min_green=5,
        max_green=50
    )
    
    # Select single traffic light for testing
    traffic_lights = env.traffic_lights
    test_traffic_light = traffic_lights[0]  # Use the first traffic light
    print(f"Training with traffic light: {test_traffic_light}")
    
    # Initialize agent
    state_size = env.observation_spaces[test_traffic_light].shape[0]
    action_size = env.action_spaces[test_traffic_light].n
    agent = DQNAgent(state_size, action_size, agent_config)
    
    # Setup logging
    logger = setup_logger("single_agent_test", "experiments/logs/single_agent_test.log")
    
    # Training loop
    for episode in range(num_episodes):
        states, _ = env.reset()
        state = states[test_traffic_light]
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            # Select action for single traffic light
            action = agent.act(state)
            
            # Create actions dict with only our traffic light
            actions = {test_traffic_light: action}
            
            # Take step in environment
            next_states, rewards, done, _, info = env.step(actions)
            next_state = next_states[test_traffic_light]
            reward = rewards[test_traffic_light]
            
            # Store transition and train agent
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                logger.info(f"Episode {episode+1}, Step {step}, Loss: {loss:.4f}")
            
            state = next_state
            episode_reward += reward
            step += 1
            
            # Update target network periodically
            if step % 100 == 0:
                agent.update_target_network()
        
        # Log episode results
        logger.info(f"Episode {episode+1}/{num_episodes}")
        logger.info(f"Total Reward: {episode_reward:.2f}")
        logger.info(f"Average Reward: {episode_reward/step:.2f}")
        logger.info(f"Epsilon: {agent.epsilon:.4f}")
        logger.info("-" * 50)
    
    env.close()
    return agent

if __name__ == "__main__":
    # Load configurations
    env_config = load_config('config/env_config.yaml')
    agent_config = load_config('config/agent_config.yaml')
    
    # Modify paths for Version1
    env_config['net_file'] = "Version1/2024-11-05-18-42-37/osm.net.xml.gz"
    env_config['route_file'] = "Version1/2024-11-05-18-42-37/osm.passenger.trips.xml"
    
    # Run test training
    agent = test_single_agent_training(env_config, agent_config)
    
    # Save trained agent
    save_path = Path("experiments/models/single_agent_test.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(save_path)
    
    print("Test training completed!")
    print(f"Model saved to: {save_path}")