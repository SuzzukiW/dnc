# experiments/scenarios/communication/independent_agents.py

import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.agents.dqn_agent import DQNAgent
from src.environment.multi_agent_sumo_env import MultiAgentSumoEnvironment
from src.utils.logger import setup_logger

class IndependentAgents:
    """
    Independent agents scenario - no communication between agents
    Each agent learns independently using its own experience
    """
    def __init__(self, env_config, agent_config):
        self.env = MultiAgentSumoEnvironment(**env_config)
        self.traffic_lights = self.env.traffic_lights
        
        # Create independent DQN agent for each traffic light
        self.agents = {}
        for tl_id in self.traffic_lights:
            state_size = self.env.observation_spaces[tl_id].shape[0]
            action_size = self.env.action_spaces[tl_id].n
            self.agents[tl_id] = DQNAgent(state_size, action_size, agent_config)
    
    def train(self, num_episodes, log_dir):
        """Train independent agents"""
        logger = setup_logger("independent_training", log_dir / "training.log")
        metrics = defaultdict(list)
        
        for episode in range(num_episodes):
            states, _ = self.env.reset()
            episode_rewards = defaultdict(float)
            done = False
            
            while not done:
                # Each agent selects action independently
                actions = {}
                for tl_id in self.traffic_lights:
                    actions[tl_id] = self.agents[tl_id].act(states[tl_id])
                
                # Environment step
                next_states, rewards, done, _, info = self.env.step(actions)
                
                # Independent learning
                for tl_id in self.traffic_lights:
                    # Store experience in agent's own memory
                    self.agents[tl_id].remember(
                        states[tl_id],
                        actions[tl_id],
                        rewards[tl_id],
                        next_states[tl_id],
                        done
                    )
                    
                    # Individual learning step
                    if len(self.agents[tl_id].memory) > self.agents[tl_id].batch_size:
                        loss = self.agents[tl_id].replay()
                        metrics[f'loss_{tl_id}'].append(loss)
                    
                    episode_rewards[tl_id] += rewards[tl_id]
                
                states = next_states
            
            # Log episode results
            mean_reward = np.mean([r for r in episode_rewards.values()])
            metrics['mean_rewards'].append(mean_reward)
            metrics['global_rewards'].append(info['global_reward'])
            
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            logger.info(f"Mean reward: {mean_reward:.2f}")
            logger.info(f"Global reward: {info['global_reward']:.2f}")
            
            # Save models periodically
            if (episode + 1) % 100 == 0:
                self.save_agents(log_dir / f"models_episode_{episode + 1}")
        
        return metrics
    
    def save_agents(self, save_dir):
        """Save all agents"""
        save_dir.mkdir(parents=True, exist_ok=True)
        for tl_id, agent in self.agents.items():
            agent.save(save_dir / f"agent_{tl_id}.pt")
    
    def load_agents(self, load_dir):
        """Load all agents"""
        for tl_id, agent in self.agents.items():
            agent.load(load_dir / f"agent_{tl_id}.pt")
    
    def close(self):
        """Clean up environment"""
        self.env.close()

def run_independent_scenario(env_config, agent_config, num_episodes=1000):
    """Run independent agents scenario"""
    # Setup logging directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path("experiments/logs/independent_agents") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configurations
    with open(log_dir / "env_config.yaml", 'w') as f:
        yaml.dump(env_config, f)
    with open(log_dir / "agent_config.yaml", 'w') as f:
        yaml.dump(agent_config, f)
    
    # Run training
    scenario = IndependentAgents(env_config, agent_config)
    try:
        metrics = scenario.train(num_episodes, log_dir)
        # Save final models and metrics
        scenario.save_agents(log_dir / "final_models")
        return metrics
    finally:
        scenario.close()

if __name__ == "__main__":
    # Load configurations
    with open("config/env_config.yaml", 'r') as f:
        env_config = yaml.safe_load(f)
    with open("config/agent_config.yaml", 'r') as f:
        agent_config = yaml.safe_load(f)
    
    # Run scenario
    metrics = run_independent_scenario(env_config, agent_config)
    print("Independent agents training completed!")