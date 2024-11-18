import os
import sys
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.environment.maddpg_env import MultiAgentEnv
from src.agents.maddpg_agent import MADDPGAgent
from src.utils.replay_buffer import ReplayBuffer

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_dummy_multi_agent_env():
    """Create a dummy multi-agent environment for testing"""
    class DummyMultiAgentEnv:
        def __init__(self, num_agents=2, state_dim=4, action_dim=2):
            self.num_agents = num_agents
            self.observation_space = [np.zeros(state_dim) for _ in range(num_agents)]
            self.action_space = [np.zeros(action_dim) for _ in range(num_agents)]
        
        def reset(self):
            states = [np.random.rand(len(self.observation_space[0])) for _ in range(self.num_agents)]
            return states, {}
        
        def step(self, actions):
            next_states = [np.random.rand(len(self.observation_space[0])) for _ in range(self.num_agents)]
            rewards = np.random.rand(self.num_agents)
            dones = np.zeros(self.num_agents, dtype=bool)
            truncated = False
            info = {}
            return next_states, rewards, dones, truncated, info
    
    return DummyMultiAgentEnv()

def train_maddpg(config):
    """Train Multi-Agent Deep Deterministic Policy Gradient (MADDPG)"""
    # Use dummy environment for now
    env = create_dummy_multi_agent_env()
    
    # Get environment details
    num_agents = env.num_agents
    state_size = len(env.observation_space[0])
    action_size = len(env.action_space[0])
    
    # Initialize agents
    agents = [
        MADDPGAgent(
            state_size=state_size, 
            action_size=action_size, 
            num_agents=num_agents, 
            random_seed=config['training']['seed'] + i
        ) for i in range(num_agents)
    ]
    
    # Replay buffer
    replay_buffer = ReplayBuffer(
        buffer_size=config['training']['buffer_size'], 
        batch_size=config['training']['batch_size'], 
        seed=config['training']['seed']
    )
    
    # Training parameters
    n_episodes = config['training']['n_episodes']
    max_t = config['training']['max_timesteps']
    
    # Logging
    scores_window = []
    scores_global = []
    
    # Training loop
    for i_episode in tqdm(range(1, n_episodes + 1)):
        # Reset environment
        states, _ = env.reset()
        
        # Reset agents
        for agent in agents:
            agent.reset()
        
        # Episode tracking
        episode_scores = np.zeros(num_agents)
        
        # Episode steps
        for t in range(max_t):
            # Agent actions
            actions = [
                agent.act(state.reshape(1, -1)).squeeze() 
                for agent, state in zip(agents, states)
            ]
            
            # Environment step
            next_states, rewards, dones, truncated, _ = env.step(actions)
            
            # Store experience in replay buffer
            replay_buffer.add(states, actions, rewards, next_states, dones)
            
            # Update agents if enough samples in buffer
            if len(replay_buffer) > config['training']['batch_size']:
                experiences = replay_buffer.sample()
                
                # Reshape experiences to match multi-agent format
                states = experiences[0].reshape(experiences[0].shape[0], num_agents, -1)
                actions = experiences[1].reshape(experiences[1].shape[0], num_agents, -1)
                rewards = experiences[2].reshape(experiences[2].shape[0], num_agents)
                next_states = experiences[3].reshape(experiences[3].shape[0], num_agents, -1)
                dones = experiences[4].reshape(experiences[4].shape[0], num_agents)
                
                experiences_ma = (states, actions, rewards, next_states, dones)
                
                for agent_idx, agent in enumerate(agents):
                    agent.learn(experiences_ma, agent_idx)
            
            # Update states and scores
            states = next_states
            episode_scores += rewards
            
            # Episode termination
            if np.any(dones) or truncated:
                break
        
        # Store episode scores
        scores_window.append(episode_scores)
        scores_global.append(np.mean(episode_scores))
        
        # Print progress
        if i_episode % config['training']['print_interval'] == 0:
            print(f'Episode {i_episode}: Average Score: {np.mean(scores_global[-config["training"]["print_interval"]:])}')
    
    # Save model
    for idx, agent in enumerate(agents):
        torch.save(agent.actor_local.state_dict(), 
                   f'maddpg_actor_agent_{idx}.pth')
        torch.save(agent.critic_local.state_dict(), 
                   f'maddpg_critic_agent_{idx}.pth')
    
    return scores_global

def main():
    # Load configuration
    config_path = os.path.join(project_root, 'config', 'maddpg.yaml')
    config = load_config(config_path)
    
    # Train MADDPG
    train_maddpg(config)

if __name__ == '__main__':
    main()
