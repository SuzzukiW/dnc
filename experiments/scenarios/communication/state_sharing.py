# experiments/scenarios/communication/state_sharing.py

import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
import random

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.agents.dqn_agent import DQNNetwork
from src.environment.multi_agent_sumo_env import MultiAgentSumoEnvironment
from src.utils.logger import setup_logger

class StateShareNetwork(nn.Module):
    """Neural network that processes both local and neighbor states"""
    def __init__(self, local_state_size, neighbor_state_size, action_size, hidden_size=64):
        super(StateShareNetwork, self).__init__()
        
        # Process local state
        self.local_net = nn.Sequential(
            nn.Linear(local_state_size, hidden_size),
            nn.ReLU()
        )
        
        # Process neighbor states
        self.neighbor_net = nn.Sequential(
            nn.Linear(neighbor_state_size, hidden_size),
            nn.ReLU()
        )
        
        # Combine local and neighbor information
        self.combine_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, local_state, neighbor_states):
        local_features = self.local_net(local_state)
        neighbor_features = self.neighbor_net(neighbor_states)
        combined = torch.cat([local_features, neighbor_features], dim=1)
        return self.combine_net(combined)

class StateShareAgent:
    """Agent that shares state information with neighbors"""
    def __init__(self, agent_id, state_size, action_size, neighbor_ids, config):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.neighbor_ids = neighbor_ids
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.95)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 32)
        self.memory_size = config.get('memory_size', 10000)
        
        # Calculate neighbor state size (sum of all neighbor states)
        self.neighbor_state_size = state_size * len(neighbor_ids) if neighbor_ids else state_size
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = StateShareNetwork(
            state_size, self.neighbor_state_size, action_size).to(self.device)
        self.target_net = StateShareNetwork(
            state_size, self.neighbor_state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Replay memory
        self.memory = deque(maxlen=self.memory_size)
        self.Transition = namedtuple('Transition',
                                   ('local_state', 'neighbor_states', 'action', 
                                    'reward', 'next_local_state', 'next_neighbor_states', 'done'))
    
    def get_neighbor_states(self, all_states):
        """Combine states of neighboring agents"""
        if not self.neighbor_ids:
            return torch.zeros(1, self.neighbor_state_size).to(self.device)
        
        neighbor_states = []
        for neighbor_id in self.neighbor_ids:
            if neighbor_id in all_states:
                neighbor_states.append(all_states[neighbor_id])
        
        if not neighbor_states:
            return torch.zeros(1, self.neighbor_state_size).to(self.device)
        
        return torch.FloatTensor(np.concatenate(neighbor_states)).to(self.device)
    
    def remember(self, local_state, neighbor_states, action, reward, 
                next_local_state, next_neighbor_states, done):
        """Store transition in memory"""
        self.memory.append(self.Transition(
            local_state, neighbor_states, action, reward,
            next_local_state, next_neighbor_states, done))
    
    def act(self, local_state, neighbor_states, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            local_state = torch.FloatTensor(local_state).unsqueeze(0).to(self.device)
            neighbor_states = torch.FloatTensor(neighbor_states).unsqueeze(0).to(self.device)
            q_values = self.policy_net(local_state, neighbor_states)
            return q_values.argmax().item()
    
    def replay(self):
        """Train on batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return
        
        transitions = random.sample(self.memory, self.batch_size)
        batch = self.Transition(*zip(*transitions))
        
        # Prepare batch tensors
        local_state_batch = torch.FloatTensor(batch.local_state).to(self.device)
        neighbor_states_batch = torch.FloatTensor(batch.neighbor_states).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_local_state_batch = torch.FloatTensor(batch.next_local_state).to(self.device)
        next_neighbor_states_batch = torch.FloatTensor(batch.next_neighbor_states).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(
            local_state_batch, neighbor_states_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute next Q values
        next_q_values = self.target_net(
            next_local_state_batch, next_neighbor_states_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        """Save model weights"""
        torch.save({
            'agent_id': self.agent_id,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class StateShareScenario:
    """
    State Sharing scenario - agents share state and action information
    with their neighbors for better coordination
    """
    def __init__(self, env_config, agent_config):
        self.env = MultiAgentSumoEnvironment(**env_config)
        self.traffic_lights = self.env.traffic_lights
        self.neighbor_map = self.env.get_neighbor_map()
        
        # Create agents with state sharing capability
        self.agents = {}
        for tl_id in self.traffic_lights:
            state_size = self.env.observation_spaces[tl_id].shape[0]
            action_size = self.env.action_spaces[tl_id].n
            neighbor_ids = self.neighbor_map.get(tl_id, [])
            
            self.agents[tl_id] = StateShareAgent(
                agent_id=tl_id,
                state_size=state_size,
                action_size=action_size,
                neighbor_ids=neighbor_ids,
                config=agent_config
            )
    
    def train(self, num_episodes, log_dir):
        """Train agents with state sharing"""
        logger = setup_logger("state_sharing_training", log_dir / "training.log")
        metrics = defaultdict(list)
        
        for episode in range(num_episodes):
            states, _ = self.env.reset()
            episode_rewards = defaultdict(float)
            done = False
            
            while not done:
                # Each agent selects action based on local and neighbor states
                actions = {}
                for tl_id in self.traffic_lights:
                    neighbor_states = self.agents[tl_id].get_neighbor_states(states)
                    actions[tl_id] = self.agents[tl_id].act(states[tl_id], neighbor_states)
                
                # Environment step
                next_states, rewards, done, _, info = self.env.step(actions)
                
                # Update agents
                for tl_id in self.traffic_lights:
                    agent = self.agents[tl_id]
                    
                    # Get current and next neighbor states
                    current_neighbor_states = agent.get_neighbor_states(states)
                    next_neighbor_states = agent.get_neighbor_states(next_states)
                    
                    # Store experience
                    agent.remember(
                        states[tl_id], current_neighbor_states,
                        actions[tl_id], rewards[tl_id],
                        next_states[tl_id], next_neighbor_states,
                        done
                    )
                    
                    # Learning step
                    if len(agent.memory) > agent.batch_size:
                        loss = agent.replay()
                        metrics[f'loss_{tl_id}'].append(loss)
                    
                    episode_rewards[tl_id] += rewards[tl_id]
                
                states = next_states
            
            # Update target networks periodically
            if episode % 10 == 0:
                for agent in self.agents.values():
                    agent.update_target_network()
            
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

def run_state_sharing_scenario(env_config, agent_config, num_episodes=1000):
    """Run state sharing scenario"""
    # Setup logging directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path("experiments/logs/state_sharing") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configurations
    with open(log_dir / "env_config.yaml", 'w') as f:
        yaml.dump(env_config, f)
    with open(log_dir / "agent_config.yaml", 'w') as f:
        yaml.dump(agent_config, f)
    
    # Run training
    scenario = StateShareScenario(env_config, agent_config)
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
    metrics = run_state_sharing_scenario(env_config, agent_config)
    print("State sharing training completed!")