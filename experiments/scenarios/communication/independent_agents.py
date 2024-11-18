# experiments/scenarios/communication/independent_agents.py
# Part I: Core independent agent components

import os
import sys
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import random
import logging
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class IndependentMemory:
    """Memory buffer for individual agents"""
    
    def __init__(self, capacity: int):
        """Initialize replay memory"""
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.position = 0
        
        # Performance tracking
        self.insertion_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
    def push(self, state: np.ndarray, 
             action: int, 
             reward: float,
             next_state: np.ndarray, 
             done: bool):
        """Store transition in memory"""
        self.memory.append((state, action, reward, next_state, done))
        self.insertion_count += 1
        
        # Track rewards
        self.episode_rewards.append(reward)
        if done and self.episode_rewards:
            self.episode_lengths.append(len(self.episode_rewards))
            self.episode_rewards = []
            
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample random batch of transitions"""
        return random.sample(self.memory, min(batch_size, len(self)))
    
    def get_statistics(self) -> Dict[str, float]:
        """Get memory statistics"""
        return {
            'memory_utilization': len(self) / self.capacity,
            'total_insertions': self.insertion_count,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'max_episode_length': max(self.episode_lengths) if self.episode_lengths else 0
        }
    
    def __len__(self) -> int:
        return len(self.memory)
        
class IndependentNetwork(nn.Module):
    """Neural network for independent learning agent"""
    
    def __init__(self,
                input_size: int,
                output_size: int,
                hidden_size: int = 64):
        """Initialize network"""
        super(IndependentNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Performance tracking
        self.training_steps = 0
        self.loss_history = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get network statistics"""
        return {
            'training_steps': self.training_steps,
            'avg_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            'min_loss': min(self.loss_history) if self.loss_history else 0,
            'max_loss': max(self.loss_history) if self.loss_history else 0
        }

class IndependentAgent:
    """Individual learning agent without communication"""
    
    def __init__(self,
                agent_id: str,
                state_size: int,
                action_size: int,
                config: dict):
        """Initialize independent agent"""
        self.id = agent_id
        self.state_size = state_size 
        self.action_size = action_size
        self.config = config
        
        # Learning parameters
        self.gamma = config.get('gamma', 0.95)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.target_update = config.get('target_update', 10)
        self.tau = config.get('tau', 0.001)  # For soft updates
        
        # Exploration parameters
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        
        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = IndependentNetwork(
            state_size,
            action_size,
            config.get('hidden_size', 64)
        ).to(self.device)
        
        self.target_net = IndependentNetwork(
            state_size, 
            action_size,
            config.get('hidden_size', 64)
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )
        
        # Initialize memory
        self.memory = IndependentMemory(
            config.get('memory_size', 10000)
        )
        
        # Performance tracking
        self.episode_rewards = []
        self.training_steps = 0
        self.updates = 0
        self.action_counts = defaultdict(int)
        
        # Set up logging
        self.logger = logging.getLogger(f'Agent_{agent_id}')
        self.logger.setLevel(logging.INFO)
        
        # Training mode
        self.training = True
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        if self.training and random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                action = q_values.argmax().item()
        
        # Track action selection
        self.action_counts[action] += 1
        return action
    
    def store_transition(self, 
                        state: np.ndarray,
                        action: int,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool):
        """Store transition in memory"""
        self.memory.push(state, action, reward, next_state, done)
        
        if done:
            self.episode_rewards.append(reward)
    
    def update(self) -> Optional[float]:
        """Update policy network"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample transitions
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(batch[0]).to(self.device)
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(batch[3]).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)
        
        # Calculate current Q values
        current_q = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Calculate target Q values (double DQN)
        with torch.no_grad():
            next_state_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_q = self.target_net(next_state_batch).gather(1, next_state_actions)
            target_q = reward_batch.unsqueeze(1) + self.gamma * next_q * (1 - done_batch.unsqueeze(1))
        
        # Calculate loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Track metrics
        loss_value = loss.item()
        self.policy_net.loss_history.append(loss_value)
        self.training_steps += 1
        
        # Update target network if needed
        if self.training_steps % self.target_update == 0:
            self._update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min,
                         self.epsilon * self.epsilon_decay)
        
        return loss_value

# Part II: Manager class and supporting functionality
    
    def _update_target_network(self):
        """Update target network using soft update"""
        if self.tau < 1:
            # Soft update
            for target_param, policy_param in zip(
                    self.target_net.parameters(),
                    self.policy_net.parameters()):
                target_param.data.copy_(
                    self.tau * policy_param.data +
                    (1 - self.tau) * target_param.data
                )
        else:
            # Hard update
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )
        
        self.updates += 1
    
    def get_statistics(self) -> Dict[str, float]:
        """Get agent statistics"""
        # Calculate action distribution
        total_actions = sum(self.action_counts.values())
        action_dist = {
            f'action_{a}': count/max(total_actions, 1)
            for a, count in self.action_counts.items()
        }
        
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'training_steps': self.training_steps,
            'network_updates': self.updates,
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'action_distribution': action_dist,
            **self.memory.get_statistics(),
            **self.policy_net.get_statistics()
        }
    
    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'updates': self.updates,
            'statistics': self.get_statistics()
        }, path)
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.updates = checkpoint['updates']

class IndependentAgentManager:
    """Manager class for multiple independent learning agents"""
    
    def __init__(self,
                state_size: int,
                action_size: int,
                num_agents: int,
                config: dict):
        """Initialize agent manager"""
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.config = config
        
        # Initialize agents
        self.agents = {
            f'agent_{i}': IndependentAgent(
                f'agent_{i}',
                state_size,
                action_size,
                config
            ) for i in range(num_agents)
        }
        
        # Performance tracking
        self.episode_metrics = defaultdict(list)
        self.global_metrics = defaultdict(list)
        self.step_counter = 0
        
        # Set up logging
        self.logger = logging.getLogger('AgentManager')
        self.logger.setLevel(logging.INFO)
        
    def select_actions(self, states: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Select actions for all agents"""
        actions = {}
        for agent_id, state in states.items():
            if agent_id in self.agents:
                actions[agent_id] = self.agents[agent_id].select_action(state)
        return actions
    
    def store_transitions(self,
                       states: Dict[str, np.ndarray],
                       actions: Dict[str, int],
                       rewards: Dict[str, float],
                       next_states: Dict[str, np.ndarray],
                       dones: Dict[str, bool]):
        """Store transitions for all agents"""
        for agent_id in self.agents:
            if agent_id in states:
                self.agents[agent_id].store_transition(
                    states[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_states[agent_id],
                    dones[agent_id] if isinstance(dones, dict) else dones
                )
    
    def update_agents(self) -> Dict[str, float]:
        """Update all agents"""
        losses = {}
        for agent_id, agent in self.agents.items():
            loss = agent.update()
            if loss is not None:
                losses[agent_id] = loss
        return losses
    
    def train_episode(self,
                     env,
                     max_steps: int = 1000) -> Dict[str, float]:
        """Train agents for one episode"""
        episode_rewards = defaultdict(float)
        states, _ = env.reset()
        
        for step in range(max_steps):
            # Select actions
            actions = self.select_actions(states)
            
            # Take actions in environment
            next_states, rewards, done, _, info = env.step(actions)
            
            # Convert single done boolean to dict if necessary
            if isinstance(done, bool):
                dones = {agent_id: done for agent_id in self.agents}
            else:
                dones = done
            
            # Store transitions
            self.store_transitions(states, actions, rewards, next_states, dones)
            
            # Update agents
            losses = self.update_agents()
            
            # Track rewards
            for agent_id, reward in rewards.items():
                episode_rewards[agent_id] += reward
            
            # Update states
            states = next_states
            self.step_counter += 1
            
            # Break if done
            if isinstance(done, bool):
                if done:
                    break
            elif all(dones.values()):
                break
        
        # Calculate episode metrics
        metrics = {
            'episode_length': step + 1,
            'mean_reward': np.mean(list(episode_rewards.values())),
            'min_reward': min(episode_rewards.values()),
            'max_reward': max(episode_rewards.values())
        }
        
        # Update tracking
        for key, value in metrics.items():
            self.episode_metrics[key].append(value)
        self.global_metrics['episode_rewards'].append(metrics['mean_reward'])
        
        return metrics
    
    def test_episode(self,
                    env,
                    max_steps: int = 1000,
                    render: bool = False) -> Dict[str, float]:
        """Test agents for one episode"""
        # Set agents to evaluation mode
        for agent in self.agents.values():
            agent.training = False
        
        episode_rewards = defaultdict(float)
        states, _ = env.reset()
        
        for step in range(max_steps):
            if render:
                env.render()
            
            # Select actions
            actions = self.select_actions(states)
            
            # Take actions in environment
            next_states, rewards, done, _, info = env.step(actions)
            
            # Track rewards
            for agent_id, reward in rewards.items():
                episode_rewards[agent_id] += reward
            
            # Update states
            states = next_states
            
            # Break if done
            if isinstance(done, bool):
                if done:
                    break
            elif all(done.values()):
                break
        
        # Calculate test metrics
        metrics = {
            'test_episode_length': step + 1,
            'test_mean_reward': np.mean(list(episode_rewards.values())),
            'test_min_reward': min(episode_rewards.values()),
            'test_max_reward': max(episode_rewards.values())
        }
        
        # Set agents back to training mode
        for agent in self.agents.values():
            agent.training = True
        
        return metrics
    
    def save_agents(self, save_dir: str):
        """Save all agents"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            agent_path = save_path / f"{agent_id}.pt"
            agent.save(str(agent_path))
        
        # Save metrics
        metrics_path = save_path / "metrics.json"
        metrics_data = {
            'episode_metrics': dict(self.episode_metrics),
            'global_metrics': dict(self.global_metrics)
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=4)
    
    def load_agents(self, load_dir: str):
        """Load all agents"""
        load_path = Path(load_dir)
        
        for agent_id, agent in self.agents.items():
            agent_path = load_path / f"{agent_id}.pt"
            if agent_path.exists():
                agent.load(str(agent_path))
        
        # Load metrics if available
        metrics_path = load_path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
                self.episode_metrics = defaultdict(list, metrics_data['episode_metrics'])
                self.global_metrics = defaultdict(list, metrics_data['global_metrics'])