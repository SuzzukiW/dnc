# src/agents/dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
from collections import deque, namedtuple
import random

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, action_type='discrete'):
        super(DQNNetwork, self).__init__()
        self.action_type = action_type
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Output layers based on action type
        if action_type == 'discrete':
            # Discrete action space
            self.output_layer = nn.Linear(hidden_size, action_size)
        else:
            # Continuous action space
            # Output mean and log standard deviation for a Gaussian distribution
            self.output_layer = nn.Linear(hidden_size, action_size * 2)
    
    def forward(self, state):
        # Shared feature extraction
        features = self.shared_layers(state)
        
        # Output based on action type
        if self.action_type == 'discrete':
            # Discrete action: direct Q-value output
            return self.output_layer(features)
        else:
            # Continuous action: output mean and log std
            output = self.output_layer(features)
            mean, log_std = torch.chunk(output, 2, dim=-1)
            
            # Clamp log standard deviation to prevent numerical instability
            log_std = torch.clamp(log_std, min=-5, max=2)
            
            # Return mean of continuous action distribution
            return mean

class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.95)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 32)
        self.memory_size = config.get('memory_size', 10000)
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(state_size, action_size, action_type=config.get('action_type', 'discrete')).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size, action_type=config.get('action_type', 'discrete')).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Replay memory
        self.Transition = namedtuple('Transition', 
                                   ('state', 'action', 'reward', 'next_state', 'done'))
        self.memory = deque(maxlen=self.memory_size)
        
    def remember(self, state, action, reward, next_state, done):
        """Store transition in replay memory"""
        self.memory.append(self.Transition(state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            if self.policy_net.action_type == 'discrete':
                return random.randrange(self.action_size)
            else:
                return np.random.uniform(-1, 1, self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            if self.policy_net.action_type == 'discrete':
                return q_values.argmax().item()
            else:
                return q_values.squeeze(0).cpu().numpy()
    
    def replay(self):
        """Train on batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return
        
        transitions = random.sample(self.memory, self.batch_size)
        batch = self.Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device) if self.policy_net.action_type == 'discrete' else torch.FloatTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        if self.policy_net.action_type == 'discrete':
            state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        else:
            state_action_values = self.policy_net(state_batch)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach() if self.policy_net.action_type == 'discrete' else self.target_net(next_state_batch)
        
        # Compute the expected Q values
        expected_state_action_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)
        
        # Compute loss
        if self.policy_net.action_type == 'discrete':
            loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        else:
            loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        
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