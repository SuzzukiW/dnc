# src/agents/dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_sizes: list = [64, 32]):
        super(QNetwork, self).__init__()
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, action_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)

class DQNAgent:
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Neural Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience Replay
        self.memory = deque(maxlen=memory_size)
        
        logger.info(f"Initialized DQN Agent - State size: {state_size}, Action size: {action_size}")
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store transition in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def get_action(self, state: np.ndarray) -> int:
        """Get action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()
    
    def train(self) -> float:
        """Train the agent using experience replay."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample mini-batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network periodically
        if random.random() < 0.01:  # 1% chance each update
            self.target_model.load_state_dict(self.model.state_dict())
        
        return loss.item()
    
    def save_model(self, path: str):
        """Save the model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.target_model.load_state_dict(self.model.state_dict())
        logger.info(f"Model loaded from {path}")