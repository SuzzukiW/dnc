# src/models/dqn_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    """
    Neural network for DQN agent
    Architecture: Fully connected layers with ReLU activation
    """
    def __init__(self, state_size, action_size, hidden_size=64):
        """
        Initialize DQN network
        
        Args:
            state_size: Dimension of input state
            action_size: Dimension of output actions
            hidden_size: Size of hidden layers
        """
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        """
        Forward pass through network
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for each action
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)