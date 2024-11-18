# src/models/dqn_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=None):
        """
        Adaptive DQN Network with flexible architecture
        
        Args:
            state_size: Input state dimension
            action_size: Output action dimension
            hidden_sizes: Optional list of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()
        
        # Ensure state_size is an integer
        state_size = int(state_size)
        
        # Default hidden layer configuration if not provided
        if hidden_sizes is None:
            # Adaptive hidden layer sizes based on input state size
            hidden_sizes = [
                max(32, min(256, int(state_size * 1.5))),  # First hidden layer
                max(16, min(128, int(state_size)))         # Second hidden layer
            ]
        
        # Input layer with adaptive size
        self.input_layer = nn.Linear(state_size, hidden_sizes[0])
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        prev_size = hidden_sizes[0]
        for hidden_size in hidden_sizes[1:]:
            layer = nn.Linear(prev_size, hidden_size)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.hidden_layers.append(layer)
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, action_size)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, state):
        """
        Forward pass with adaptive processing
        
        Args:
            state: Input state tensor
        
        Returns:
            Q-values for actions
        """
        # Ensure input is a 2D tensor with correct dtype
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Ensure state is float and on the correct device
        state = state.float()
        
        # Input layer with ReLU activation
        x = F.relu(self.input_layer(state))
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        # Output layer
        return self.output_layer(x)
    
    def reset_noise(self):
        """
        Reset noise for exploration (optional)
        Useful for techniques like Noisy Networks
        """
        pass