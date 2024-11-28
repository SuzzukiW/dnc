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
        
        # Ensure state_size is an integer and matches input dimension
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        
        # Default hidden layer configuration if not provided
        if hidden_sizes is None:
            # Scale hidden layers based on input size
            hidden_sizes = [
                min(384, max(64, self.state_size * 2)),
                min(192, max(32, self.state_size)),
                min(96, max(16, self.state_size // 2))
            ]
        
        # Input layer matching the state size exactly
        self.input_layer = nn.Linear(self.state_size, hidden_sizes[0])
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
        
        # Output layer for Q-values
        self.output_layer = nn.Linear(prev_size, action_size)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
        # Activation functions
        self.activation = nn.ReLU()
        
    def forward(self, state):
        """
        Forward pass with input validation and adaptive processing
        
        Args:
            state: Input state tensor
        
        Returns:
            Q-values for actions
        """
        # Input validation and conversion
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Ensure state matches expected input size
        if state.shape[-1] != self.state_size:
            raise ValueError(f"Input state size {state.shape[-1]} does not match expected size {self.state_size}")
        
        # Forward pass through network
        x = self.activation(self.input_layer(state))
        
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        
        # Output Q-values
        q_values = self.output_layer(x)
        
        return q_values
    
    def save(self, path):
        """Save model state"""
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """Load model state"""
        self.load_state_dict(torch.load(path))