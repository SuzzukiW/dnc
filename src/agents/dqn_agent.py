# src/agents/dqn_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np
from collections import deque, namedtuple
import random

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, action_type='discrete'):
        super(DQNNetwork, self).__init__()
        self.action_type = action_type
        self.state_size = None  # Will be set dynamically
        self.hidden_size = hidden_size
        self.action_size = action_size
        
        # Initialize with placeholders to ensure parameters exist
        self.shared_layers = nn.Sequential(
            nn.Linear(1, hidden_size),  # Placeholder input size
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        if action_type == 'discrete':
            self.output_layer = nn.Linear(hidden_size, action_size)
        else:
            self.output_layer = nn.Linear(hidden_size, action_size * 2)
    
    def _adapt_layers(self, input_size):
        """Adapt layers to new input size if needed"""
        if self.state_size == input_size:
            return
            
        self.state_size = input_size
        device = next(self.parameters()).device
        
        # Create new layers with correct input size
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        ).to(device)
        
        if self.action_type == 'discrete':
            self.output_layer = nn.Linear(self.hidden_size, self.action_size).to(device)
        else:
            self.output_layer = nn.Linear(self.hidden_size, self.action_size * 2).to(device)
    
    def forward(self, state):
        # Convert numpy arrays to tensors if necessary
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
            
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Adapt layers if input size has changed
        if self.state_size != state.shape[1]:
            self._adapt_layers(state.shape[1])
        
        # Forward pass
        features = self.shared_layers(state)
        
        if self.action_type == 'discrete':
            return self.output_layer(features)
        else:
            output = self.output_layer(features)
            mean, log_std = torch.chunk(output, 2, dim=-1)
            log_std = torch.clamp(log_std, min=-5, max=2)
            return mean

class DQNAgent:
    def __init__(self, state_size, action_size, config):
            self.action_size = action_size
            
            # Use MPS (Metal Performance Shaders) for M1/M2/M3 Macs
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                
            print(f"Using device: {self.device}")
            
            # Training parameters
            self.gamma = config.get('gamma', 0.95)
            self.learning_rate = config.get('learning_rate', 0.001)
            self.epsilon = config.get('epsilon_start', 1.0)
            self.epsilon_min = config.get('epsilon_min', 0.01)
            self.epsilon_decay = config.get('epsilon_decay', 0.995)
            self.batch_size = config.get('batch_size', 32)
            self.memory_size = config.get('memory_size', 10000)
            
            # Initialize networks
            self.policy_net = DQNNetwork(
                state_size=1,
                action_size=action_size,
                hidden_size=config.get('hidden_sizes', [64])[0],
                action_type=config.get('action_type', 'discrete')
            ).to(self.device)
            
            self.target_net = DQNNetwork(
                state_size=1,
                action_size=action_size,
                hidden_size=config.get('hidden_sizes', [64])[0],
                action_type=config.get('action_type', 'discrete')
            ).to(self.device)
            
            # Initialize target network with policy network's weights
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Initialize optimizer with larger batch size for M3
            self.optimizer = optim.Adam(
                self.policy_net.parameters(),
                lr=self.learning_rate,
                eps=1e-4  # Increased epsilon for better numerical stability
            )
            
            # Use a more efficient memory structure
            self.memory = deque(maxlen=self.memory_size)
            self.Transition = namedtuple('Transition', 
                                    ('state', 'action', 'reward', 'next_state', 'done'))
                                    
            # Batch tensors for reuse
            self.batch_indices = torch.arange(self.batch_size, device=self.device)
        
    def remember(self, state, action, reward, next_state, done):
        """Store transition in replay memory"""
        if len(self.memory) >= self.memory_size:
            self.memory.popleft()  # Use popleft() instead of pop(0)
        self.memory.append(self.Transition(state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
                
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def replay(self):
            """Train on batch from replay memory with optimized processing"""
            if len(self.memory) < self.batch_size:
                return None
                
            # Sample and process batch more efficiently
            transitions = random.sample(self.memory, self.batch_size)
            batch = self.Transition(*zip(*transitions))
            
            # Pre-allocate tensors on device
            state_batch = torch.stack([torch.FloatTensor(s) for s in batch.state]).to(self.device)
            action_batch = torch.tensor(batch.action, device=self.device, dtype=torch.long)
            reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float)
            next_state_batch = torch.stack([torch.FloatTensor(s) for s in batch.next_state]).to(self.device)
            done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float)
            
            # Compute current Q values in single pass
            current_q_values = self.policy_net(state_batch)
            state_action_values = current_q_values.gather(1, action_batch.unsqueeze(1))
            
            # Compute next Q values in single pass
            with torch.no_grad():
                next_state_values = self.target_net(next_state_batch).max(1)[0]
                expected_state_action_values = (reward_batch + self.gamma * next_state_values * (1 - done_batch))
            
            # Compute loss and update in single step
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
            self.optimizer.step()
            
            # Update epsilon more efficiently
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
            
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