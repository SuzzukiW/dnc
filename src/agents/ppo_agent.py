# src/agents/ppo_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque

from src.models.ppo_network import ActorCriticNetwork
import torch.nn.functional as F

class PPOAgent:
    """
    PPO Agent implementation for traffic light control
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Dict,
        agent_id: str
    ):
        """
        Initialize PPO Agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
            agent_id: Unique identifier for the agent
        """
        self.config = config
        self.agent_id = agent_id
        self.device = torch.device("cpu")  # Stick to CPU for consistent performance
        
        # Initialize network with simplified architecture
        self.network = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_hidden_sizes=config['network']['actor']['hidden_sizes'],
            critic_hidden_sizes=config['network']['critic']['hidden_sizes'],
            activation=config['network']['actor']['activation']
        )
        
        # Fast optimizer setup
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config['ppo']['learning_rate']
        )
        
        # Pre-allocate memory buffers as lists for speed
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []
        
        # Initialize episode metrics
        self.episode_rewards = []
        self.training_step = 0
        
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Optimized action selection"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), value.item(), log_prob.item()
        
    def store_transition(self, state, action, reward, value, log_prob, mask):
        """Fast transition storage"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.masks.append(mask)
        
    def clear_memory(self):
        """Clear memory buffers"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.masks.clear()
        
    def compute_gae(
        self, 
        rewards: List[float], 
        values: List[float], 
        dones: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: List of rewards
            values: List of state values
            dones: List of done flags
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        
        gae = 0
        for step in reversed(range(len(rewards))):
            # Compute delta
            delta = (rewards[step] + 
                     (1 - dones[step]) * self.config['ppo']['gamma'] * values[step + 1] - 
                     values[step])
            
            # Compute GAE
            gae = delta + (1 - dones[step]) * \
                  self.config['ppo']['gamma'] * \
                  self.config['ppo']['gae_lambda'] * gae
            
            advantages.insert(0, gae)
            
            # Compute returns
            returns.insert(0, gae + values[step])
        
        return advantages, returns

    def update(self):
        """
        Perform PPO update on the policy and value networks
        """
        # Prepare data tensors
        states = torch.FloatTensor(np.array(self.states))
        old_actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        rewards = torch.FloatTensor(self.rewards)
        masks = torch.FloatTensor(self.masks)
        
        # Compute advantages and returns
        values = [v for v in self.values] + [0]  # Add terminal value
        advantages, returns = self.compute_gae(
            rewards=self.rewards, 
            values=values, 
            dones=[1 - m for m in self.masks]
        )
        
        # Convert to tensors
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Mini-batch training
        batch_size = self.config['ppo']['batch_size']
        n_epochs = self.config['ppo']['n_epochs']
        
        for _ in range(n_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = old_actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                action_probs, values = self.network(batch_states)
                
                # Compute log probabilities
                dist = torch.distributions.Categorical(action_probs)
                log_probs = dist.log_prob(batch_actions)
                
                # Compute policy ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Compute PPO loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 
                    1 - self.config['ppo']['clip_range'], 
                    1 + self.config['ppo']['clip_range']
                ) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy loss
                entropy_loss = -torch.mean(dist.entropy())
                
                # Total loss
                loss = (
                    policy_loss + 
                    self.config['ppo']['vf_coef'] * value_loss + 
                    self.config['ppo']['ent_coef'] * entropy_loss
                )
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), 
                    self.config['ppo']['max_grad_norm']
                )
                
                self.optimizer.step()
        
        # Clear memory
        self.clear_memory()
        
    def train(self):
        """
        Train the agent using collected transitions
        """
        if len(self.states) > 0:  # Only train if we have collected some transitions
            self.update()
            self.clear_memory()
        
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step
        }, path)
        
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']