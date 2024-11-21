# experiments/scenarios/communication/hierarchical.py

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import random
import logging
import networkx as nx
import sumolib
import json

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise ValueError("Please declare environment variable 'SUMO_HOME'")

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.utils import setup_logger
from src.utils.replay_buffer import PrioritizedReplayBuffer

class Message:
    """Represents a communication message between agents"""
    
    TYPES = {
        'STATE_UPDATE': 1,     # Share state information
        'ACTION_PLAN': 2,      # Share planned actions
        'COORDINATION': 3,     # Coordination requests/responses
        'RECOMMENDATION': 4,   # Regional recommendations
        'ALERT': 5            # Critical situation alerts
    }
    
    def __init__(self,
                msg_type: int,
                sender: str,
                content: dict,
                priority: float = 1.0,
                timestamp: Optional[float] = None):
        """
        Initialize message
        
        Args:
            msg_type: Type of message (from TYPES)
            sender: ID of sending agent
            content: Message content dictionary
            priority: Message priority (0-1)
            timestamp: Message creation time
        """
        self.type = msg_type
        self.sender = sender
        self.content = content
        self.priority = priority
        self.timestamp = timestamp or traci.simulation.getTime()
        
        # Track message handling
        self.processed = False
        self.response_to = None
        self.responses = []

class RegionalCoordinator:
    """Manages coordination between traffic lights in a region"""
    
    def __init__(self,
                region_id: str,
                config: dict):
        """
        Initialize regional coordinator
        
        Args:
            region_id: Unique identifier for the region
            config: Configuration dictionary
        """
        self.region_id = region_id
        self.member_agents = set()
        self.messages = []
        self.config = config
        self.performance_history = deque(maxlen=100)
    
    def add_agent(self, agent_id: str):
        self.member_agents.add(agent_id)
    
    def remove_agent(self, agent_id: str):
        self.member_agents.remove(agent_id)
    
    def clear_messages(self):
        self.messages = []
    
    def add_message(self, message: Message):
        self.messages.append(message)
    
    def get_performance_score(self) -> float:
        return np.mean(self.performance_history) if self.performance_history else 0
    
    def update_performance(self, score: float):
        self.performance_history.append(score)

class PolicyNetwork(nn.Module):
    """Enhanced policy network with attention and traffic-specific features"""

    def __init__(self, state_size, action_size, message_size=32, hidden_size=128):
        """
        Initialize enhanced policy network
        
        Args:
            state_size: Size of state input
            action_size: Number of possible actions
            message_size: Size of message vectors
            hidden_size: Size of hidden layers
        """
        super().__init__()
        
        # Traffic state processing
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Message processing with attention
        self.message_size = message_size
        self.message_attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=0.1)
        self.message_encoder = nn.Sequential(
            nn.Linear(message_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Traffic pattern recognition
        self.pattern_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU()
        )
        
        # Combined processing with residual connections
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_size * 2 + hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # Dual heads for policy and value with separate networks
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights with orthogonal initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def forward(self, local_state, messages=None):
        """
        Forward pass with attention and traffic pattern recognition
        
        Args:
            local_state: Local state input tensor [batch_size, state_size]
            messages: List of message tensors [batch_size, message_size]
            
        Returns:
            action_logits: Action probability logits
            value: State value estimate
            attention_weights: Optional attention weights for visualization
        """
        batch_size = local_state.shape[0]
        
        # Process local state
        state_features = self.state_encoder(local_state)
        
        # Process messages with attention if available
        if messages and len(messages) > 0:
            message_tensors = torch.stack(messages, dim=0)  # [num_messages, batch_size, message_size]
            message_features = self.message_encoder(message_tensors)
            
            # Multi-head attention
            message_features = message_features.permute(1, 0, 2)  # [batch_size, num_messages, hidden_size]
            attended_messages, attention_weights = self.message_attention(
                message_features, message_features, message_features
            )
            message_features = attended_messages.mean(dim=1)  # Pool attention outputs
        else:
            message_features = torch.zeros(batch_size, self.hidden_size, device=local_state.device)
            attention_weights = None
        
        # Extract traffic patterns
        pattern_features = self.pattern_network(state_features)
        
        # Combine features with residual connections
        combined_features = torch.cat([
            state_features,
            message_features,
            pattern_features
        ], dim=1)
        
        fused_features = self.fusion_network(combined_features)
        
        # Generate policy and value outputs
        action_logits = self.policy_head(fused_features)
        value = self.value_head(fused_features)
        
        return action_logits, value, attention_weights

class HierarchicalAgent:
    """Enhanced agent with prioritized experience replay and advanced learning"""
    
    def __init__(self,
                agent_id: str,
                state_size: int,
                action_size: int,
                config: dict):
        """Initialize hierarchical agent with advanced features"""
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        
        # Get config parameters with defaults
        train_config = config.get('training', {})
        self.gamma = train_config.get('gamma', 0.99)
        self.epsilon_start = train_config.get('epsilon_start', 1.0)
        self.epsilon_end = train_config.get('epsilon_end', 0.01)
        self.epsilon_decay = train_config.get('epsilon_decay', 0.995)
        self.batch_size = train_config.get('batch_size', 64)
        self.learning_rate = train_config.get('learning_rate', 0.0003)
        self.tau = train_config.get('tau', 0.005)  # Soft update parameter
        
        # Initialize networks with larger hidden size
        self.policy_net = PolicyNetwork(state_size, action_size, hidden_size=256)
        self.target_net = PolicyNetwork(state_size, action_size, hidden_size=256)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer with gradient clipping
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.grad_clip = train_config.get('grad_clip', 0.5)
        
        # Adaptive epsilon decay
        self.epsilon = self.epsilon_start
        self.epsilon_decay_steps = train_config.get('epsilon_decay_steps', 100000)
        
        # Prioritized experience replay
        self.memory = PrioritizedReplayBuffer(
            size=train_config.get('memory_size', 100000),
            alpha=0.6,  # Prioritization exponent
            beta=0.4    # Initial importance sampling weight
        )
        
        # Message handling
        self.message_buffer = []
        self.max_messages = train_config.get('max_messages', 10)
        self.message_importance = train_config.get('message_importance', 0.5)
        
        # Performance tracking
        self.episode_rewards = []
        self.waiting_times = []
        self.update_count = 0
        
        # Region info
        self.region_id = None
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        # Initialize metrics
        self.metrics = defaultdict(float)
    
    def select_action(self, state: np.ndarray, messages: Optional[List[torch.Tensor]] = None, training: bool = True) -> int:
        """Select action using epsilon-greedy with adaptive noise"""
        if training:
            # Adaptive epsilon decay
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                         np.exp(-self.update_count / self.epsilon_decay_steps)
            
            if random.random() < self.epsilon:
                return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if messages:
                messages = [m.to(self.device) for m in messages[-self.max_messages:]]
            
            action_logits, _, _ = self.policy_net(state, messages)
            
            # Add small noise for exploration even during exploitation
            if training:
                action_logits += torch.randn_like(action_logits) * 0.1
            
            return action_logits.argmax(dim=1).item()
    
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, 
              done: bool, messages: Optional[List[torch.Tensor]] = None) -> Tuple[float, float]:
        """Update agent with prioritized experience replay and advanced learning"""
        # Store experience with maximum priority for new experiences
        self.memory.add(
            state, action, reward, next_state, done,
            messages if messages else [],
            priority=self.memory.max_priority
        )
        
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0
        
        self.update_count += 1
        
        # Sample batch with importance sampling
        beta = min(1.0, 0.4 + self.update_count * 0.6 / 100000)  # Anneal beta to 1
        experiences = self.memory.sample(self.batch_size, beta)
        
        states, actions, rewards, next_states, dones, batch_messages, weights, indices = experiences
        
        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Get current Q values
        q_values, state_values, _ = self.policy_net(states, batch_messages)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values with double Q-learning
        with torch.no_grad():
            next_q_values, next_state_values, _ = self.policy_net(next_states, batch_messages)
            next_actions = next_q_values.argmax(dim=1, keepdim=True)
            
            target_q_values, _, _ = self.target_net(next_states, batch_messages)
            next_q = target_q_values.gather(1, next_actions).squeeze(1)
            
            # Compute target with TD(λ) and GAE
            target_q = rewards + (1 - dones) * self.gamma * (
                0.8 * next_q + 0.2 * next_state_values.squeeze(1)
            )
        
        # Compute losses with prioritized replay correction
        q_loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        value_loss = (weights * F.mse_loss(state_values.squeeze(1), target_q.detach(), reduction='none')).mean()
        
        # Combined loss with L2 regularization
        loss = q_loss + 0.5 * value_loss + 0.01 * sum(p.pow(2.0).sum() for p in self.policy_net.parameters())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()
        
        # Update priorities
        td_errors = torch.abs(target_q - current_q).detach().cpu().numpy()
        priorities = (td_errors + 1e-6) ** 0.6  # Convert to priorities using alpha=0.6
        self.memory.update_priorities(indices, priorities)
        
        # Store metrics
        self.metrics = {
            'td_error': float(td_errors.mean()),
            'priority': float(priorities.mean()),
            'beta': beta,
            'q_loss': q_loss.item(),
            'value_loss': value_loss.item()
        }
        
        # Soft update target network
        if self.update_count % 10 == 0:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
                )
        
        return q_loss.item(), value_loss.item()
    
    def process_messages(self, messages: List[Message]) -> List[torch.Tensor]:
        """Process and prioritize messages"""
        if not messages:
            return []
        
        # Sort messages by priority and recency
        messages.sort(key=lambda m: (m.priority, -m.timestamp))
        messages = messages[-self.max_messages:]  # Keep only most recent/important messages
        
        # Convert messages to tensors
        message_tensors = []
        for msg in messages:
            if msg.type == Message.TYPES['STATE_UPDATE']:
                # Encode state information
                state_tensor = torch.FloatTensor(msg.content['state']).to(self.device)
                encoded_state = self.policy_net.state_encoder(state_tensor.unsqueeze(0))
                message_tensors.append(encoded_state.squeeze(0))
            elif msg.type == Message.TYPES['ACTION_PLAN']:
                # Encode action information
                action_tensor = torch.zeros(self.action_size, device=self.device)
                action_tensor[msg.content['action']] = 1.0
                message_tensors.append(action_tensor)
        
        return message_tensors
    
    def clear_messages(self):
        """Clear message buffer"""
        self.message_buffer = []
    
    def add_message(self, message: Message):
        """Add message to buffer with priority handling"""
        self.message_buffer.append(message)
        if len(self.message_buffer) > self.max_messages:
            # Remove lowest priority messages
            self.message_buffer.sort(key=lambda m: (m.priority, -m.timestamp))
            self.message_buffer = self.message_buffer[-self.max_messages:]
    
    def save(self, path: str):
        """
        Save agent's models and training state
        
        Args:
            path: Path to save the agent state
        """
        save_dict = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count,
            'metrics': dict(self.metrics)
        }
        torch.save(save_dict, path)
    
    def load(self, path: str):
        """
        Load agent's models and training state
        
        Args:
            path: Path to load the agent state from
        """
        if not os.path.exists(path):
            print(f"Warning: No saved model found at {path}")
            return
            
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint['update_count']
        self.metrics.update(checkpoint['metrics'])

class HierarchicalManager:
    """Manages hierarchical multi-agent system and training"""
    
    def __init__(self,
                state_size: int,
                action_size: int,
                net_file: str,
                config: dict):
        """Initialize hierarchical system manager"""
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.net_file = net_file
        
        # Get hierarchical config with defaults
        self.hierarchical_config = config.get('hierarchical', {})
        self.influence_radius = self.hierarchical_config.get('influence_radius', 150.0)
        self.region_size = self.hierarchical_config.get('region_size', 3)
        self.coordination_interval = self.hierarchical_config.get('coordination_interval', 5)
        self.message_size = self.hierarchical_config.get('message_size', 32)
        self.coordination_threshold = self.hierarchical_config.get('coordination_threshold', 0.6)
        
        # Agents and coordinators
        self.agents = {}
        self.coordinators = {}
        
        # Initialize metrics with default values
        self.metrics = {
            # Training metrics
            'avg_td_error': 0.0,
            'avg_priority': 0.0,
            'current_beta': 0.4,
            'current_epsilon': 1.0,
            'learning_rate': self.config.get('training', {}).get('learning_rate', 0.001),
            
            # Agent metrics
            'num_regions': 0,
            'avg_region_size': 0,
            'coordination_rate': 0.0,
            
            # Traffic metrics
            'avg_queue_length': 0.0,
            'avg_speed': 0.0,
            'throughput': 0.0,
            'avg_waiting_time': 0.0,
            
            # Buffer metrics
            'buffer_size': 0
        }
        
        # Performance tracking
        self.episode_stats = {
            'coordination_count': 0,
            'total_messages': 0,
            'queue_lengths': [],
            'speeds': [],
            'waiting_times': [],
            'vehicle_counts': []
        }

    def add_agent(self, agent_id: str):
        self.agents[agent_id] = HierarchicalAgent(
            agent_id,
            self.state_size,
            self.action_size,
            self.config
        )
    
    def update_regions(self, positions: Dict[str, Tuple[float, float]]):
        # Skip if no positions provided
        if not positions:
            return
            
        # Only consider agents that are in the positions dictionary
        valid_agents = set(positions.keys())
        
        # Clear existing regions
        old_regions = {agent.region_id for agent in self.agents.values() if agent.region_id}
        self.coordinators = {}
        
        # Group agents by proximity
        unassigned = list(valid_agents)
        region_id = 0
        
        while unassigned:
            # Start new region
            current = unassigned.pop(0)
            current_pos = positions[current]
            
            # Create coordinator
            coordinator = RegionalCoordinator(region_id, self.config)
            coordinator.add_agent(current)
            self.agents[current].region_id = region_id
            
            # Find nearby agents
            i = 0
            while i < len(unassigned):
                other = unassigned[i]
                other_pos = positions[other]
                
                # Check if within influence radius
                if self._calculate_distance(current_pos, other_pos) <= self.influence_radius:
                    coordinator.add_agent(other)
                    self.agents[other].region_id = region_id
                    unassigned.pop(i)
                else:
                    i += 1
            
            self.coordinators[region_id] = coordinator
            region_id += 1
        
        # Update metrics
        self.metrics['num_regions'] = len(self.coordinators)
        self.metrics['avg_region_size'] = np.mean([len(c.member_agents) for c in self.coordinators.values()])
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        return np.sqrt(np.sum((np.array(pos1) - np.array(pos2)) ** 2))
    
    def step(self, states, training=True):
        """
        Take a step with all agents
        
        Args:
            states: Dictionary of states for each agent {agent_id: state}
                   or tuple of (states_dict, global_state)
            training: Whether this is a training step
            
        Returns:
            Dictionary of actions for each agent
        """
        # Handle case where states is (states_dict, global_state) tuple
        if isinstance(states, tuple):
            states_dict, global_state = states
        else:
            states_dict = states
            global_state = None
            
        actions = {}
        
        # First get coordinator actions for each region
        for coordinator in self.coordinators.values():
            region_id = coordinator.region_id
            if f"coordinator_{region_id}" in states_dict:
                region_state = states_dict[f"coordinator_{region_id}"]
                # For now, use a simple strategy selection (can be enhanced later)
                actions[f"coordinator_{region_id}"] = 0  # Default strategy
        
        # Then get actions for each agent
        for agent_id, state in states_dict.items():
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                
                # Ensure state is numpy array
                if not isinstance(state, np.ndarray):
                    state = np.array(state, dtype=np.float32)
                
                # Include global state if available
                if global_state is not None:
                    if not isinstance(global_state, np.ndarray):
                        global_state = np.array(global_state, dtype=np.float32)
                    # Ensure dimensions match before concatenating
                    if state.ndim == 1 and global_state.ndim == 1:
                        combined_state = np.concatenate([state, global_state])
                    else:
                        # Handle case where either state has extra dimensions
                        flat_state = state.flatten()
                        flat_global = global_state.flatten()
                        combined_state = np.concatenate([flat_state, flat_global])
                else:
                    combined_state = state
                    
                action = agent.select_action(combined_state, training=training)
                actions[agent_id] = action
                
        return actions
    
    def process_communications(self):
        # Clear old messages
        for agent in self.agents.values():
            agent.clear_messages()
        for coordinator in self.coordinators.values():
            coordinator.clear_messages()
    
    def update(self, states: Dict[str, np.ndarray], actions: Dict[str, int], rewards: Dict[str, float], next_states: Dict[str, np.ndarray], dones: Dict[str, bool], info: Dict[str, dict] = None) -> Dict[str, float]:
        """
        Update all agents and collect metrics
        
        Args:
            states: Current states for each agent
            actions: Actions taken by each agent
            rewards: Rewards received by each agent
            next_states: Next states for each agent
            dones: Done flags for each agent
            info: Additional information from environment
            
        Returns:
            Dictionary of agent losses
        """
        losses = {}
        total_td_error = 0.0
        total_priority = 0.0
        total_beta = 0.0
        total_epsilon = 0.0
        num_agents = len(self.agents)
        
        # Reset episode statistics
        self.episode_stats['queue_lengths'] = []
        self.episode_stats['speeds'] = []
        self.episode_stats['waiting_times'] = []
        self.episode_stats['vehicle_counts'] = []
        
        for agent_id in states:
            if agent_id not in self.agents:
                continue
            
            agent = self.agents[agent_id]
            messages = []
            
            # Get messages from region
            if agent.region_id is not None:
                coordinator = self.coordinators[agent.region_id]
                messages = coordinator.messages
            
            # Update agent
            loss = agent.update(
                states[agent_id],
                actions[agent_id],
                rewards[agent_id],
                next_states[agent_id],
                dones[agent_id],
                messages
            )
            
            losses[agent_id] = loss
            
            # Collect metrics from agent
            if hasattr(agent, 'metrics'):
                total_td_error += agent.metrics.get('td_error', 0.0)
                total_priority += agent.metrics.get('priority', 0.0)
                total_beta += agent.metrics.get('beta', 0.4)
                total_epsilon += agent.metrics.get('epsilon', 1.0)
            
            # Collect traffic metrics from info
            if info and agent_id in info:
                agent_info = info[agent_id]
                self.episode_stats['queue_lengths'].append(agent_info.get('queue_length', 0))
                self.episode_stats['speeds'].append(agent_info.get('speed', 0))
                self.episode_stats['waiting_times'].append(agent_info.get('waiting_time', 0))
                self.episode_stats['vehicle_counts'].append(agent_info.get('vehicle_count', 0))
            
            # Update region performance
            if agent.region_id is not None:
                coordinator = self.coordinators[agent.region_id]
                coordinator.update_performance(rewards[agent_id])
        
        # Update all metrics
        if num_agents > 0:
            # Update training metrics
            self.metrics.update({
                'avg_td_error': total_td_error / num_agents,
                'avg_priority': total_priority / num_agents,
                'current_beta': total_beta / num_agents,
                'current_epsilon': total_epsilon / num_agents,
                'learning_rate': self.config.get('training', {}).get('learning_rate', 0.001),
                
                # Update agent metrics
                'num_regions': len(self.coordinators),
                'avg_region_size': np.mean([len(c.member_agents) for c in self.coordinators.values()]),
                'coordination_rate': self.episode_stats['coordination_count'] / max(1, self.episode_stats['total_messages']),
                
                # Update traffic metrics
                'avg_queue_length': np.mean(self.episode_stats['queue_lengths']) if self.episode_stats['queue_lengths'] else 0,
                'avg_speed': np.mean(self.episode_stats['speeds']) if self.episode_stats['speeds'] else 0,
                'avg_waiting_time': np.mean(self.episode_stats['waiting_times']) if self.episode_stats['waiting_times'] else 0,
                'throughput': sum(self.episode_stats['vehicle_counts']) if self.episode_stats['vehicle_counts'] else 0,
                
                # Update buffer metrics
                'buffer_size': sum(len(agent.memory) for agent in self.agents.values()) if hasattr(next(iter(self.agents.values())), 'memory') else 0
            })
        
        return losses

    def get_metrics(self) -> Dict[str, float]:
        """Return metrics from manager"""
        return self.metrics.copy()  # Return a copy to prevent external modifications

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save agents
        for agent_id, agent in self.agents.items():
            agent.save(str(path / f"agent_{agent_id}.pt"))
        
        # Save metrics
        with open(path / "metrics.json", 'w') as f:
            json.dump(self.metrics, f)
    
    def load(self, path: str):
        path = Path(path)
        
        # Load agents
        for agent_id, agent in self.agents.items():
            agent_path = path / f"agent_{agent_id}.pt"
            if agent_path.exists():
                agent.load(str(agent_path))
        
        # Load metrics
        metrics_path = path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                self.metrics = defaultdict(list, json.load(f))

    def update_learning_rate(self, lr: float):
        """
        Update learning rate for all agents
        
        Args:
            lr: New learning rate value
        """
        for agent in self.agents.values():
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = lr
    
    def process_communications(self):
        """
        Process communications between agents and return coordination metrics
        
        Returns:
            dict: Coordination metrics including:
                - coordinated_agents: Number of agents that coordinated
                - total_interactions: Total number of possible interactions
        """
        coordinated_agents = 0
        total_interactions = 0
        
        # Process communications for each region
        for coordinator in self.coordinators.values():
            agents_in_region = len(coordinator.member_agents)
            if agents_in_region > 1:
                # Count potential interactions
                total_interactions += (agents_in_region * (agents_in_region - 1)) // 2
                
                # Count actual coordinations (simplified for now)
                coordinated_agents += agents_in_region
        
        return {
            'coordinated_agents': coordinated_agents,
            'total_interactions': total_interactions
        }

    def save_model(self, path: str):
        """
        Save model to specified path.
        This is an alias for the save method to maintain compatibility with training script.
        
        Args:
            path: Path to save model to
        """
        self.save(path)

def main():
    """Example usage and testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test hierarchical communication')
    parser.add_argument('--net-file', required=True,
                      help='Path to SUMO network file')
    parser.add_argument('--route-file', required=True,
                      help='Path to SUMO route file')
    parser.add_argument('--num-episodes', type=int, default=100,
                      help='Number of episodes to train')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'max_region_size': 8,
        'min_region_size': 3,
        'overlap_threshold': 0.2,
        'regional_weight': 0.3,
        'coordination_threshold': 0.7,
        'hierarchical': {
            'influence_radius': 150.0,
            'region_size': 3,
            'coordination_interval': 5,
            'message_size': 32,
            'coordination_threshold': 0.6
        }
    }
    
    # Initialize environment
    env = MultiAgentSumoEnvironment(
        net_file=args.net_file,
        route_file=args.route_file,
        use_gui=False
    )
    
    # Create manager
    manager = HierarchicalManager(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        net_file=args.net_file,
        config=config
    )
    
    # Add agents
    for tl_id in env.traffic_lights:
        manager.add_agent(tl_id)
    
    # Training loop
    try:
        print("\nStarting training...")
        for episode in range(args.num_episodes):
            states, _ = env.reset()
            episode_reward = defaultdict(float)
            done = False
            
            while not done:
                # Update regions based on current positions
                positions = env.get_agent_positions()
                manager.update_regions(positions)
                
                # Get actions
                actions = manager.step(states)
                
                # Execute in environment
                next_states, rewards, done, _, info = env.step(actions)
                
                # Update agents
                losses = manager.update(states, actions, rewards, next_states,
                                     {tl: done for tl in states}, info)
                
                # Track rewards
                for tl_id, reward in rewards.items():
                    episode_reward[tl_id] += reward
                
                states = next_states
            
            # Log progress
            metrics = manager.get_metrics()
            print(f"\nEpisode {episode + 1}/{args.num_episodes}")
            print(f"Average Reward: {metrics['avg_reward']:.2f}")
            print(f"Number of Regions: {metrics['num_regions']}")
            print(f"Average Region Size: {metrics['avg_region_size']:.1f}")
            print(f"Message Rate: {metrics['message_rate']:.1f}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Save final state
        manager.save('models/hierarchical_final')
        env.close()

if __name__ == "__main__":
    main()