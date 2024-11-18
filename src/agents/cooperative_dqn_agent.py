# src/agents/cooperative_dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, defaultdict, deque
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.models.dqn_network import DQNNetwork

class SharedReplayBuffer:
    """Shared replay buffer for all agents"""
    def __init__(self, capacity):
        """
        Initialize shared replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
        # Separate memories by state size and agent
        self.agent_memories = defaultdict(list)
        
        # Flexible transition named tuple
        self.Transition = namedtuple('Transition', 
                                     ('state', 'action', 'reward', 'next_state', 'done', 'agent_id', 'state_size'))
    
    def push(self, state, action, reward, next_state, done, agent_id):
        """
        Store a transition in the shared memory
        
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Episode termination flag
            agent_id: ID of the agent
        """
        # Determine state size
        state_size = len(state) if isinstance(state, (list, np.ndarray)) else state.shape[0]
        
        # Create transition
        transition = self.Transition(
            state=state, 
            action=action, 
            reward=reward, 
            next_state=next_state, 
            done=done, 
            agent_id=agent_id,
            state_size=state_size
        )
        
        # Add to global memory
        self.memory.append(transition)
        
        # Add to agent-specific memory
        self.agent_memories[state_size].append(transition)
    
    def sample(self, batch_size, state_size=None):
        """
        Sample a batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
            state_size: Optional state size to filter transitions
        
        Returns:
            Batch of transitions
        """
        # Use state-specific memory if state_size is provided
        if state_size is not None:
            memory_pool = self.agent_memories.get(state_size, [])
        else:
            memory_pool = self.memory
        
        # Ensure enough samples
        if len(memory_pool) < batch_size:
            return None
        
        # Sample transitions
        batch = random.sample(memory_pool, batch_size)
        
        # Prepare batch tensors
        states = torch.FloatTensor([t.state for t in batch])
        actions = torch.FloatTensor([t.action for t in batch])
        rewards = torch.FloatTensor([t.reward for t in batch])
        next_states = torch.FloatTensor([t.next_state for t in batch])
        dones = torch.FloatTensor([float(t.done) for t in batch])
        agent_ids = [t.agent_id for t in batch]
        
        return states, actions, rewards, next_states, dones, agent_ids
    
    def __len__(self):
        """
        Get current memory size
        
        Returns:
            Number of transitions in memory
        """
        return len(self.memory)

class CooperativeDQNAgent:
    """Individual cooperative DQN agent"""
    def __init__(self, agent_id, state_size, action_size, shared_memory, neighbor_ids, config):
        """
        Initialize a Cooperative DQN Agent with adaptive network configuration
        
        Args:
            agent_id: Unique identifier for the agent
            state_size: Dimension of input state
            action_size: Dimension of output actions
            shared_memory: Shared replay buffer
            neighbor_ids: List of neighboring agent IDs
            config: Configuration dictionary
        """
        self.agent_id = agent_id
        
        # Ensure state_size and action_size are integers
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        
        self.shared_memory = shared_memory
        self.neighbor_ids = neighbor_ids
        self.config = config  # Store the entire config
        
        # Hyperparameters with more robust defaults
        self.gamma = float(config.get('gamma', 0.99))
        self.learning_rate = float(config.get('learning_rate', 0.001))
        self.epsilon = float(config.get('epsilon_start', 1.0))
        self.epsilon_min = float(config.get('epsilon_min', 0.01))
        self.epsilon_decay = float(config.get('epsilon_decay', 0.995))
        self.batch_size = int(config.get('batch_size', 64))
        
        # Regional learning parameters
        self.regional_weight = float(config.get('regional_weight', 0.3))
        self.use_regional_input = bool(config.get('use_regional_input', False))
        
        # Adaptive network configuration
        hidden_sizes = config.get('hidden_sizes', None)
        
        # Networks with adaptive architecture
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create policy network
        self.policy_net = DQNNetwork(
            state_size=self.state_size, 
            action_size=self.action_size, 
            hidden_sizes=hidden_sizes
        ).to(self.device)
        
        # Create target network
        self.target_net = DQNNetwork(
            state_size=self.state_size, 
            action_size=self.action_size, 
            hidden_sizes=hidden_sizes
        ).to(self.device)
        
        # Copy weights from policy to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer with adaptive learning rate
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5  # Add L2 regularization
        )
        
        # Freeze target network parameters
        for param in self.target_net.parameters():
            param.requires_grad = False
    
    def remember(self, state, action, reward, next_state, done):
        """Store transition in shared memory"""
        self.shared_memory.push(state, action, reward, next_state, done, self.agent_id)
    
    def act(self, state, training=True, regional_recommendation=None):
        """Select action using epsilon-greedy policy with continuous action support"""
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Exploration vs exploitation
        if training and random.random() < self.epsilon:
            # Random continuous action within the range [-1, 1]
            action = np.random.uniform(-1, 1, size=self.action_size)
        else:
            # Use policy network to select action
            with torch.no_grad():
                # Get Q-values from policy network
                q_values = self.policy_net(state_tensor)
                
                # Convert Q-values to continuous action
                # This assumes the network outputs a continuous action directly
                action = q_values.cpu().numpy()[0]
        
        # Apply regional recommendation if provided
        if regional_recommendation is not None:
            # Blend local action with regional recommendation
            action = (1 - self.regional_weight) * action + self.regional_weight * regional_recommendation
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return action
    
    def learn_from_neighbors(self, other_agents):
        """Learn from neighbor experiences"""
        neighbor_transitions = []
        for neighbor_id in self.neighbor_ids:
            if neighbor_id in other_agents:
                neighbor = other_agents[neighbor_id]
                # Only share weights if state sizes match
                if neighbor.state_size == self.state_size:
                    self._share_weights(neighbor.policy_net)
    
    def _share_weights(self, other_network, share_ratio=0.1):
        """Share a portion of weights with another network"""
        for param, other_param in zip(self.policy_net.parameters(), other_network.parameters()):
            param.data.copy_(share_ratio * other_param.data + (1 - share_ratio) * param.data)
    
    def replay(self, global_reward=0):
        """Train on batch from shared memory with both local and global rewards"""
        if len(self.shared_memory) < self.batch_size:
            return
        
        batch = self.shared_memory.sample(self.batch_size, self.state_size)
        
        if batch is None:
            return
        
        states, actions, rewards, next_states, dones, agent_ids = batch
        
        # Convert to tensors
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target net
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        
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
        # Convert path to Path object
        save_path = Path(path)
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'agent_id': self.agent_id,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, save_path)
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class MultiAgentDQN:
    """Manager class for multiple cooperative DQN agents"""
    def __init__(self, state_size, action_size, agent_ids, neighbor_map, config):
        """
        Initialize multi-agent DQN system

        Args:
            state_size: Default state size (can be overridden per agent)
            action_size: Default action size (can be overridden per agent)
            agent_ids: List of agent IDs
            neighbor_map: Mapping of agents to their neighbors
            config: Configuration dictionary
        """
        self.config = config  # Store the config
        self.shared_memory = SharedReplayBuffer(config.get('memory_size', 100000))
        self.agents = {}
        
        # Create an agent for each traffic light
        for agent_id in agent_ids:
            # Get state and action size for this specific agent
            agent_state_size = config.get(f'{agent_id}_state_size', state_size)
            agent_action_size = config.get(f'{agent_id}_action_size', action_size)
            
            neighbor_ids = neighbor_map.get(agent_id, [])
            self.agents[agent_id] = CooperativeDQNAgent(
                agent_id=agent_id,
                state_size=agent_state_size,
                action_size=agent_action_size,
                shared_memory=self.shared_memory,
                neighbor_ids=neighbor_ids,
                config=config
            )
    
    def act(self, states, training=True, regional_recommendations=None):
        """
        Get actions for all agents

        Args:
            states: Dictionary of states for each agent
            training: Whether in training mode
            regional_recommendations: Optional regional action recommendations

        Returns:
            Dictionary of actions for each agent
        """
        actions = {}
        for agent_id, state in states.items():
            # Get regional recommendation if provided
            regional_rec = (regional_recommendations.get(agent_id) 
                            if regional_recommendations else None)
            
            # Get action from the specific agent
            actions[agent_id] = self.agents[agent_id].act(
                state, 
                training=training, 
                regional_recommendation=regional_rec
            )
        
        return actions
    
    def step(self, states, actions, rewards, next_states, dones, global_reward=0):
        """Update all agents"""
        losses = {}
        
        # Store experiences
        for agent_id in states.keys():
            self.agents[agent_id].remember(
                states[agent_id],
                actions[agent_id],
                rewards[agent_id],
                next_states[agent_id],
                dones[agent_id]
            )
        
        # Update agents
        for agent_id, agent in self.agents.items():
            # Learn from neighbors
            agent.learn_from_neighbors(self.agents)
            # Update from experiences
            loss = agent.replay(global_reward)
            losses[agent_id] = loss
            
            # Update target network
            agent.update_target_network()
        
        return losses
    
    def save_agents(self, directory):
        """Save all agents"""
        for agent_id, agent in self.agents.items():
            path = f"{directory}/agent_{agent_id}.pt"
            agent.save(path)
    
    def load_agents(self, directory):
        """Load all agents"""
        for agent_id, agent in self.agents.items():
            path = f"{directory}/agent_{agent_id}.pt"
            agent.load(path)