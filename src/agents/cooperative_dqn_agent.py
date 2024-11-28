# src/agents/cooperative_dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, defaultdict, deque
from typing import Dict, List, Optional, Any
from pathlib import Path
import torch.cuda.amp as amp
import torch.nn.functional as F

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
        # Ensure action is a numpy array
        if isinstance(action, (int, float)):
            action = np.array([action], dtype=np.float32)
        elif isinstance(action, list):
            action = np.array(action, dtype=np.float32)
        elif isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
            
        # Ensure state and next_state are numpy arrays
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
            
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
        # Filter by state size if specified
        if state_size is not None:
            memory_pool = self.agent_memories[state_size]
        else:
            memory_pool = self.memory
        
        # Ensure enough samples
        if len(memory_pool) < batch_size:
            return None
            
        # Sample transitions with matching state sizes
        batch = random.sample(memory_pool, batch_size)
        return batch
    
    def __len__(self):
        """
        Get current memory size
        
        Returns:
            Number of transitions in memory
        """
        return len(self.memory)

class CooperativeDQNAgent:
    def __init__(self, agent_id, state_size, action_size, shared_memory, neighbor_ids, config):
        self.agent_id = agent_id
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        
        self.shared_memory = shared_memory
        self.neighbor_ids = neighbor_ids
        self.config = config
        
        # Hyperparameters
        self.gamma = float(config.get('gamma', 0.99))
        self.learning_rate = float(config.get('learning_rate', 0.001))
        self.epsilon = float(config.get('epsilon_start', 1.0))
        self.epsilon_min = float(config.get('epsilon_min', 0.01))
        self.epsilon_decay = float(config.get('epsilon_decay', 0.995))
        self.batch_size = int(config.get('batch_size', 64))
        self.regional_weight = float(config.get('regional_weight', 0.3))
        
        # Setup device with Apple Silicon support
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Enable float32 matrix multiplication on Apple Silicon
        if self.device == torch.device("mps"):
            torch.set_default_dtype(torch.float32)
        
        # Setup networks with specific state size for this agent
        hidden_sizes = [
            min(384, max(64, self.state_size * 2)),
            min(192, max(32, self.state_size)),
            min(96, max(16, self.state_size // 2))
        ]
        
        # Initialize policy and target networks with agent-specific dimensions
        self.policy_net = DQNNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_sizes=hidden_sizes
        ).to(self.device)
        
        self.target_net = DQNNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_sizes=hidden_sizes
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer setup with Apple Silicon optimizations
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
            eps=1e-7  # Increased epsilon for better numerical stability
        )
        
        # Enable AMP (Automatic Mixed Precision) for better performance
        self.scaler = amp.GradScaler()
        self.use_amp = True
    
    def remember(self, state, action, reward, next_state, done):
        """Store transition in shared memory"""
        # Ensure action is a numpy array
        if isinstance(action, (int, float)):
            action = np.array([action], dtype=np.float32)
        elif isinstance(action, list):
            action = np.array(action, dtype=np.float32)
        elif isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
            
        # Ensure state and next_state are numpy arrays
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
            
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
                
                # Get action index with highest Q-value
                action_index = q_values.argmax(dim=1).long()
                
                # Convert to one-hot action
                action = torch.zeros_like(q_values[0])
                action[action_index] = 1
                action = action.cpu().numpy()
        
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
                # Only share weights if state sizes match and action sizes match
                if (neighbor.state_size == self.state_size and 
                    neighbor.action_size == self.action_size):
                    self._share_weights(neighbor.policy_net)
    
    def _share_weights(self, other_network, share_ratio=0.1):
        """Share a portion of weights with another network"""
        # Only share weights if network architectures match
        if (list(self.policy_net.parameters())[0].shape == 
            list(other_network.parameters())[0].shape):
            for param, other_param in zip(self.policy_net.parameters(), 
                                        other_network.parameters()):
                if param.shape == other_param.shape:
                    param.data.copy_(share_ratio * other_param.data + 
                                   (1 - share_ratio) * param.data)
    
    def replay(self, global_reward=0):
        """Train on batch from shared memory with both local and global rewards"""
        # Sample batch of transitions with matching state size
        batch = self.shared_memory.sample(self.batch_size, state_size=self.state_size)
        if batch is None:
            return
        
        states_array = np.array([t.state for t in batch])
        actions_array = np.vstack([t.action for t in batch])
        rewards_array = np.array([t.reward for t in batch])
        next_states_array = np.array([t.next_state for t in batch])
        dones_array = np.array([float(t.done) for t in batch])
        
        # Convert numpy arrays to tensors
        states = torch.FloatTensor(states_array).to(self.device)
        actions = torch.LongTensor(actions_array).to(self.device)
        rewards = torch.FloatTensor(rewards_array).to(self.device)
        next_states = torch.FloatTensor(next_states_array).to(self.device)
        dones = torch.FloatTensor(dones_array).to(self.device)
        
        # Add global reward component
        rewards = rewards + global_reward
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Get current Q values
            current_q_values = self.policy_net(states)
            action_indices = actions.argmax(dim=1).long()  # Convert actions to indices
            state_action_values = current_q_values.gather(1, action_indices.unsqueeze(1))

            # Compute next state values
            with torch.no_grad():
                next_state_values = self.target_net(next_states).max(1)[0]
                next_state_values[dones.bool()] = 0.0  # Use boolean indexing
                expected_state_action_values = (next_state_values * self.gamma) + rewards

            # Compute loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model with gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

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
    def __init__(self, valid_traffic_lights, observation_spaces, action_spaces, neighbor_map, config):
        self.config = config
        self.shared_memory = SharedReplayBuffer(config.get('memory_size', 100000))
        self.agents = {}
        
        # Create agents with individual state sizes
        for agent_id in valid_traffic_lights:
            # Get state and action sizes from spaces
            agent_state_size = observation_spaces[agent_id].shape[0]
            agent_action_size = action_spaces[agent_id].shape[0]
            
            # Create agent with its specific dimensions
            self.agents[agent_id] = CooperativeDQNAgent(
                agent_id=agent_id,
                state_size=agent_state_size,
                action_size=agent_action_size,
                shared_memory=self.shared_memory,
                neighbor_ids=neighbor_map.get(agent_id, []),
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