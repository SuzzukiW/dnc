# src/agents/cooperative_dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
from collections import deque, namedtuple, defaultdict
import random
from pathlib import Path

from src.models.dqn_network import DQNNetwork

class SharedReplayBuffer:
    """Shared replay buffer for all agents"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.agent_memories = defaultdict(list)  # Separate memories by state size
        self.Transition = namedtuple('Transition', 
                                   ('state', 'action', 'reward', 'next_state', 'done', 'agent_id'))
        
    def push(self, state, action, reward, next_state, done, agent_id):
        """Save experience to shared memory"""
        transition = self.Transition(state, action, reward, next_state, done, agent_id)
        self.memory.append(transition)
        state_size = len(state)
        
        # Keep agent memories within capacity
        if len(self.agent_memories[state_size]) >= self.capacity:
            self.agent_memories[state_size].pop(0)
        self.agent_memories[state_size].append(transition)
    
    def sample(self, batch_size, state_size=None):
        """Sample batch of transitions, optionally filtered by state size"""
        if state_size is not None and state_size in self.agent_memories:
            memory = self.agent_memories[state_size]
        else:
            memory = self.memory
            
        if len(memory) < batch_size:
            return random.sample(list(memory), len(memory))
        return random.sample(list(memory), batch_size)
    
    def __len__(self):
        return len(self.memory)

class CooperativeDQNAgent:
    """Individual cooperative DQN agent"""
    def __init__(self, agent_id, state_size, action_size, shared_memory, neighbor_ids, config):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.shared_memory = shared_memory
        self.neighbor_ids = neighbor_ids
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.95)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 32)
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    
    def remember(self, state, action, reward, next_state, done):
        """Store transition in shared memory"""
        self.shared_memory.push(state, action, reward, next_state, done, self.agent_id)
    
    def act(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
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
        
        transitions = self.shared_memory.sample(self.batch_size)
        
        # Filter transitions for this agent and similar sized states
        my_transitions = []
        for t in transitions:
            # Only use transitions from agents with same state size
            if len(t.state) == self.state_size:
                my_transitions.append(t)
        
        # If not enough valid transitions, skip update
        if len(my_transitions) < self.batch_size // 2:  # Allow half batch size minimum
            return
        
        # Sample from valid transitions
        if len(my_transitions) > self.batch_size:
            my_transitions = random.sample(my_transitions, self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([t.state for t in my_transitions]).to(self.device)
        actions = torch.LongTensor([t.action for t in my_transitions]).to(self.device)
        rewards = torch.FloatTensor([t.reward + global_reward for t in my_transitions]).to(self.device)
        next_states = torch.FloatTensor([t.next_state for t in my_transitions]).to(self.device)
        dones = torch.FloatTensor([t.done for t in my_transitions]).to(self.device)
        
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
        self.shared_memory = SharedReplayBuffer(config.get('memory_size', 100000))
        self.agents = {}
        
        # Create an agent for each traffic light
        for agent_id in agent_ids:
            neighbor_ids = neighbor_map.get(agent_id, [])
            self.agents[agent_id] = CooperativeDQNAgent(
                agent_id=agent_id,
                state_size=state_size,
                action_size=action_size,
                shared_memory=self.shared_memory,
                neighbor_ids=neighbor_ids,
                config=config
            )
    
    def act(self, states, training=True):
        """Get actions for all agents"""
        actions = {}
        for agent_id, state in states.items():
            actions[agent_id] = self.agents[agent_id].act(state, training)
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