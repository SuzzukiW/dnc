# experiments/scenarios/communication/independent_agents.py

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict, deque

class IndependentAgent:
    """
    Independent agent implementation for traffic light control.
    Each agent acts independently without knowledge of other agents.
    """
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        
        # Simple network architecture
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        ).to(device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)  # Simple replay buffer
        
    def select_action(self, state, epsilon=0.1):
        """Select an action using epsilon-greedy policy"""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.network(state)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=32):
        """Train the agent using a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for idx in batch:
            s, a, r, ns, d = self.memory[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q = self.network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class IndependentAgentsManager:
    """
    Manages multiple independent agents for traffic light control
    """
    def __init__(self, traffic_light_ids, state_size, action_size, device='cpu'):
        self.agents = {}
        self.device = device
        
        # Create an agent for each traffic light
        for tl_id in traffic_light_ids:
            self.agents[tl_id] = IndependentAgent(
                state_size=state_size,
                action_size=action_size,
                device=device
            )
    
    def select_actions(self, states, epsilon=0.1):
        """Select actions for all agents"""
        actions = {}
        for tl_id, state in states.items():
            actions[tl_id] = self.agents[tl_id].select_action(state, epsilon)
        return actions
    
    def store_transitions(self, transitions):
        """Store transitions for all agents"""
        for tl_id, transition in transitions.items():
            self.agents[tl_id].store_transition(*transition)
    
    def train_agents(self, batch_size=32):
        """Train all agents"""
        losses = []
        for tl_id, agent in self.agents.items():
            loss = agent.train(batch_size)
            if loss is not None:
                losses.append(loss)
        
        return np.mean(losses) if losses else None

    def save_agents(self, save_dir):
        """Save all agents"""
        os.makedirs(save_dir, exist_ok=True)
        for tl_id, agent in self.agents.items():
            save_path = os.path.join(save_dir, f'agent_{tl_id}.pth')
            torch.save(agent.network.state_dict(), save_path)
    
    def load_agents(self, save_dir):
        """Load all agents"""
        for tl_id, agent in self.agents.items():
            save_path = os.path.join(save_dir, f'agent_{tl_id}.pth')
            if os.path.exists(save_path):
                agent.network.load_state_dict(torch.load(save_path))