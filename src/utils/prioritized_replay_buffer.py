# src/utils/prioritized_replay_buffer.py

import numpy as np
from typing import Tuple, Dict
import tensorflow as tf

class PrioritizedReplayBuffer:
    def __init__(self, size, alpha, state_dim, action_dim, beta_start=0.4, beta_end=1.0, beta_steps=100000):
        self.size = size
        self.alpha = alpha
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Beta annealing parameters for importance sampling
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.beta = beta_start
        self.step_count = 0
        
        # Storage - explicitly match environment dimensions
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_states = np.zeros((size, state_dim), dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        
        # Priority storage with segment tree for efficient sampling
        self.max_priority = 1.0
        self.tree_capacity = 1
        while self.tree_capacity < size:
            self.tree_capacity *= 2
            
        self.sum_tree = np.zeros(2 * self.tree_capacity - 1)
        self.min_tree = np.float32(np.inf) * np.ones(2 * self.tree_capacity - 1)
        self.priorities = np.zeros(size, dtype=np.float32)
        self.attention_states = [{} for _ in range(size)]
        
        self.pos = 0
        self.size_now = 0
        
    def _propagate_sum(self, idx, change):
        """Propagate changes up the sum tree."""
        parent = (idx - 1) // 2
        self.sum_tree[parent] += change
        if parent != 0:
            self._propagate_sum(parent, change)
            
    def _propagate_min(self, idx, value):
        """Propagate changes up the min tree."""
        parent = (idx - 1) // 2
        left = 2 * parent + 1
        right = left + 1
        self.min_tree[parent] = min(self.min_tree[left], self.min_tree[right])
        if parent != 0:
            self._propagate_min(parent, value)
            
    def _update_tree(self, idx, priority):
        """Update both sum and min trees."""
        change = priority - self.sum_tree[idx]
        self.sum_tree[idx] = priority
        self._propagate_sum(idx, change)
        
        self.min_tree[idx] = priority
        self._propagate_min(idx, priority)
        
    def _get_priority(self, error):
        """Convert TD-error to priority."""
        return np.power(np.abs(error) + 1e-6, self.alpha)
        
    def add(self, state, action, reward, next_state, done, priority, attention_states=None):
        """Add experience to buffer with priority."""
        try:
            # Ensure state and action dimensions match buffer
            if isinstance(state, tf.Tensor):
                state = state.numpy()
            if isinstance(action, tf.Tensor):
                action = action.numpy()
            if isinstance(next_state, tf.Tensor):
                next_state = next_state.numpy()
                
            # Reshape if needed
            if len(state.shape) > 1:
                state = state.reshape(-1)[:self.state_dim]
            if len(action.shape) > 1:
                action = action.reshape(-1)[:self.action_dim]
            if len(next_state.shape) > 1:
                next_state = next_state.reshape(-1)[:self.state_dim]
                
            # Store experience
            idx = self.pos + self.tree_capacity - 1
            self.states[self.pos] = state
            self.actions[self.pos] = action
            self.rewards[self.pos] = reward
            self.next_states[self.pos] = next_state
            self.dones[self.pos] = done
            
            # Update priorities
            priority = self._get_priority(priority)
            self.max_priority = max(self.max_priority, priority)
            self._update_tree(idx, priority)
            self.priorities[self.pos] = priority
            
            if attention_states:
                self.attention_states[self.pos] = attention_states
                
            self.pos = (self.pos + 1) % self.size
            self.size_now = min(self.size_now + 1, self.size)
            
        except Exception as e:
            print(f"Error adding to buffer: {e}")
            print(f"State shape: {state.shape}, expected: {self.state_dim}")
            print(f"Action shape: {action.shape}, expected: {self.action_dim}")
            
    def sample(self, batch_size, beta=None):
        """Sample a batch of experiences with importance sampling."""
        if beta is None:
            # Update beta according to schedule
            fraction = min(self.step_count / self.beta_steps, 1.0)
            self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)
            beta = self.beta
        
        # Calculate segment boundaries
        segment = self.sum_tree[0] / batch_size
        priorities = np.zeros(batch_size, dtype=np.float32)
        indices = np.zeros(batch_size, dtype=np.int32)
        states = np.zeros((batch_size, self.state_dim), dtype=np.float32)
        actions = np.zeros((batch_size, self.action_dim), dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        next_states = np.zeros((batch_size, self.state_dim), dtype=np.float32)
        dones = np.zeros(batch_size, dtype=np.float32)
        
        # Sample from each segment
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            mass = np.random.uniform(a, b)
            
            # Traverse tree to find index
            idx = 0
            while idx < self.tree_capacity - 1:
                left = 2 * idx + 1
                if mass <= self.sum_tree[left]:
                    idx = left
                else:
                    mass -= self.sum_tree[left]
                    idx = left + 1
                    
            # Get sample index
            sample_idx = idx - (self.tree_capacity - 1)
            priorities[i] = self.priorities[sample_idx]
            indices[i] = sample_idx
            
            # Get experience
            states[i] = self.states[sample_idx]
            actions[i] = self.actions[sample_idx]
            rewards[i] = self.rewards[sample_idx]
            next_states[i] = self.next_states[sample_idx]
            dones[i] = self.dones[sample_idx]
            
        # Calculate importance sampling weights
        total = self.size_now if self.size_now < self.size else self.size
        max_weight = (total * np.min(priorities)) ** (-beta)
        weights = ((total * priorities) ** (-beta)) / max_weight
        
        self.step_count += 1
        
        return (
            states, actions, rewards, next_states, dones,
            indices, weights.astype(np.float32)
        )
        
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        for idx, priority in zip(indices, priorities):
            priority = self._get_priority(priority)
            tree_idx = idx + self.tree_capacity - 1
            self._update_tree(tree_idx, priority)
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
            
    def get_all(self):
        """Get all experiences in the buffer."""
        return (
            self.states[:self.size_now],
            self.actions[:self.size_now],
            self.rewards[:self.size_now],
            self.next_states[:self.size_now],
            self.dones[:self.size_now]
        )
        
    def __len__(self):
        return self.size_now