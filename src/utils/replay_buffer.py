# src/utils/replay_buffer.py

import numpy as np
import random
import tensorflow as tf

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, epsilon=0.01):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, reward, next_state, neighbor_next_state, neighbor_next_action, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, neighbor_next_state, neighbor_next_action, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, neighbor_next_state, neighbor_next_action, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, neighbor_next_states, neighbor_next_actions, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(neighbor_next_states),
            np.array(neighbor_next_actions),
            np.array(dones, dtype=np.float32),
            weights,
            indices
        )

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            if isinstance(td_error, np.ndarray):
                td_error = td_error.item()
            elif isinstance(td_error, tf.Tensor):
                td_error = td_error.numpy().item()
            else:
                td_error = float(td_error)
            
            self.priorities[idx] = (abs(td_error) + self.epsilon) ** self.alpha

    def __len__(self):
        return len(self.buffer)