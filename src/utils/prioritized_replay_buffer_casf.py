# src/utils/prioritized_replay_buffer_casf.py

import numpy as np
from typing import Tuple

class SumTree:
    """
    SumTree data structure for prioritized replay buffer.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority

        # Propagate the changes up
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, v):
        parent_idx = 0

        while True:
            left = 2 * parent_idx + 1
            right = left + 1

            if left >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left]:
                    parent_idx = left
                else:
                    v -= self.tree[left]
                    parent_idx = right

        data_idx = leaf_idx - (self.capacity - 1)

        # Ensure the data index is valid and within the initialized entries
        if data_idx < 0 or data_idx >= self.n_entries:
            # Resample if invalid index
            if self.total_priority == 0:
                # If total priority is zero, return a default value to prevent infinite recursion
                return leaf_idx, 1e-6, (np.zeros(self.capacity, dtype=np.float32),) * 7
            else:
                return self.get_leaf(np.random.uniform(0, self.total_priority))

        data = self.data[data_idx]
        return leaf_idx, self.tree[leaf_idx], data

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(
        self,
        size: int,
        alpha: float,
        state_dim: int,
        action_dim: int,
        max_neighbors: int,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 100000
    ):
        self.size = size
        self.alpha = alpha
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_neighbors = max_neighbors

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.beta = beta_start
        self.step_count = 0

        self.sum_tree = SumTree(size)
        self.max_priority = 1.0

    def _get_priority(self, error: float) -> float:
        return (np.abs(error) + 1e-6) ** self.alpha

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        neighbor_next_state: np.ndarray,
        neighbor_next_action: np.ndarray,
        done: bool,
        td_error: float
    ):
        # Store states as is without reshaping
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        neighbor_next_state = np.array(neighbor_next_state, dtype=np.float32)
        neighbor_next_action = np.array(neighbor_next_action, dtype=np.float32)
        action = np.array(action, dtype=np.float32)

        priority = self._get_priority(td_error)
        data = (state, action, reward, next_state, neighbor_next_state, neighbor_next_action, done)
        self.sum_tree.add(priority, data)
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        if self.sum_tree.n_entries == 0:
            raise ValueError("The SumTree is empty. Cannot sample.")

        batch_size = min(batch_size, self.sum_tree.n_entries)

        states = np.zeros((batch_size, self.state_dim), dtype=np.float32)
        actions = np.zeros((batch_size, self.action_dim), dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        next_states = np.zeros((batch_size, self.state_dim), dtype=np.float32)
        neighbor_next_states = np.zeros((batch_size, self.max_neighbors, self.state_dim), dtype=np.float32)
        neighbor_next_actions = np.zeros((batch_size, self.max_neighbors, self.action_dim), dtype=np.float32)
        dones = np.zeros(batch_size, dtype=np.float32)
        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)

        segment = self.sum_tree.total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.sum_tree.get_leaf(s)

            # Ensure states have correct dimensions
            states[i] = np.array(data[0], dtype=np.float32)[:self.state_dim]
            actions[i] = np.array(data[1], dtype=np.float32)[:self.action_dim]
            rewards[i] = data[2]
            next_states[i] = np.array(data[3], dtype=np.float32)[:self.state_dim]
            neighbor_next_states[i] = np.array(data[4], dtype=np.float32)[:, :self.state_dim]
            neighbor_next_actions[i] = np.array(data[5], dtype=np.float32)[:, :self.action_dim]
            dones[i] = data[6]
            indices[i] = idx
            weights[i] = (self.size * p / self.sum_tree.total_priority) ** (-self.beta)

        weights /= weights.max()
        self.beta = min(self.beta_end, self.beta + (self.beta_end - self.beta_start) / self.beta_steps)
        self.step_count += 1

        return (
            states,
            actions,
            rewards,
            next_states,
            neighbor_next_states,
            neighbor_next_actions,
            dones,
            indices,
            weights.astype(np.float32)
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, td_error in zip(indices, td_errors):
            priority = self._get_priority(td_error)
            self.sum_tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.sum_tree.n_entries