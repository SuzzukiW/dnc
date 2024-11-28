# src/utils/prioritized_replay_buffer.py

import numpy as np
from typing import Tuple
import math

class SumTree:
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
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

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

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
        """
        Initializes the Prioritized Replay Buffer.

        Args:
            size (int): Maximum number of experiences the buffer can hold.
            alpha (float): How much prioritization is used (0 - no prioritization, 1 - full prioritization).
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            max_neighbors (int): Maximum number of neighbors for each agent.
            beta_start (float): Initial value of beta for importance sampling.
            beta_end (float): Final value of beta after annealing.
            beta_steps (int): Number of steps over which beta is annealed from beta_start to beta_end.
        """
        self.size = size
        self.alpha = alpha
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_neighbors = max_neighbors

        # Beta annealing parameters for importance sampling
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.beta = beta_start
        self.step_count = 0

        # Storage for experiences
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_states = np.zeros((size, state_dim), dtype=np.float32)
        self.neighbor_next_states = np.zeros((size, max_neighbors, state_dim), dtype=np.float32)
        self.neighbor_next_actions = np.zeros((size, max_neighbors, action_dim), dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)

        # Priority storage using SumTree
        self.sum_tree = SumTree(size)
        self.max_priority = 1.0

    def _get_priority(self, error: float) -> float:
        """Convert TD-error to priority."""
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
        """
        Add experience to buffer with priority.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            neighbor_next_state (np.ndarray): Next states of neighboring agents.
            neighbor_next_action (np.ndarray): Next actions of neighboring agents.
            done (bool): Whether the episode is done.
            td_error (float): TD-error for priority calculation.
        """
        priority = self._get_priority(td_error)
        data = (state, action, reward, next_state, neighbor_next_state, neighbor_next_action, done)
        self.sum_tree.add(priority, data)
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences with importance sampling weights.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            Tuple containing sampled states, actions, rewards, next_states, neighbor_next_states,
            neighbor_next_actions, dones, indices, and IS weights.
        """
        batch = []
        indices = []
        priorities = []

        segment = self.sum_tree.total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.sum_tree.get_leaf(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(p)

        # Convert batch to separate arrays
        states, actions, rewards, next_states, neighbor_next_states, neighbor_next_actions, dones = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        neighbor_next_states = np.array(neighbor_next_states)
        neighbor_next_actions = np.array(neighbor_next_actions)
        dones = np.array(dones)

        # Calculate importance sampling weights
        probs = np.array(priorities) / self.sum_tree.total_priority
        beta = self.beta
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()

        # Anneal beta towards beta_end
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
            np.array(indices),
            weights.astype(np.float32)
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update the priorities of sampled experiences based on new TD-errors.

        Args:
            indices (np.ndarray): Array of indices of experiences to update.
            td_errors (np.ndarray): Array of TD-errors corresponding to the indices.
        """
        for idx, td_error in zip(indices, td_errors):
            priority = self._get_priority(td_error)
            self.sum_tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        """Returns the current size of internal memory."""
        return self.sum_tree.n_entries
