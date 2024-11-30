# src/utils/prioritized_replay_buffer_maddpg.py

import numpy as np
from typing import Tuple, List, Dict

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
            # Return with very low priority to avoid selecting this again
            return leaf_idx, 1e-6, None

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
        n_agents: int,
        state_dims: List[int],
        action_dims: List[int],
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 100000
    ):
        """
        Initialize a prioritized replay buffer for MADDPG.
        
        Args:
            size: Maximum size of the buffer
            alpha: Priority exponent parameter
            n_agents: Number of agents in the environment
            state_dims: List of state dimensions for each agent
            action_dims: List of action dimensions for each agent
            beta_start: Initial importance sampling weight
            beta_end: Final importance sampling weight
            beta_steps: Number of steps to anneal beta from start to end
        """
        self.size = size
        self.alpha = alpha
        self.n_agents = n_agents
        self.state_dims = state_dims
        self.action_dims = action_dims

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
        states: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_states: Dict[str, np.ndarray],
        dones: Dict[str, bool],
        td_error: float
    ):
        """
        Add a new experience to the buffer.
        
        Args:
            states: Dictionary of states for each agent
            actions: Dictionary of actions for each agent
            rewards: Dictionary of rewards for each agent
            next_states: Dictionary of next states for each agent
            dones: Dictionary of done flags for each agent
            td_error: TD error for prioritization
        """
        priority = self._get_priority(td_error)
        data = (states, actions, rewards, next_states, dones)
        self.sum_tree.add(priority, data)
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], ...]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple containing:
            - Dictionary of states for each agent
            - Dictionary of actions for each agent
            - Dictionary of rewards for each agent
            - Dictionary of next states for each agent
            - Dictionary of done flags for each agent
            - Array of indices for updating priorities
            - Array of importance sampling weights
        """
        if self.sum_tree.n_entries == 0:
            raise ValueError("The SumTree is empty. Cannot sample.")

        batch_size = min(batch_size, self.sum_tree.n_entries)
        batch = []
        indices = []
        priorities = []

        segment = self.sum_tree.total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.sum_tree.get_leaf(s)
            if data is None:
                # Handle invalid data by creating zero-filled experience
                states = {str(i): np.zeros(dim) for i, dim in enumerate(self.state_dims)}
                actions = {str(i): np.zeros(dim) for i, dim in enumerate(self.action_dims)}
                rewards = {str(i): 0.0 for i in range(self.n_agents)}
                next_states = {str(i): np.zeros(dim) for i, dim in enumerate(self.state_dims)}
                dones = {str(i): True for i in range(self.n_agents)}
                data = (states, actions, rewards, next_states, dones)
            batch.append(data)
            indices.append(idx)
            priorities.append(p)

        states, actions, rewards, next_states, dones = zip(*batch)

        # Combine experiences across the batch
        combined_states = {
            agent_id: np.array([states[b][agent_id] for b in range(batch_size)])
            for agent_id in states[0].keys()
        }
        combined_actions = {
            agent_id: np.array([actions[b][agent_id] for b in range(batch_size)])
            for agent_id in actions[0].keys()
        }
        combined_rewards = {
            agent_id: np.array([rewards[b][agent_id] for b in range(batch_size)])
            for agent_id in rewards[0].keys()
        }
        combined_next_states = {
            agent_id: np.array([next_states[b][agent_id] for b in range(batch_size)])
            for agent_id in next_states[0].keys()
        }
        combined_dones = {
            agent_id: np.array([dones[b][agent_id] for b in range(batch_size)])
            for agent_id in dones[0].keys()
        }

        probs = np.array(priorities) / self.sum_tree.total_priority
        beta = self.beta
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()

        self.beta = min(self.beta_end, self.beta + (self.beta_end - self.beta_start) / self.beta_steps)
        self.step_count += 1

        return (
            combined_states,
            combined_actions,
            combined_rewards,
            combined_next_states,
            combined_dones,
            np.array(indices),
            weights.astype(np.float32)
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Array of indices to update
            td_errors: Array of new TD errors for each experience
        """
        for idx, td_error in zip(indices, td_errors):
            priority = self._get_priority(td_error)
            self.sum_tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.sum_tree.n_entries
