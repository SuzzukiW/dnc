import numpy as np
import torch
from collections import namedtuple, deque
from typing import Dict, List, Tuple
import random

class SegmentTree:
    """Segment tree data structure for efficient priority sampling"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.n_entries % self.capacity
        self.data[idx] = data
        self.update(idx, priority)
        self.n_entries += 1

    def update(self, idx, priority):
        if idx not in self.pending_idx:
            self.pending_idx.add(idx)
            change = priority - self.tree[idx + self.capacity - 1]
            self.tree[idx + self.capacity - 1] = priority
            self._propagate(idx + self.capacity - 1, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        self.pending_idx.discard(dataIdx)
        return (idx, self.tree[idx], self.data[dataIdx])

class EnhancedReplayBuffer:
    """Enhanced replay buffer with prioritized experience replay and n-step returns"""
    def __init__(self, capacity: int, n_step: int = 3, gamma: float = 0.99, 
                 alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000):
        """
        Initialize enhanced replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
            n_step: Number of steps for n-step return
            gamma: Discount factor
            alpha: Priority exponent
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames over which to anneal beta
        """
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # Current frame number for beta annealing
        
        self.tree = SegmentTree(capacity)
        self.max_priority = 1.0
        
        # Buffer for n-step returns
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Transition tuple
        self.Transition = namedtuple('Transition', 
            ('state', 'action', 'reward', 'next_state', 'done', 'agent_id', 'state_size'))
            
        # Separate memories by state size and agent
        self.agent_memories = {}

    def _get_beta(self):
        """Get current beta value for importance sampling"""
        beta = self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames)
        self.frame = min(self.frame + 1, self.beta_frames)
        return min(1.0, beta)

    def _compute_n_step_return(self, n_step_buffer: deque) -> Tuple[float, torch.Tensor, bool]:
        """Compute n-step return from buffer"""
        reward = 0
        next_state = n_step_buffer[-1].next_state
        done = n_step_buffer[-1].done

        for idx, transition in enumerate(n_step_buffer):
            reward += (self.gamma ** idx) * transition.reward
            if transition.done:
                next_state = transition.next_state
                done = True
                break

        return reward, next_state, done

    def push(self, state, action, reward, next_state, done, agent_id):
        """Store a transition in the buffer"""
        # Convert inputs to numpy arrays if needed
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        if isinstance(action, (int, float)):
            action = np.array([action], dtype=np.float32)
        elif isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        # Create transition
        transition = self.Transition(state, action, reward, next_state, done, agent_id,
                                  len(state) if isinstance(state, (list, np.ndarray)) else state.shape[0])

        # Add to n-step buffer
        self.n_step_buffer.append(transition)

        # If n-step buffer is ready
        if len(self.n_step_buffer) == self.n_step:
            # Get n-step return
            n_step_reward, n_step_next_state, n_step_done = self._compute_n_step_return(self.n_step_buffer)
            
            # Create n-step transition
            n_step_transition = self.Transition(
                self.n_step_buffer[0].state,
                self.n_step_buffer[0].action,
                n_step_reward,
                n_step_next_state,
                n_step_done,
                self.n_step_buffer[0].agent_id,
                self.n_step_buffer[0].state_size
            )

            # Add to tree with max priority
            self.tree.add(self.max_priority, n_step_transition)

            # Add to agent-specific memories
            if n_step_transition.state_size not in self.agent_memories:
                self.agent_memories[n_step_transition.state_size] = []
            self.agent_memories[n_step_transition.state_size].append(n_step_transition)

    def sample(self, batch_size: int, state_size: int = None) -> Tuple[List, List, List]:
        """Sample a batch of transitions"""
        batch = []
        indices = []
        priorities = []
        
        # Get current beta for importance sampling
        beta = self._get_beta()
        
        # Compute sampling probabilities
        total_priority = self.tree.total()
        
        # Sample transitions
        for _ in range(batch_size):
            mass = random.random() * total_priority
            idx, priority, data = self.tree.get(mass)
            
            if state_size is None or data.state_size == state_size:
                batch.append(data)
                indices.append(idx)
                priorities.append(priority)
        
        # Compute importance sampling weights
        samples_priority = np.array(priorities)
        probs = samples_priority / total_priority
        weights = (self.capacity * probs) ** -beta
        weights = weights / weights.max()
        
        return batch, indices, weights

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities of sampled transitions"""
        for idx, priority in zip(indices, priorities):
            # Clip priority to prevent excessive values
            priority = min(max(priority, 1e-6), 1e3)
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority ** self.alpha)

    def __len__(self) -> int:
        """Return current size of the buffer"""
        return self.tree.n_entries
