"""
Prioritized Experience Replay - samples transitions based on TD-error magnitude.
High-error transitions (where the model was most wrong) are sampled more frequently.
"""

import numpy as np
from collections import deque
from typing import Tuple


class SumTree:
    """Efficient binary tree for priority-based sampling."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """Update parent nodes recursively."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf corresponding to priority sum 's'."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        return self.tree[0]
    
    def add(self, priority: float, data: object):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Key improvements over uniform sampling:
    - High TD-error transitions sampled 5-10x more frequently
    - Importance sampling corrects bias
    - Learns from mistakes faster
    """
    
    def __init__(
        self,
        capacity: int = 500000,
        alpha: float = 0.6,           # Priority exponent (0=uniform, 1=full priority)
        beta_start: float = 0.4,      # IS weight (anneals to 1.0)
        beta_frames: int = 100000,
        epsilon: float = 0.01,        # Minimum priority
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.max_priority = 1.0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store with max priority to ensure new experiences are sampled."""
        data = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, data)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch with priorities."""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size
        
        # Anneal beta
        self.beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            
            if data is None:
                continue
            
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)
        
        # Importance sampling weights
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** -self.beta
        weights /= weights.max()
        
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: list, td_errors: np.ndarray):
        """Update priorities after learning."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def is_ready(self, warmup_steps: int = 10000) -> bool:
        return self.tree.n_entries >= warmup_steps
    
    def __len__(self) -> int:
        return self.tree.n_entries