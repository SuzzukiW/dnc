"""
Segment tree data structure for efficient priority-based sampling.
Implements both Sum and Min segment trees.
"""

import operator
from typing import Callable

class SegmentTree:
    """Base class for segment trees."""
    
    def __init__(self, capacity: int, operation: Callable, neutral_element: float):
        """Initialize segment tree.
        
        Args:
            capacity (int): Maximum size of segment tree
            operation (callable): Operation to perform on tree nodes (e.g., min or sum)
            neutral_element (float): Neutral element for the operation
        """
        self.capacity = capacity
        self.operation = operation
        self.neutral_element = neutral_element
        
        # Allocate tree storage
        self.tree = [neutral_element for _ in range(2 * capacity)]
        
    def _reduce(self, start: int = 0, end: int = None) -> float:
        """Returns result of operation over a contiguous subsequence."""
        if end is None:
            end = self.capacity
            
        # Init with valid boundaries for the loop
        start += self.capacity
        end += self.capacity
        
        # Init result with neutral element
        result = self.neutral_element
        
        # Traverse through all required nodes
        while start < end:
            if start & 1:
                result = self.operation(result, self.tree[start])
                start += 1
            if end & 1:
                end -= 1
                result = self.operation(result, self.tree[end])
            start >>= 1
            end >>= 1
            
        return result
    
    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val
        
        # Propagate change up through tree
        idx >>= 1
        while idx >= 1:
            self.tree[idx] = self.operation(
                self.tree[2 * idx],
                self.tree[2 * idx + 1]
            )
            idx >>= 1
            
    def __getitem__(self, idx: int) -> float:
        """Get value from tree."""
        return self.tree[idx + self.capacity]

class SumSegmentTree(SegmentTree):
    """Segment tree that finds prefix sums."""
    
    def __init__(self, capacity: int):
        """Initialize SumSegmentTree."""
        super().__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )
        
    def sum(self, start: int = 0, end: int = None) -> float:
        """Returns sum over [start, end)."""
        return self._reduce(start, end)
    
    def find_prefixsum_idx(self, prefixsum: float) -> int:
        """Find the highest index `i` in the tree with sum[0, i] <= prefixsum.
        
        Args:
            prefixsum (float): Upper bound on prefix sum to find
            
        Returns:
            int: Highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        
        while idx < self.capacity:  # While non-leaf
            left = 2 * idx
            right = left + 1
            
            if self.tree[left] > prefixsum:
                idx = left
            else:
                prefixsum -= self.tree[left]
                idx = right
                
        return idx - self.capacity

class MinSegmentTree(SegmentTree):
    """Segment tree that finds minimum values."""
    
    def __init__(self, capacity: int):
        """Initialize MinSegmentTree."""
        super().__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )
        
    def min(self, start: int = 0, end: int = None) -> float:
        """Returns min over [start, end)."""
        return self._reduce(start, end)
