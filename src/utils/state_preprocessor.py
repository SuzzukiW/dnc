# src/utils/state_preprocessor.py

import numpy as np
import torch
from typing import Dict, Any, Union

class StatePreprocessor:
    """Preprocesses environment states for MADDPG agents."""
    
    def __init__(self, target_state_size: int):
        """Initialize the state preprocessor.
        
        Args:
            target_state_size: The desired size of the processed state
        """
        self.target_state_size = target_state_size
        
    def preprocess_state(self, state: Union[np.ndarray, Dict[str, np.ndarray]]) -> torch.Tensor:
        """Preprocess a state for input to the neural network.
        
        Args:
            state: Raw state from the environment
            
        Returns:
            Preprocessed state as a torch tensor
        """
        if isinstance(state, dict):
            # If state is a dictionary, extract the numpy array
            state_array = next(iter(state.values()))
        else:
            state_array = state
            
        # Convert to numpy array if needed
        if not isinstance(state_array, np.ndarray):
            state_array = np.array(state_array)
            
        # Ensure state is 1D
        state_array = state_array.flatten()
        
        # Pad or truncate to match target size
        if len(state_array) < self.target_state_size:
            # Pad with zeros
            padded_state = np.zeros(self.target_state_size)
            padded_state[:len(state_array)] = state_array
            state_array = padded_state
        elif len(state_array) > self.target_state_size:
            # Truncate
            state_array = state_array[:self.target_state_size]
            
        # Convert to torch tensor
        return torch.FloatTensor(state_array).unsqueeze(0)  # Add batch dimension
