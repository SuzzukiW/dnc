# src/utils/state_dimensionality_reducer.py

import torch
import torch.nn as nn

class StateDimensionalityReducer(nn.Module):
    """Advanced state dimensionality reducer with multiple reduction strategies."""
    
    def __init__(self, input_size: int, output_size: int = 32, reduction_method: str = 'adaptive'):
        """
        Initialize state dimensionality reducer.
        
        Args:
            input_size: Original state vector size
            output_size: Desired output state vector size (default 32)
            reduction_method: Method to reduce dimensionality 
        """
        super(StateDimensionalityReducer, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.reduction_method = reduction_method
        
        # Create a reduction network that can handle very large inputs
        reduction_stages = []
        
        # Dynamically determine layer sizes
        layer_sizes = [
            input_size,
            max(input_size, 160),
            max(input_size // 2, 80),
            output_size
        ]
        
        # Create layers
        for i in range(len(layer_sizes) - 1):
            reduction_stages.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            # Add ReLU and BatchNorm after each layer except the last
            if i < len(layer_sizes) - 2:
                reduction_stages.append(nn.ReLU())
                reduction_stages.append(nn.BatchNorm1d(layer_sizes[i+1]))
        
        self.reducer = nn.Sequential(*reduction_stages)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduce state dimensionality.
        
        Args:
            x: Input state tensor
        
        Returns:
            Reduced dimensionality state tensor
        """
        # Ensure input is a 2D tensor
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Flatten if more than 2 dimensions
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Ensure input is float
        x = x.float()
        
        # Truncate or pad to match input size
        if x.size(1) > self.input_size:
            x = x[:, :self.input_size]
        elif x.size(1) < self.input_size:
            padding = torch.zeros(
                x.size(0), 
                self.input_size - x.size(1), 
                device=x.device,
                dtype=x.dtype
            )
            x = torch.cat([x, padding], dim=1)
        
        # Apply reduction
        try:
            reduced = self.reducer(x)
        except RuntimeError as e:
            print(f"Reduction error: {e}")
            print(f"Reducer input shape: {x.shape}")
            print(f"Reducer layers:")
            for i, layer in enumerate(self.reducer):
                print(f"Layer {i}: {layer}")
            raise
        
        return reduced
