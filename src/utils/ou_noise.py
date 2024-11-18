import numpy as np
import random

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process.
        
        Params
        ======
            size (tuple): Dimension of noise vector
            seed (int): Random seed
            mu (float): Long-term mean
            theta (float): Mean reversion strength
            sigma (float): Noise volatility
        """
        if isinstance(size, int):
            size = (size,)
        elif isinstance(size, tuple) and len(size) == 1:
            size = size[0]
        
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        
        self.state = None
        self.reset()
    
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.copy(self.mu)
    
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(size=x.shape)
        self.state = x + dx
        return self.state
