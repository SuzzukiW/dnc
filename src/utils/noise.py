# src/utils/noise.py

import numpy as np
from typing import Optional

class OUNoise:
    """Ornstein-Uhlenbeck process noise generator."""
    
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15,
                 sigma: float = 0.2, seed: Optional[int] = None):
        """Initialize parameters and noise process.
        
        Args:
            size: Size of the action space
            mu: Mean of the noise distribution
            theta: Parameter controlling the mean reversion strength
            sigma: Parameter controlling the noise scale
            seed: Random seed
        """
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            
        self.reset()
        
    def reset(self):
        """Reset the internal state (= noise) to mean."""
        self.state = self.mu.copy()
        
    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample.
        
        Returns:
            Current noise state
        """
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.standard_normal(self.size)
        self.state += dx
        return self.state
    

class GaussianNoise:
    """Gaussian noise generator with decay."""
    
    def __init__(self, size: int, mu: float = 0.0, sigma: float = 0.2,
                 decay_rate: float = 0.9995, min_sigma: float = 0.01,
                 seed: Optional[int] = None):
        """Initialize parameters and noise process.
        
        Args:
            size: Size of the action space
            mu: Mean of the noise distribution
            sigma: Initial standard deviation
            decay_rate: Rate at which sigma decays
            min_sigma: Minimum value for sigma
            seed: Random seed
        """
        self.size = size
        self.mu = mu
        self.sigma = sigma
        self.decay_rate = decay_rate
        self.min_sigma = min_sigma
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            
    def sample(self) -> np.ndarray:
        """Generate and return a noise sample.
        
        Returns:
            Noise sample
        """
        noise = np.random.normal(self.mu, self.sigma, self.size)
        self.sigma = max(self.min_sigma, self.sigma * self.decay_rate)
        return noise
    
    def reset(self):
        """Reset is not needed for Gaussian noise but included for compatibility."""
        pass


class AdaptiveNoise:
    """Adaptive noise generator that adjusts based on learning progress."""
    
    def __init__(self, size: int, initial_sigma: float = 0.2,
                 adaptation_rate: float = 0.01, window_size: int = 100,
                 seed: Optional[int] = None):
        """Initialize parameters and noise process.
        
        Args:
            size: Size of the action space
            initial_sigma: Initial standard deviation
            adaptation_rate: Rate at which sigma adapts
            window_size: Window size for reward averaging
            seed: Random seed
        """
        self.size = size
        self.sigma = initial_sigma
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        self.reward_history = []
        
        if seed is not None:
            np.random.seed(seed)
            
    def sample(self) -> np.ndarray:
        """Generate and return a noise sample.
        
        Returns:
            Noise sample
        """
        return np.random.normal(0, self.sigma, self.size)
    
    def update(self, reward: float):
        """Update noise parameters based on received reward.
        
        Args:
            reward: Latest reward received
        """
        self.reward_history.append(reward)
        if len(self.reward_history) > self.window_size:
            self.reward_history.pop(0)
            
            # Calculate improvement
            old_avg = np.mean(self.reward_history[:-self.window_size//2])
            new_avg = np.mean(self.reward_history[-self.window_size//2:])
            improvement = new_avg - old_avg
            
            # Adapt sigma based on improvement
            if improvement < 0:
                # Increase exploration if performance is degrading
                self.sigma *= (1 + self.adaptation_rate)
            else:
                # Decrease exploration if performance is improving
                self.sigma *= (1 - self.adaptation_rate)
    
    def reset(self):
        """Reset reward history."""
        self.reward_history = []