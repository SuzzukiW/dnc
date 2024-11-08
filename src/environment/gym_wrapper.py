# src/environment/gym_wrapper.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Union, Dict, Any, Tuple

from .single_agent_sumo_env import SingleAgentSumoEnvironment
from .multi_agent_sumo_env import MultiAgentSumoEnvironment

class SumoGymEnv(gym.Env):
    """
    Gymnasium wrapper for SUMO environment
    
    This wrapper makes our SUMO environments fully compliant with the Gymnasium interface,
    allowing use of standard RL libraries and tools.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        env_config: dict,
        multi_agent: bool = False,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize SUMO Gym Environment
        
        Args:
            env_config: Configuration dictionary for SUMO environment
            multi_agent: Whether to use multi-agent environment
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.multi_agent = multi_agent
        
        # Force GUI mode if human rendering is requested
        if render_mode == "human":
            env_config['use_gui'] = True
        
        # Create appropriate environment
        if multi_agent:
            self.env = MultiAgentSumoEnvironment(**env_config)
            # Multi-agent spaces are handled differently
            self.observation_space = spaces.Dict({
                agent_id: space for agent_id, space in 
                self.env.observation_spaces.items()
            })
            self.action_space = spaces.Dict({
                agent_id: space for agent_id, space in 
                self.env.action_spaces.items()
            })
        else:
            self.env = SingleAgentSumoEnvironment(**env_config)
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
        
        # Store environment configuration
        self.env_config = env_config
        
        # For metrics tracking
        self.current_episode = 0
        self.current_step = 0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Union[Tuple[np.ndarray, dict], Tuple[Dict[str, np.ndarray], dict]]:
        """
        Reset environment to initial state
        
        Args:
            seed: Random seed
            options: Additional options for reset
            
        Returns:
            Initial observation and info dictionary
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_episode += 1
        
        observation, info = self.env.reset(seed=seed)
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(
        self,
        action: Union[int, Dict[str, int]]
    ) -> Union[
        Tuple[np.ndarray, float, bool, bool, dict],
        Tuple[Dict[str, np.ndarray], Dict[str, float], bool, bool, dict]
    ]:
        """
        Take a step in the environment
        
        Args:
            action: Action to take (integer for single-agent, dict for multi-agent)
            
        Returns:
            observation: Next state observation
            reward: Reward received
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        self.current_step += 1
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        if self.render_mode == "human":
            self._render_frame()
        
        # Add episode info
        info.update({
            'episode': self.current_episode,
            'step': self.current_step,
        })
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment
        
        Note: For SUMO, rendering is handled through the GUI when enabled
        """
        if self.render_mode == "human":
            return self._render_frame()
        
        return None
    
    def _render_frame(self):
        """
        Render a frame
        
        For SUMO, this is handled automatically by the GUI when enabled
        """
        pass
    
    def close(self):
        """Close environment"""
        self.env.close()
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed"""
        super().reset(seed=seed)
    
    def get_metrics(self) -> dict:
        """Get current metrics"""
        return self.env.get_metrics()

class SingleAgentSumoGymEnv(SumoGymEnv):
    """Convenience class for single-agent SUMO environment"""
    def __init__(self, env_config: dict, render_mode: Optional[str] = None):
        super().__init__(env_config, multi_agent=False, render_mode=render_mode)

class MultiAgentSumoGymEnv(SumoGymEnv):
    """Convenience class for multi-agent SUMO environment"""
    def __init__(self, env_config: dict, render_mode: Optional[str] = None):
        super().__init__(env_config, multi_agent=True, render_mode=render_mode)