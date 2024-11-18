import numpy as np
import gymnasium as gym
from typing import List, Tuple, Dict, Any

class MultiAgentEnv:
    """Multi-Agent Environment for MADDPG Training"""
    
    def __init__(self, env_name: str, num_agents: int):
        """Initialize the multi-agent environment
        
        Params
        ======
            env_name (str): Name of the environment from Gymnasium
            num_agents (int): Number of agents in the environment
        """
        self.env = gym.make(env_name)
        self.num_agents = num_agents
        
        # Validate environment compatibility
        self._validate_env()
    
    def _validate_env(self):
        """Validate that the environment supports multi-agent setup"""
        assert isinstance(self.env.observation_space, gym.spaces.Tuple), \
            "Environment must support multi-agent observations"
        assert isinstance(self.env.action_space, gym.spaces.Tuple), \
            "Environment must support multi-agent actions"
        
        assert len(self.env.observation_space) == self.num_agents, \
            f"Number of observations ({len(self.env.observation_space)}) must match num_agents ({self.num_agents})"
        assert len(self.env.action_space) == self.num_agents, \
            f"Number of actions ({len(self.env.action_space)}) must match num_agents ({self.num_agents})"
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return initial states
        
        Returns
        =======
            states (np.ndarray): Initial states for all agents
            info (dict): Additional environment information
        """
        states, info = self.env.reset()
        return np.array(states), info
    
    def step(self, actions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """Take a step in the environment with actions from all agents
        
        Params
        ======
            actions (List[np.ndarray]): Actions for each agent
        
        Returns
        =======
            next_states (np.ndarray): Next states for all agents
            rewards (np.ndarray): Rewards for each agent
            dones (np.ndarray): Episode termination flags
            truncated (bool): Whether episode was truncated
            info (dict): Additional environment information
        """
        assert len(actions) == self.num_agents, \
            f"Number of actions ({len(actions)}) must match num_agents ({self.num_agents})"
        
        next_states, rewards, dones, truncated, info = self.env.step(actions)
        
        return (np.array(next_states), 
                np.array(rewards), 
                np.array(dones), 
                truncated, 
                info)
    
    def render(self):
        """Render the environment"""
        return self.env.render()
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    @property
    def observation_space(self):
        """Get observation space for the environment"""
        return self.env.observation_space
    
    @property
    def action_space(self):
        """Get action space for the environment"""
        return self.env.action_space
