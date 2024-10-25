# src/environment/sumo_env.py

import os
import gymnasium as gym  # Changed from 'gym' to 'gymnasium'
import traci
import numpy as np
from typing import Tuple, Dict, List
from gymnasium import spaces  # Changed from 'gym.spaces' to 'gymnasium.spaces'
from src.utils.data_collector import SUMODataCollector

class SumoEnvironment(gym.Env):
    """
    SUMO Environment for traffic signal control
    """
    def __init__(self, 
                 config_file: str,
                 use_gui: bool = True,
                 num_seconds: int = 3600,
                 max_depart_delay: int = 100000,
                 time_to_teleport: int = -1,
                 delta_time: int = 5,
                 yellow_time: int = 2,
                 min_green: int = 5,
                 max_green: int = 50):
        super().__init__()  # Added super().__init__()
        
        self.config_file = config_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.max_depart_delay = max_depart_delay
        self.time_to_teleport = time_to_teleport
        
        # Initialize SUMO
        if self.use_gui:
            self._sumo_binary = 'sumo-gui'
        else:
            self._sumo_binary = 'sumo'
            
        # Initialize data collector
        self.data_collector = SUMODataCollector()
        
        # Get traffic light IDs and initialize their states
        self.traffic_lights = []
        self.phases = {}
        self.observation_spaces = {}
        self.action_spaces = {}
        
        # Connect to SUMO and initialize spaces
        self._start_simulation()
        self._initialize_spaces()

        # Required for gymnasium
        self.observation_space = spaces.Dict({
            tl_id: space for tl_id, space in self.observation_spaces.items()
        })
        self.action_space = spaces.Dict({
            tl_id: space for tl_id, space in self.action_spaces.items()
        })
        
    def _start_simulation(self):
        """Start SUMO simulation"""
        sumo_cmd = [
            self._sumo_binary,
            '-c', self.config_file,
            '--no-warnings',
            '--no-step-log',
            '--random',
            '--time-to-teleport', str(self.time_to_teleport),
            '--max-depart-delay', str(self.max_depart_delay)
        ]
        
        traci.start(sumo_cmd)
        self.data_collector.setup_collection()
        
        # Get all traffic light IDs
        self.traffic_lights = traci.trafficlight.getIDList()
        
        # Get phases for each traffic light
        for tl_id in self.traffic_lights:
            self.phases[tl_id] = traci.trafficlight.getAllProgramLogics(tl_id)[0].phases
            
    def _initialize_spaces(self):
        """Initialize observation and action spaces for each traffic light"""
        for tl_id in self.traffic_lights:
            # Observation space: [queue_length, waiting_time, phase] for each lane
            num_lanes = len(traci.trafficlight.getControlledLanes(tl_id))
            
            self.observation_spaces[tl_id] = spaces.Box(
                low=np.zeros(num_lanes * 3 + 1),  # +1 for current phase
                high=np.ones(num_lanes * 3 + 1),
                dtype=np.float32
            )
            
            # Action space: duration for each phase
            num_phases = len(self.phases[tl_id])
            self.action_spaces[tl_id] = spaces.Discrete(num_phases)
            
    def _get_state(self, tl_id: str) -> np.ndarray:
        """Get the current state for a traffic light"""
        state = self.data_collector.get_state_observation(tl_id)
        return state
        
    def _apply_action(self, tl_id: str, action: int):
        """Apply the selected action to a traffic light"""
        # Set the new phase
        traci.trafficlight.setPhase(tl_id, action)
        
        # Wait for yellow phase if changing
        if action != traci.trafficlight.getPhase(tl_id):
            traci.trafficlight.setPhase(tl_id, 'y')  # Yellow phase
            for _ in range(self.yellow_time):
                traci.simulationStep()
                
        # Set the new phase duration
        traci.trafficlight.setPhaseDuration(tl_id, self.min_green)
        
    def _get_reward(self, tl_id: str) -> float:
        """Calculate reward for a traffic light"""
        # Get metrics from data collector
        metrics = self.data_collector.collected_data['traffic_metrics'][-1]
        
        # Calculate reward based on waiting time and queue length
        reward = -(metrics['total_waiting'] + metrics['vehicle_count'])
        
        return reward
        
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Execute one step in the environment"""
        # Apply actions
        for tl_id, action in actions.items():
            self._apply_action(tl_id, action)
            
        # Run simulation for delta_time steps
        for _ in range(self.delta_time):
            traci.simulationStep()
            self.data_collector.step_collection()
            
        # Get new states and rewards
        states = {tl_id: self._get_state(tl_id) for tl_id in self.traffic_lights}
        rewards = {tl_id: self._get_reward(tl_id) for tl_id in self.traffic_lights}
        
        # Check if simulation is done
        done = traci.simulation.getTime() > self.num_seconds
        dones = {tl_id: done for tl_id in self.traffic_lights}
        
        # Additional info
        infos = {tl_id: {} for tl_id in self.traffic_lights}
        
        return states, rewards, dones, infos
        
    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:  # Updated reset method signature
        """Reset the environment"""
        # Close existing connection and restart SUMO
        try:
            traci.close()
        except:
            pass
            
        self._start_simulation()
        
        # Get initial states
        states = {tl_id: self._get_state(tl_id) for tl_id in self.traffic_lights}
        
        # Return states and empty info dict (required by gymnasium)
        return states, {}
        
    def close(self):
        """Close the environment"""
        try:
            traci.close()
        except:
            pass

    def render(self, mode="human"):
        """
        Render the environment
        The SUMO-GUI already handles visualization
        """
        pass