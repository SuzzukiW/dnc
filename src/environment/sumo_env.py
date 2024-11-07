# src/environment/sumo_env.py

import os
import sys
import traci
import numpy as np
from gymnasium import Env, spaces
import sumolib

class SUMOEnvironment(Env):
    def __init__(self, 
                 net_file,
                 route_file,
                 out_csv_name,
                 use_gui=False,
                 num_seconds=20000,
                 max_depart_delay=100000,
                 time_to_teleport=-1,
                 delta_time=5,
                 yellow_time=2,
                 min_green=5,
                 max_green=50):
        
        self.net_file = net_file
        self.route_file = route_file
        self.out_csv_name = out_csv_name
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.max_depart_delay = max_depart_delay
        self.time_to_teleport = time_to_teleport
        
        # SUMO initialization
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            raise ValueError("Please declare SUMO_HOME environment variable")
            
        self.sumo_binary = 'sumo-gui' if self.use_gui else 'sumo'
        self.sumo_cmd = [
            self.sumo_binary,
            '-n', self.net_file,
            '-r', self.route_file,
            '--max-depart-delay', str(self.max_depart_delay),
            '--waiting-time-memory', '10000',
            '--time-to-teleport', str(self.time_to_teleport)
        ]
        
        # Initialize connection with sumo
        traci.start(self.sumo_cmd)
        
        # Get traffic light IDs
        self.traffic_lights = traci.trafficlight.getIDList()
        
        # Initialize observation and action spaces
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(12,),  # Example: [queue_length, waiting_time, density] for 4 approaches
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(4)  # Example: 4 different signal phases
        
        self.current_phase = {tl: 0 for tl in self.traffic_lights}
        self.yellow_phase_dict = self._init_yellow_phases()
        
    def _init_yellow_phases(self):
        """Initialize yellow phases for each traffic light"""
        yellow_dict = {}
        for tl in self.traffic_lights:
            phases = traci.trafficlight.getAllProgramLogics(tl)[0].phases
            yellow_dict[tl] = [i for i, p in enumerate(phases) if 'y' in p.state.lower()]
        return yellow_dict
    
    def _get_state(self, traffic_light_id):
        """Get state for a specific traffic light"""
        incoming_lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
        
        state = []
        for lane in incoming_lanes:
            # Get queue length
            queue_length = traci.lane.getLastStepHaltingNumber(lane)
            # Get waiting time
            waiting_time = traci.lane.getWaitingTime(lane)
            # Get density
            density = traci.lane.getLastStepVehicleNumber(lane) / traci.lane.getLength(lane)
            
            # Normalize values
            state.extend([
                min(1.0, queue_length / 10.0),
                min(1.0, waiting_time / 100.0),
                min(1.0, density)
            ])
            
        return np.array(state, dtype=np.float32)
    
    def _apply_action(self, traffic_light_id, action):
        """Apply action to a specific traffic light"""
        # First, set yellow phase if needed
        if self.current_phase[traffic_light_id] != action:
            yellow_phase = self.yellow_phase_dict[traffic_light_id][0]  # Get first yellow phase
            traci.trafficlight.setPhase(traffic_light_id, yellow_phase)
            traci.simulationStep()
            
        # Then set the new phase
        traci.trafficlight.setPhase(traffic_light_id, action)
        self.current_phase[traffic_light_id] = action
    
    def _get_reward(self, traffic_light_id):
        """Calculate reward for a specific traffic light"""
        incoming_lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
        
        total_waiting_time = 0
        total_queue_length = 0
        
        for lane in incoming_lanes:
            total_waiting_time += traci.lane.getWaitingTime(lane)
            total_queue_length += traci.lane.getLastStepHaltingNumber(lane)
            
        # Negative reward based on waiting time and queue length
        reward = -(total_waiting_time + total_queue_length)
        
        return reward
    
    def reset(self, seed=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        if traci.isLoaded():
            traci.close()
        traci.start(self.sumo_cmd)
        
        self.current_phase = {tl: 0 for tl in self.traffic_lights}
        
        # Get initial state
        initial_state = self._get_state(self.traffic_lights[0])  # For single agent
        
        return initial_state, {}
    
    def step(self, action):
        """Execute action and return new state, reward, done, truncated, and info"""
        # Apply action for single traffic light (first one in list)
        traffic_light_id = self.traffic_lights[0]
        self._apply_action(traffic_light_id, action)
        
        # Run simulation for delta_time steps
        for _ in range(self.delta_time):
            traci.simulationStep()
        
        # Get new state
        new_state = self._get_state(traffic_light_id)
        
        # Calculate reward
        reward = self._get_reward(traffic_light_id)
        
        # Check if simulation is done
        done = traci.simulation.getTime() >= self.num_seconds
        
        # Additional info
        info = {
            'time': traci.simulation.getTime(),
            'total_waiting_time': -reward  # Negative of reward since reward is negative of waiting time
        }
        
        return new_state, reward, done, False, info
    
    def close(self):
        """Close the environment"""
        if traci.isLoaded():
            traci.close()