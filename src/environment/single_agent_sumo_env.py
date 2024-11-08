# src/environment/single_agent_sumo_env.py

import os
import sys
import traci
import numpy as np
from gymnasium import Env, spaces
import sumolib

class SingleAgentSumoEnvironment(Env):
    """
    Single-agent environment for traffic signal control
    """
    def _init_valid_phases(self, tl_id):
        """Initialize valid phases for a traffic light"""
        try:
            program = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            # Find phases with at least one green light
            valid_phases = [
                i for i, phase in enumerate(program.phases)
                if any(c in 'Gg' for c in phase.state)
            ]
            if not valid_phases and program.phases:
                valid_phases = [0]
                print(f"Warning: No valid phases found for {tl_id}, using first phase")
            return valid_phases
        except Exception as e:
            print(f"Error getting phases for {tl_id}: {e}")
            return [0]
        
    def __init__(self, 
                net_file,
                route_file,
                out_csv_name,
                use_gui=False,
                num_seconds=1800,      # 30 minutes default
                max_depart_delay=100,  # Shorter delay
                time_to_teleport=100,  # Quicker teleport
                delta_time=5,          # More frequent decisions
                yellow_time=2,         # Shorter yellow time
                min_green=8,           # More flexible timing
                max_green=30,          # Shorter maximum green
                tl_id=None):          # Traffic light ID
        
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
        self.selected_tl = tl_id
        
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
        
        # Validate and set traffic light
        if not self.selected_tl:
            # If no traffic light specified, select the first one
            all_traffic_lights = traci.trafficlight.getIDList()
            if not all_traffic_lights:
                raise ValueError("No traffic lights found in the network")
            self.selected_tl = all_traffic_lights[0]
        elif self.selected_tl not in traci.trafficlight.getIDList():
            raise ValueError(f"Traffic light {self.selected_tl} not found in network")
        
        # Initialize valid phases
        self.valid_phases = self._init_valid_phases(self.selected_tl)
        
        # Get dimensions
        self.num_green_phases = len(self.valid_phases)
        self.num_lanes = len(traci.trafficlight.getControlledLanes(self.selected_tl))
        
        # Initialize spaces
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_lanes * 3,),  # queue, waiting time, density for each lane
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_green_phases)
        
        # Initialize phase
        self.current_phase = self.valid_phases[0]
        self.yellow_phase = self._get_yellow_phase()
        
        # Performance tracking
        self.metrics = {
            'waiting_times': [],
            'queue_lengths': [],
            'throughput': [],
            'traffic_pressure': []
        }
        
    def _get_yellow_phase(self):
        """Get yellow phase for the traffic light"""
        phases = traci.trafficlight.getAllProgramLogics(self.selected_tl)[0].phases
        yellow_phases = [i for i, p in enumerate(phases) if 'y' in p.state.lower()]
        return yellow_phases[0] if yellow_phases else 0
    
    def _get_state(self):
        """Get current state"""
        controlled_lanes = traci.trafficlight.getControlledLanes(self.selected_tl)
        
        state = []
        for lane in controlled_lanes:
            # Queue length (number of stopped vehicles)
            queue = traci.lane.getLastStepHaltingNumber(lane)
            # Waiting time
            wait_time = traci.lane.getWaitingTime(lane)
            # Density
            density = traci.lane.getLastStepVehicleNumber(lane) / traci.lane.getLength(lane)
            
            # Normalize values
            state.extend([
                min(1.0, queue / 10.0),
                min(1.0, wait_time / 100.0),
                min(1.0, density)
            ])
            
        return np.array(state, dtype=np.float32)
    
    def _apply_action(self, action):
        """Apply action to traffic light"""
        # Convert action index to phase index
        phase_index = self.valid_phases[action]
        
        if self.current_phase != phase_index:
            # Set yellow phase
            traci.trafficlight.setPhase(self.selected_tl, self.yellow_phase)
            for _ in range(self.yellow_time):
                traci.simulationStep()
            
            # Set new phase
            traci.trafficlight.setPhase(self.selected_tl, phase_index)
            self.current_phase = phase_index
    
    def _get_reward(self):
        """Calculate reward"""
        controlled_lanes = traci.trafficlight.getControlledLanes(self.selected_tl)
        
        total_waiting_time = 0
        total_queue_length = 0
        total_vehicles = 0
        total_speed = 0
        
        for lane in controlled_lanes:
            # Cap waiting time at 3 minutes
            waiting_time = min(traci.lane.getWaitingTime(lane), 180.0)
            # Cap queue length
            queue_length = min(traci.lane.getLastStepHaltingNumber(lane), 10)
            # Get vehicle count and speed
            vehicles = traci.lane.getLastStepVehicleNumber(lane)
            speed = traci.lane.getLastStepMeanSpeed(lane)
            max_speed = traci.lane.getMaxSpeed(lane)
            
            total_waiting_time += waiting_time
            total_queue_length += queue_length
            total_vehicles += vehicles
            total_speed += speed / max_speed if max_speed > 0 else 0
        
        # Normalize metrics
        num_lanes = len(controlled_lanes)
        avg_waiting = total_waiting_time / (num_lanes * 180.0)
        avg_queue = total_queue_length / (num_lanes * 10.0)
        avg_speed = total_speed / num_lanes
        throughput = total_vehicles / (num_lanes * 10.0)
        
        # Combined reward
        reward = (
            -0.4 * avg_waiting    # Penalize waiting time
            -0.3 * avg_queue      # Penalize queue length
            +0.2 * avg_speed      # Reward higher speeds
            +0.1 * throughput     # Reward throughput
        )
        
        return reward * 100  # Scale reward
    
    def _get_info(self):
        """Get additional information"""
        controlled_lanes = traci.trafficlight.getControlledLanes(self.selected_tl)
        
        # Calculate metrics
        queue_length = sum(traci.lane.getLastStepHaltingNumber(lane) 
                         for lane in controlled_lanes)
        waiting_time = sum(traci.lane.getWaitingTime(lane) 
                         for lane in controlled_lanes)
        throughput = sum(traci.lane.getLastStepVehicleNumber(lane)
                        for lane in controlled_lanes)
        
        # Calculate traffic pressure
        traffic_pressure = sum(
            traci.lane.getLastStepVehicleNumber(lane) / 
            max(traci.lane.getLength(lane), 1)
            for lane in controlled_lanes
        ) / len(controlled_lanes)
        
        # Update metrics history
        self.metrics['waiting_times'].append(waiting_time)
        self.metrics['queue_lengths'].append(queue_length)
        self.metrics['throughput'].append(throughput)
        self.metrics['traffic_pressure'].append(traffic_pressure)
        
        return {
            'time': traci.simulation.getTime(),
            'waiting_time': waiting_time,
            'queue_length': queue_length,
            'throughput': throughput,
            'traffic_pressure': traffic_pressure,
            'current_phase': self.current_phase,
            'num_vehicles': sum(traci.lane.getLastStepVehicleNumber(lane) 
                              for lane in controlled_lanes)
        }
    
    def reset(self, seed=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        if traci.isLoaded():
            traci.close()
        traci.start(self.sumo_cmd)
        
        # Reset phase
        self.current_phase = self.valid_phases[0]
        
        # Reset metrics
        self.metrics = {
            'waiting_times': [],
            'queue_lengths': [],
            'throughput': [],
            'traffic_pressure': []
        }
        
        return self._get_state(), {}
    
    def step(self, action):
        """Execute action and return results"""
        # Ensure action is valid
        action = min(action, len(self.valid_phases) - 1)
        
        # Apply action
        self._apply_action(action)
        
        # Run simulation
        for _ in range(self.delta_time):
            traci.simulationStep()
        
        # Get new state and reward
        new_state = self._get_state()
        reward = self._get_reward()
        
        # Check if episode is done
        done = traci.simulation.getTime() >= self.num_seconds
        
        # Get additional info
        info = self._get_info()
        
        return new_state, reward, done, False, info
    
    def close(self):
        """Close environment"""
        if traci.isLoaded():
            traci.close()
    
    def get_metrics(self):
        """Return current metrics"""
        return self.metrics