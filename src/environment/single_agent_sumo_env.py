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
                num_seconds=3600,     # 1 hour default
                delta_time=1,         # 1 second default
                yellow_time=2,        # Default yellow time
                min_green=10,         # Minimum green time
                max_green=60,         # Maximum green time
                sumo_warnings=True):  # Whether to show SUMO warnings
        self.net_file = net_file
        self.route_file = route_file
        self.out_csv_name = out_csv_name
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.sumo_warnings = sumo_warnings
        
        # Initialize metrics storage
        self.metrics = {
            'waiting_times': [],
            'queue_lengths': [],
            'throughput': [],
            'rewards': []
        }
        
        # Vehicle tracking
        self.vehicles_seen = set()
        self.vehicles_completed = set()
        
        # Set up SUMO
        if 'SUMO_HOME' not in os.environ:
            os.environ['SUMO_HOME'] = '/opt/homebrew/opt/sumo/share/sumo'
        
        # Load network
        self.net = sumolib.net.readNet(self.net_file)
        
        # Set up SUMO command with optimized settings
        sumo_binary = 'sumo-gui' if self.use_gui else 'sumo'
        self.sumo_cmd = [
            sumo_binary,
            '-n', self.net_file,
            '-r', self.route_file,
            '--waiting-time-memory', '10000',
            '--time-to-teleport', '-1',
            '--no-step-log', 'true',
            '--random', 'false',  # Disable randomness for reproducibility
            '--no-warnings', 'true',
            '--duration-log.disable', 'true',
            '--tripinfo-output.write-unfinished',
            '--device.rerouting.probability', '0.8',  # Enable dynamic rerouting
            '--device.rerouting.period', '60',  # Rerouting check period
            '--lanechange.duration', '2',  # Faster lane changes
            '--collision.action', 'teleport',  # Handle collisions with teleport
            '--collision.mingap-factor', '0',  # Reduce minimum gap
            '--time-to-impatience', '30',  # Reduce time to impatience
        ]
        
        # Start SUMO
        try:
            traci.start(self.sumo_cmd)
        except traci.exceptions.TraCIException as e:
            print(f"Error starting SUMO: {e}")
            if traci.isLoaded():
                traci.close()
            traci.start(self.sumo_cmd)
        
        # Get traffic light ID (using the first one in the network)
        self.tl_id = list(traci.trafficlight.getIDList())[0]
        
        # Get valid phases
        self.valid_phases = self._init_valid_phases(self.tl_id)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.valid_phases))
        
        # Observation space: [queue_length, waiting_time] for each incoming lane
        num_lanes = len(traci.trafficlight.getControlledLanes(self.tl_id))
        self.observation_space = spaces.Box(
            low=0,
            high=float('inf'),
            shape=(num_lanes * 2,),
            dtype=np.float32
        )
        
        # Initialize phase
        self.current_phase = self.valid_phases[0]
        self.yellow_phase = self._get_yellow_phase()
    
    def _get_yellow_phase(self):
        """Get yellow phase for the traffic light"""
        phases = traci.trafficlight.getAllProgramLogics(self.tl_id)[0].phases
        yellow_phases = [i for i, p in enumerate(phases) if 'y' in p.state.lower()]
        return yellow_phases[0] if yellow_phases else 0
    
    def _get_state(self):
        """Get current state"""
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        
        state = []
        for lane in controlled_lanes:
            # Queue length (number of stopped vehicles)
            queue = traci.lane.getLastStepHaltingNumber(lane)
            # Waiting time
            wait_time = traci.lane.getWaitingTime(lane)
            
            state.extend([
                queue,
                wait_time
            ])
            
        return np.array(state, dtype=np.float32)
    
    def _apply_action(self, action):
        """Apply action to traffic light"""
        # Convert action index to phase index
        phase_index = self.valid_phases[action]
        
        if self.current_phase != phase_index:
            # Set yellow phase
            traci.trafficlight.setPhase(self.tl_id, self.yellow_phase)
            for _ in range(self.yellow_time):
                traci.simulationStep()
            
            # Set new phase
            traci.trafficlight.setPhase(self.tl_id, phase_index)
            self.current_phase = phase_index
    
    def _get_reward(self):
        """Calculate reward based on global traffic metrics"""
        # Get all vehicle IDs in the network
        vehicle_ids = traci.vehicle.getIDList()
        
        # Calculate global metrics
        total_waiting_time = 0
        total_queue_length = 0
        total_speed = 0
        num_vehicles = len(vehicle_ids)
        
        if num_vehicles > 0:
            # Calculate metrics for all vehicles in the network
            for vid in vehicle_ids:
                # Waiting time (time spent with speed below 0.1 m/s)
                waiting_time = traci.vehicle.getWaitingTime(vid)
                # Time loss (time lost due to driving below ideal speed)
                time_loss = traci.vehicle.getTimeLoss(vid)
                # Current speed
                speed = traci.vehicle.getSpeed(vid)
                # Route ID and edges
                route_id = traci.vehicle.getRouteID(vid)
                
                total_waiting_time += waiting_time + time_loss
                if speed < 0.1:  # Vehicle is considered stopped
                    total_queue_length += 1
                total_speed += speed
            
            # Calculate averages
            avg_waiting_time = total_waiting_time / num_vehicles
            avg_queue_length = total_queue_length / num_vehicles
            avg_speed = total_speed / num_vehicles
            
            # Get throughput (completed trips)
            throughput = traci.simulation.getArrivedNumber()
            
            # Update metrics for logging
            self.metrics['waiting_times'].append(avg_waiting_time)
            self.metrics['queue_lengths'].append(avg_queue_length)
            self.metrics['throughput'].append(throughput)
            
            # Calculate normalized reward components
            waiting_penalty = min(1.0, avg_waiting_time / 120.0)  # Cap at 2 minutes
            queue_penalty = min(1.0, avg_queue_length / 15.0)    # Cap at 15 vehicles
            speed_reward = avg_speed / traci.vehicle.getMaxSpeed(vehicle_ids[0])
            throughput_reward = min(1.0, throughput / 5.0)      # Cap at 5 vehicles (more achievable)
            
            # Penalize having too few vehicles (encourage processing more traffic)
            low_traffic_penalty = max(0, 0.5 - (num_vehicles / 200.0))  # Penalty if < 200 vehicles
            
            # Combined reward with adjusted weights
            reward = (
                -0.4 * waiting_penalty     # Penalty for waiting time
                -0.2 * queue_penalty       # Penalty for queue length
                +0.1 * speed_reward        # Reward for speed
                +0.3 * throughput_reward   # Increased reward for throughput
                -0.2 * low_traffic_penalty # Penalty for too few vehicles
            )
            
            # Add bonus rewards
            if waiting_penalty < 0.3:  # If waiting time is very low
                reward += 0.3
            if throughput > 0:  # Bonus for any completed vehicles
                reward += 0.5 * throughput  # Increased throughput bonus
            
            # Penalty for zero throughput
            if throughput == 0 and num_vehicles > 100:  # Only penalize if there are enough vehicles
                reward -= 0.5
            
            return reward * 100  # Scale reward
        
        return 0  # Return 0 if no vehicles in network
    
    def _get_info(self):
        """Get additional information about the environment state"""
        vehicle_ids = traci.vehicle.getIDList()
        num_vehicles = len(vehicle_ids)
        
        if num_vehicles > 0:
            # Calculate global metrics
            total_waiting_time = sum(traci.vehicle.getWaitingTime(vid) + 
                                   traci.vehicle.getTimeLoss(vid) 
                                   for vid in vehicle_ids)
            total_queue_length = sum(1 for vid in vehicle_ids 
                                   if traci.vehicle.getSpeed(vid) < 0.1)
            total_speed = sum(traci.vehicle.getSpeed(vid) for vid in vehicle_ids)
            
            # Calculate averages
            avg_waiting_time = total_waiting_time / num_vehicles
            avg_queue_length = total_queue_length
            avg_speed = total_speed / num_vehicles
            
            info = {
                'average_waiting_time': avg_waiting_time,
                'queue_length': avg_queue_length,
                'average_speed': avg_speed,
                'throughput': traci.simulation.getArrivedNumber(),
                'total_vehicles': num_vehicles
            }
        else:
            info = {
                'average_waiting_time': 0,
                'queue_length': 0,
                'average_speed': 0,
                'throughput': 0,
                'total_vehicles': 0
            }
        
        return info
    
    def step(self, action):
        """Execute action and return results"""
        # Ensure action is valid
        action = min(action, len(self.valid_phases) - 1)
        
        # Apply action
        self._apply_action(action)
        
        # Track vehicles before step
        current_vehicles = set(traci.vehicle.getIDList())
        self.vehicles_seen.update(current_vehicles)
        
        # Run simulation
        try:
            for _ in range(self.delta_time):
                traci.simulationStep()
        except traci.exceptions.TraCIException as e:
            print(f"Error during simulation step: {e}")
            return self._get_state(), -100, True, False, {}
        
        # Track completed vehicles
        new_vehicles = set(traci.vehicle.getIDList())
        completed_vehicles = self.vehicles_seen - new_vehicles - self.vehicles_completed
        self.vehicles_completed.update(completed_vehicles)
        
        # Get new state and reward
        new_state = self._get_state()
        reward = self._get_reward()
        
        # Check if episode is done
        done = traci.simulation.getTime() >= self.num_seconds
        
        # Get additional info
        info = self._get_info()
        info.update({
            'vehicles_completed': len(completed_vehicles),
            'total_vehicles_completed': len(self.vehicles_completed),
            'vehicles_in_network': len(new_vehicles)
        })
        
        return new_state, reward, done, False, info
    
    def reset(self, seed=None):
        """Reset environment"""
        # Close existing SUMO instance if any
        if traci.isLoaded():
            traci.close()
        
        # Reset vehicle tracking
        self.vehicles_seen = set()
        self.vehicles_completed = set()
        
        # Start new SUMO instance
        try:
            traci.start(self.sumo_cmd)
        except traci.exceptions.TraCIException as e:
            print(f"Error starting SUMO: {e}")
            if traci.isLoaded():
                traci.close()
            traci.start(self.sumo_cmd)
        
        # Reset metrics
        self.metrics = {
            'waiting_times': [],
            'queue_lengths': [],
            'throughput': [],
            'rewards': []
        }
        
        # Reset phase
        self.current_phase = self.valid_phases[0]
        traci.trafficlight.setPhase(self.tl_id, self.current_phase)
        
        # Get initial state
        state = self._get_state()
        
        return state, {}
    
    def close(self):
        """Close environment"""
        if traci.isLoaded():
            traci.close()
    
    def get_metrics(self):
        """Return current metrics"""
        return self.metrics