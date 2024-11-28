# src/environment/multi_agent_sumo_env_shared_experience.py

import os
import sys
import numpy as np
from pathlib import Path
import traci
import sumolib
from collections import defaultdict

class MultiAgentSumoEnvSharedExperience:
    """Multi-agent environment for SUMO traffic light control with shared experience."""
    
    def __init__(
        self,
        net_file,
        route_file,
        scenario,
        use_gui=False,
        num_seconds=3600,
        delta_time=5,
        yellow_time=2,
        min_green=5,
        max_green=50,
        sumo_seed=42,
        sumo_warnings=False
    ):
        self.net_file = net_file
        self.route_file = route_file
        self.scenario = scenario
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.sumo_seed = sumo_seed
        
        # Initialize SUMO
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            raise ValueError("SUMO_HOME environment variable not set")
            
        # Build command
        self.sumo_cmd = self._build_sumo_cmd(sumo_warnings)
        
        # Traffic light settings
        self.traffic_lights = {}
        # State size: 4 lanes * (queue + wait + speed) + phase_duration + is_yellow
        self.observation_space_size = 4 * 3 + 2  # 14 values total
        self.action_space_size = 2  # Binary: keep/change phase
        
        # Metrics
        self.metrics = defaultdict(dict)
        self.waiting_times = defaultdict(dict)
        self.queue_lengths = defaultdict(int)
        self.throughput = defaultdict(int)
        
        # Episode statistics
        self.episode_step = 0
        self.episode_length = num_seconds // delta_time
        
        # Start SUMO
        self.start_simulation()

    def _build_sumo_cmd(self, sumo_warnings):
        """Build the SUMO command with all necessary options."""
        cmd = []
        
        # Binary selection
        if self.use_gui:
            cmd.extend(['sumo-gui'])
        else:
            cmd.extend(['sumo'])

        # Basic options
        cmd.extend([
            '-n', self.net_file,
            '-r', self.route_file,
            '--max-depart-delay', '0',
            '--waiting-time-memory', '10000',
            '--time-to-teleport', '-1',
        ])

        # Optional flags
        if not sumo_warnings:
            cmd.extend(['--no-warnings'])
        
        if self.sumo_seed is not None:
            cmd.extend(['--seed', str(self.sumo_seed)])
            
        # Step length and logging options
        cmd.extend([
            '--step-length', str(self.delta_time),
            '--no-step-log',
            '--quit-on-end'
        ])
        
        return cmd

    def start_simulation(self):
        """Start or restart the SUMO simulation"""
        traci.start(self.sumo_cmd)
        self._get_traffic_lights()
        
    def _filter_valid_traffic_lights(self, traffic_lights):
        """Filter traffic lights to only include those with valid phase programs"""
        valid_traffic_lights = []
        
        for tl_id in traffic_lights:
            try:
                # Check if traffic light has valid program logic
                logic = traci.trafficlight.getAllProgramLogics(tl_id)
                if not logic:
                    continue
                
                # Check first program logic
                program = logic[0]
                
                # Check phases
                phases = program.phases
                if not phases:
                    continue
                
                # Check controlled lanes
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                if not controlled_lanes:
                    continue
                
                # Check controlled links
                controlled_links = traci.trafficlight.getControlledLinks(tl_id)
                if not controlled_links:
                    continue
                
                # If all checks pass, add to valid list
                valid_traffic_lights.append(tl_id)
                
            except Exception as e:
                print(f"Error checking traffic light {tl_id}: {str(e)}")
                continue
        
        return valid_traffic_lights

    def _get_traffic_lights(self):
        """Get all traffic lights and their properties"""
        print("\nInitializing traffic lights...")
        self.traffic_lights = {}
        
        # Get all traffic light IDs
        all_tl_ids = traci.trafficlight.getIDList()
        print(f"Found {len(all_tl_ids)} total traffic lights")
        
        # Filter valid traffic lights
        valid_tl_ids = []
        for tl_id in all_tl_ids:
            try:
                # Check if traffic light has valid program logic
                logic = traci.trafficlight.getAllProgramLogics(tl_id)
                if not logic:
                    continue
                
                # Check phases
                phases = logic[0].phases
                if not phases:
                    continue
                
                # Check controlled lanes
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                if not controlled_lanes:
                    continue
                
                # Check controlled links
                controlled_links = traci.trafficlight.getControlledLinks(tl_id)
                if not controlled_links:
                    continue
                
                valid_tl_ids.append(tl_id)
                
            except Exception as e:
                continue
        
        # Initialize valid traffic lights
        for tl_id in valid_tl_ids:
            try:
                # Get controlled lanes
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                unique_lanes = list(set(controlled_lanes))
                
                # Get program logic
                program = traci.trafficlight.getAllProgramLogics(tl_id)[0]
                phases = program.phases
                
                # Get controlled links
                links = traci.trafficlight.getControlledLinks(tl_id)
                
                # Store traffic light information
                self.traffic_lights[tl_id] = {
                    'phases': phases,
                    'links': links,
                    'lanes': unique_lanes,
                    'neighbors': {}  # Empty neighbors dict since we're not using them
                }
                
            except Exception as e:
                print(f"Error setting up traffic light {tl_id}: {str(e)}")
                continue
        
        print(f"Successfully initialized {len(self.traffic_lights)} traffic lights")

    def _get_state(self, tl_id):
        """Get state for a specific traffic light."""
        lanes = self.traffic_lights[tl_id]['lanes']
        
        # Get queue lengths and waiting times
        queue_lengths = []
        waiting_times = []
        speeds = []
        
        for lane in lanes:
            # Queue length (number of stopped vehicles)
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            queue = sum(1 for veh_id in vehicle_ids if traci.vehicle.getSpeed(veh_id) < 0.1)
            queue_lengths.append(queue)
            
            # Waiting time
            wait_time = traci.lane.getWaitingTime(lane)
            waiting_times.append(wait_time)
            
            # Average speed
            speed = traci.lane.getLastStepMeanSpeed(lane)
            speeds.append(speed)

        # Normalize to fixed size
        max_lanes = 4
        queue_lengths = queue_lengths[:max_lanes] + [0] * (max_lanes - len(queue_lengths))
        waiting_times = waiting_times[:max_lanes] + [0] * (max_lanes - len(waiting_times))
        speeds = speeds[:max_lanes] + [0] * (max_lanes - len(speeds))
        
        # Current phase info
        phase_id = traci.trafficlight.getPhase(tl_id)
        current_program = traci.trafficlight.getAllProgramLogics(tl_id)[0]
        phase_duration = traci.trafficlight.getPhaseDuration(tl_id)
        is_yellow = 'y' in current_program.phases[phase_id].state.lower()
        
        # Normalize and create state vector
        state = np.array(
            queue_lengths +  # 4 values
            waiting_times +  # 4 values
            speeds +        # 4 values
            [phase_duration / self.max_green,  # 1 value
             int(is_yellow)]                   # 1 value
        )
        
        return state
            
    def _get_reward(self, tl_id):
        """Calculate reward based on average waiting time and throughput."""
        lanes = self.traffic_lights[tl_id]['lanes']
        
        # Calculate average waiting time
        total_waiting_time = 0
        num_vehicles = 0
        
        for lane in lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for vehicle in vehicles:
                # Get waiting time (time spent with speed < 0.1 m/s)
                waiting_time = traci.vehicle.getWaitingTime(vehicle)
                total_waiting_time += waiting_time
                num_vehicles += 1
        
        # Calculate average waiting time and normalize
        avg_waiting_time = (total_waiting_time / max(1, num_vehicles))
        waiting_penalty = -min(1.0, avg_waiting_time / 180.0)  # Cap at 3 minutes
        
        # Calculate throughput
        outgoing_vehicles = 0
        for lane in self._get_outgoing_lanes(tl_id):
            # Only count vehicles moving at reasonable speed (> 1 m/s)
            outgoing_vehicles += sum(1 for veh in traci.lane.getLastStepVehicleIDs(lane)
                                   if traci.vehicle.getSpeed(veh) > 1.0)
        
        self.throughput[tl_id] += outgoing_vehicles
        normalized_throughput = min(1.0, outgoing_vehicles / (20 * len(lanes)))
        
        # Combine rewards with equal weights
        reward = 0.5 * waiting_penalty + 0.5 * normalized_throughput
        
        return reward
        
    def _get_outgoing_lanes(self, tl_id):
        """Get outgoing lanes for a traffic light."""
        outgoing_lanes = set()
        try:
            for connection in self.traffic_lights[tl_id]['links']:
                # Check if connection is valid and has enough elements
                if connection and len(connection) > 1:
                    outgoing_lanes.add(connection[1])
        except Exception as e:
            print(f"Warning: Error getting outgoing lanes for {tl_id}: {e}")
        return list(outgoing_lanes) if outgoing_lanes else []
        
    def step(self, actions):
        """Execute actions for all traffic lights and return results."""
        try:
            # Execute actions
            for tl_id, action in actions.items():
                try:
                    current_phase = traci.trafficlight.getPhase(tl_id)
                    if action == 1:  # Change phase
                        next_phase = (current_phase + 2) % len(self.traffic_lights[tl_id]['phases'])
                        traci.trafficlight.setPhase(tl_id, next_phase)
                except Exception as e:
                    print(f"Warning: Error executing action for {tl_id}: {e}")
                    continue
            
            # Run simulation step
            traci.simulationStep()
            self.episode_step += 1
            
            # Collect states and rewards
            states = {}
            rewards = {}
            
            for tl_id in self.traffic_lights:
                try:
                    states[tl_id] = self._get_state(tl_id)
                except Exception as e:
                    print(f"Warning: Error getting state for {tl_id}: {e}")
                    states[tl_id] = np.zeros(self.observation_space_size)
                    
                try:
                    rewards[tl_id] = self._get_reward(tl_id)
                except Exception as e:
                    print(f"Warning: Error getting reward for {tl_id}: {e}")
                    rewards[tl_id] = 0.0
            
            # Check if episode is done
            dones = {tl_id: self.episode_step >= self.episode_length 
                    for tl_id in self.traffic_lights}
            
            # Additional info
            info = {
                'step': self.episode_step,
                'waiting_times': dict(self.waiting_times),
                'throughput': dict(self.throughput)
            }
            
            return states, rewards, dones, info
            
        except Exception as e:
            print(f"Warning: Error in step: {e}")
            # Return safe defaults
            states = {tl_id: np.zeros(self.observation_space_size) 
                     for tl_id in self.traffic_lights}
            rewards = {tl_id: 0.0 for tl_id in self.traffic_lights}
            dones = {tl_id: True for tl_id in self.traffic_lights}
            info = {'error': str(e)}
            return states, rewards, dones, info
        
    def reset(self):
        """Reset the environment"""
        # Close existing simulation
        if traci.isLoaded():
            traci.close()
            
        # Start new simulation
        self.start_simulation()
        self.episode_step = 0
        
        # Reset metrics
        self.waiting_times.clear()
        self.throughput.clear()
        
        # Get initial states
        states = {}
        for tl_id in self.traffic_lights:
            try:
                states[tl_id] = self._get_state(tl_id)
            except Exception as e:
                states[tl_id] = np.zeros(self.observation_space_size)
        
        return states
        
    def close(self):
        """Close the environment"""
        if traci.isLoaded():
            traci.close()
            
    @property
    def intersection_ids(self):
        """Return list of traffic light IDs"""
        return list(self.traffic_lights.keys())
        
    def get_intersection_ids(self):
        """Legacy method for compatibility"""
        return self.intersection_ids