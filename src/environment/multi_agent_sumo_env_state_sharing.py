# src/environment/multi_agent_sumo_env_state_sharing.py

import os
import sys
import traci
import sumolib
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class MultiAgentSumoEnvStateSharing:
    """
    Multi-agent SUMO environment with hierarchical state sharing capabilities
    """
    def __init__(
        self,
        net_file: str,
        route_file: str,
        use_gui: bool = False,
        num_seconds: int = 3600,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 50,
        max_depart_delay: int = 100000,
        time_to_teleport: int = -1,
        sumo_seed: int = 42,
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
        max_steps: Optional[int] = None,
    ):
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.max_depart_delay = max_depart_delay
        self.time_to_teleport = time_to_teleport
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.episode_step = 0
        self.max_steps = max_steps if max_steps is not None else int(self.num_seconds / self.delta_time)

        # Initialize SUMO
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            raise ImportError("Please declare environment variable 'SUMO_HOME'")
            
        # Initialize traffic light data structures
        self.traffic_signals = {}
        self.observation_spaces = {}
        self.action_spaces = {}
        
        # Initialize episode tracking
        self.run = 0
        self.metrics = None
        
        # Initialize hierarchical structures
        self.regions = defaultdict(list)  # Region ID -> List of intersections
        self.region_states = {}  # Region ID -> Aggregated state
        self.neighbor_map = {}  # TL ID -> List of neighbor TL IDs
        
        # Load network and initialize structures
        self._load_network()

    def _load_network(self):
        """Load SUMO network and initialize traffic signals"""
        net = sumolib.net.readNet(self.net_file)
        
        # Get all traffic lights and their connections
        for tls in net.getTrafficLights():
            tl_id = tls.getID()
            
            # Store traffic light and its controlled lanes
            controlled_lanes = []
            controlled_links = []
            for connection in tls.getConnections():
                controlled_lanes.append(connection[0].getID())  # Lane object already
                controlled_links.append((connection[0].getID(),
                                      connection[1].getID()))
            
            self.traffic_signals[tl_id] = {
                'lanes': controlled_lanes,
                'links': controlled_links,
                'neighbors': set()  # Will be populated in _setup_hierarchical_structure
            }
            
        # Filter out invalid traffic lights after SUMO starts
        self._start_simulation()
        valid_tls = self._filter_valid_traffic_lights(list(self.traffic_signals.keys()))
        self.traffic_signals = {tl_id: self.traffic_signals[tl_id] for tl_id in valid_tls}
        traci.close()
        
        # Set up hierarchical structure
        self._setup_hierarchical_structure(net)

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
                
            except Exception:
                continue
        
        return valid_traffic_lights

    def _start_simulation(self):
        """Start SUMO simulation"""
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--max-depart-delay", str(self.max_depart_delay),
            "--waiting-time-memory", "10000",
            "--time-to-teleport", str(self.time_to_teleport),
            "--no-step-log",
            "--no-warnings"
        ]
        
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")
            
        # Start SUMO
        traci.start(sumo_cmd)
        
        # Initialize traffic light programs
        for tl_id, tl_data in self.traffic_signals.items():
            # Get the current program
            program_id = traci.trafficlight.getProgram(tl_id)
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            phases = logic.phases
            
            # Count green phases (any phase that has at least one 'G')
            green_phases = []
            for i, phase in enumerate(phases):
                if 'G' in phase.state:
                    green_phases.append(i)
            
            # Store program information
            tl_data['program_id'] = program_id
            tl_data['green_phases'] = green_phases
            tl_data['num_green_phases'] = len(green_phases)
            tl_data['num_phases'] = len(phases)
            
            # Set initial phase to first green phase
            if green_phases:
                traci.trafficlight.setPhase(tl_id, green_phases[0])
            
        # Run one step to ensure everything is loaded
        traci.simulationStep()

    def _setup_hierarchical_structure(self, net):
        """
        Optimized hierarchical structure setup with efficient region assignment
        """
        # Create spatial index for efficient neighbor finding
        tl_positions = {}
        for tl_id in self.traffic_signals:
            # Get node position from controlled lanes
            lanes = self.traffic_signals[tl_id]['lanes']
            if lanes:
                # Get position from first controlled lane
                lane = net.getLane(lanes[0])
                # Get the center of the lane
                shape = lane.getShape()
                if shape:
                    x_coords = [p[0] for p in shape]
                    y_coords = [p[1] for p in shape]
                    x = sum(x_coords) / len(x_coords)
                    y = sum(y_coords) / len(y_coords)
                    tl_positions[tl_id] = (x, y)

        # Efficient region assignment using grid-based approach
        grid_size = 500  # meters
        for tl_id, (x, y) in tl_positions.items():
            region_x = int(x / grid_size)
            region_y = int(y / grid_size)
            region_id = f"R_{region_x}_{region_y}"
            self.regions[region_id].append(tl_id)

        # Efficient neighbor assignment using vectorized distance calculation
        positions = np.array(list(tl_positions.values()))
        tl_ids = list(tl_positions.keys())
        
        for i, tl_id in enumerate(tl_ids):
            # Calculate distances to all other TLs efficiently
            dists = np.sqrt(np.sum((positions - positions[i])**2, axis=1))
            # Get indices of nearest neighbors (excluding self)
            nearest_indices = np.argsort(dists)[1:5]  # Get 4 nearest neighbors
            self.traffic_signals[tl_id]['neighbors'] = {tl_ids[j] for j in nearest_indices}

    def reset(self) -> Dict:
        """Reset the environment"""
        if self.run != 0:
            traci.close()
            
        self._start_simulation()
        self.run += 1
        self.episode_step = 0
        
        # Reset metrics
        self.metrics = {
            'step_waiting_times': [],
            'step_speeds': [],
            'step_queues': [],
            'step_throughput': [],
            'total_waiting_time': 0,
            'total_speed': 0,
            'total_queued': 0,
            'total_throughput': 0
        }
        
        # Get initial states for all traffic lights
        states = {}
        for tl_id in self.traffic_signals:
            states[tl_id] = self._get_state(tl_id, include_neighbors=True)
            
        return states

    def _get_states(self) -> Dict:
        """Get states for all traffic lights"""
        states = {}
        for tl_id in self.traffic_signals:
            states[tl_id] = self._get_state(tl_id)
        return states

    def _get_state(self, tl_id: str, include_neighbors: bool = True) -> Dict:
        """
        Optimized state collection focusing on wait time reduction
        """
        state = {}
        
        # Get local state efficiently
        lanes = self.traffic_signals[tl_id]['lanes']
        lane_vehicles = {lane: traci.lane.getLastStepVehicleIDs(lane) for lane in lanes}
        
        queues = []
        waiting_times = []
        speeds = []
        lane_densities = []  # Added lane density metric
        
        max_waiting_time = 0  # Track maximum waiting time
        total_vehicles = 0
        
        for lane in lanes:
            vehicles = lane_vehicles[lane]
            total_vehicles += len(vehicles)
            
            # Queue calculation with waiting time threshold
            queue = sum(1 for vid in vehicles if traci.vehicle.getSpeed(vid) < 0.3)  # Increased threshold
            queues.append(queue)
            
            # Enhanced waiting time calculation
            vehicle_wait_times = []
            for vid in vehicles:
                wait_time = traci.vehicle.getAccumulatedWaitingTime(vid)
                max_waiting_time = max(max_waiting_time, wait_time)
                vehicle_wait_times.append(wait_time)
            
            avg_wait_time = np.mean(vehicle_wait_times) if vehicle_wait_times else 0
            waiting_times.append(avg_wait_time)
            
            # Speed calculation with minimum threshold
            if vehicles:
                speeds_array = np.array([max(0.1, traci.vehicle.getSpeed(vid)) for vid in vehicles])
                avg_speed = np.mean(speeds_array)
            else:
                avg_speed = traci.lane.getMaxSpeed(lane)
            speeds.append(avg_speed)
            
            # Lane density calculation
            lane_length = traci.lane.getLength(lane)
            density = len(vehicles) / lane_length if lane_length > 0 else 0
            lane_densities.append(density)
        
        # Combine metrics with emphasis on waiting time
        state['own_state'] = np.concatenate([
            queues,
            waiting_times,
            speeds,
            lane_densities,
            [traci.trafficlight.getPhase(tl_id)],
            [max_waiting_time / 100.0],  # Normalized maximum waiting time
            [total_vehicles / len(lanes)]  # Average vehicles per lane
        ])
        
        # Get neighbor states efficiently
        if include_neighbors:
            neighbor_states = {}
            for neighbor_id in self.traffic_signals[tl_id]['neighbors']:
                if neighbor_id in self.traffic_signals:
                    neighbor_state = self._get_state(neighbor_id, include_neighbors=False)
                    neighbor_states[neighbor_id] = neighbor_state['own_state']
            state['neighbor_states'] = neighbor_states
        
        # Get region state efficiently
        region_id = next(rid for rid, tls in self.regions.items() if tl_id in tls)
        if region_id in self.region_states:
            state['region_state'] = self.region_states[region_id]
        
        return state

    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """Execute one step of the environment"""
        # Apply actions
        for tl_id, action in actions.items():
            self._apply_action(tl_id, action)
            
        # Run simulation for delta_time
        for _ in range(self.delta_time):
            traci.simulationStep()
        self.episode_step += 1
            
        # Get new states
        states = self._get_states()
        
        # Calculate rewards
        rewards = {tl_id: self._calculate_reward(tl_id) 
                  for tl_id in self.traffic_signals}
        
        # Check if simulation is done
        dones = {tl_id: self.episode_step >= self.max_steps 
                for tl_id in self.traffic_signals}
        dones['__all__'] = self.episode_step >= self.max_steps
        
        # Get info
        info = self._get_info()
        
        return states, rewards, dones, info

    def _apply_action(self, tl_id: str, action: int):
        """Apply action to traffic light"""
        if self.fixed_ts:
            return
            
        # Get available green phases and current phase
        green_phases = self.traffic_signals[tl_id]['green_phases']
        current_phase = traci.trafficlight.getPhase(tl_id)
        
        # Convert action to target green phase
        target_phase = green_phases[action % len(green_phases)]
        
        # Only change if we're not already in the target phase
        if current_phase != target_phase:
            traci.trafficlight.setPhase(tl_id, target_phase)

    def _calculate_reward(self, tl_id: str) -> float:
        """
        Enhanced reward calculation focusing on wait time reduction
        """
        reward = 0
        lanes = self.traffic_signals[tl_id]['lanes']
        
        total_waiting_time = 0
        max_waiting_time = 0
        total_vehicles = 0
        queue_penalty = 0
        
        for lane in lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            total_vehicles += len(vehicles)
            
            lane_queue = 0
            for vid in vehicles:
                # Progressive waiting time penalty
                wait_time = traci.vehicle.getAccumulatedWaitingTime(vid)
                total_waiting_time += wait_time
                max_waiting_time = max(max_waiting_time, wait_time)
                
                # Exponential penalty for long waits
                if wait_time > 0:
                    reward -= 0.1 * (wait_time / 10) ** 2
                
                # Queue length penalty
                if wait_time > 0:
                    lane_queue += 1
            
            # Progressive queue penalty
            if lane_queue > 0:
                queue_penalty -= 0.2 * (lane_queue ** 1.5)
            
            # Speed reward
            speed = traci.lane.getLastStepMeanSpeed(lane)
            max_speed = traci.lane.getMaxSpeed(lane)
            speed_ratio = speed / max_speed if max_speed > 0 else 0
            reward += 0.3 * speed_ratio
        
        # Add queue penalty
        reward += queue_penalty
        
        # Bonus for keeping average wait time under target
        if total_vehicles > 0:
            avg_wait_time = total_waiting_time / total_vehicles
            if avg_wait_time < 300:  # Target: 300 seconds
                reward += 2.0
            
            # Progressive bonus for very low wait times
            if avg_wait_time < 200:
                reward += 1.0
            if avg_wait_time < 100:
                reward += 1.0
        
        return reward

    def _get_dones(self) -> Dict:
        """Check if episodes are done"""
        dones = {tl_id: False for tl_id in self.traffic_signals}
        dones['__all__'] = traci.simulation.getTime() > self.num_seconds
        return dones

    def _get_info(self) -> Dict:
        """Get additional information"""
        info = {}
        for tl_id in self.traffic_signals:
            lanes = self.traffic_signals[tl_id]['lanes']
            
            # Collect metrics
            total_waiting_time = 0
            total_queue_length = 0
            total_throughput = 0
            avg_speed = 0
            num_vehicles = 0
            
            for lane in lanes:
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                if vehicles:
                    # Waiting time
                    total_waiting_time += sum(traci.vehicle.getWaitingTime(veh) 
                                            for veh in vehicles)
                    
                    # Queue length
                    total_queue_length += len([veh for veh in vehicles 
                                             if traci.vehicle.getSpeed(veh) < 0.1])
                    
                    # Throughput
                    total_throughput += len(vehicles)
                    
                    # Average speed
                    avg_speed += sum(traci.vehicle.getSpeed(veh) for veh in vehicles)
                    num_vehicles += len(vehicles)
            
            # Store metrics
            info[tl_id] = {
                'waiting_time': total_waiting_time,
                'queue_length': total_queue_length,
                'throughput': total_throughput,
                'average_speed': avg_speed / max(1, num_vehicles),
                'num_vehicles': num_vehicles
            }
            
        # Add global metrics
        info['global'] = {
            'time': traci.simulation.getTime(),
            'total_vehicles': traci.vehicle.getIDCount(),
            'arrived_vehicles': traci.simulation.getArrivedNumber(),
            'departed_vehicles': traci.simulation.getDepartedNumber()
        }
        
        return info

    def close(self):
        """Close the environment"""
        traci.close()

    def __del__(self):
        """Ensure environment is closed on deletion"""
        try:
            self.close()
        except:
            pass