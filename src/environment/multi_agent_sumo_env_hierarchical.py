import os
import sys
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# SUMO imports
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")
    
import traci
import sumolib

class HierarchicalTrafficEnv:
    """
    Hierarchical Multi-Agent Environment for Traffic Signal Control
    Implements a two-level hierarchy with regional coordinators and local intersection agents
    """
    
    def __init__(self, config: dict):
        """
        Initialize the hierarchical traffic environment
        
        Args:
            config: Configuration dictionary containing environment parameters
        """
        self.config = config
        
        # SUMO configuration
        self.net_file = config['environment']['sumo_net_file']
        self.route_file = config['environment']['sumo_route_file']
        self.gui = config['environment']['gui']
        self.max_episode_steps = config['environment']['max_episode_steps']
        self.delta_time = config['environment']['delta_time']
        
        # Load SUMO network
        self.net = sumolib.net.readNet(self.net_file)
        
        # Initialize SUMO connection
        self._init_sumo()
        
        # Get and filter valid traffic lights
        self.traffic_lights = self._get_valid_traffic_lights()
        if not self.traffic_lights:
            raise ValueError("No valid traffic lights found in the network")
            
        print(f"Found {len(self.traffic_lights)} valid traffic lights")
        
        # Hierarchical structure setup
        self.num_regions = min(config['hierarchy']['num_regions'], len(self.traffic_lights))
        self.intersections_per_region = config['hierarchy']['intersections_per_region']
        self.regions = self._create_regions()
        
        # State and action spaces
        self.state_size = config['agent']['state_size']
        self.coordinator_state_size = config['agent']['coordinator_state_size']
        self.action_size = config['agent']['action_size']
        
        # Vehicle tracking for metrics
        self.vehicles_seen = set()
        self.vehicles_completed = set()
        
        # Initialize metrics storage like baseline
        self.metrics_history = {
            'total_waiting_time': [],
            'waiting_time_per_vehicle': [],
            'max_waiting_time': [],
            'total_throughput': [],
            'average_speed': [],
            'vehicles_in_network': [],
            'completed_trips': []
        }
        
        # Performance metrics
        self.metrics = defaultdict(list)
        
        # Print debug information
        print("\nEnvironment initialization complete:")
        print(f"- Total traffic lights found: {len(self.traffic_lights)}")
        print(f"- Number of regions: {self.num_regions}")
        print("- Regions distribution:")
        for region_id, tls in self.regions.items():
            print(f"  Region {region_id}: {len(tls)} traffic lights")
        print()
        
    def _init_sumo(self):
        """Initialize SUMO with appropriate configuration"""
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--no-step-log", "true",
            "--waiting-time-memory", str(self.max_episode_steps),
            "--time-to-teleport", "-1",
            "--no-warnings", "true",
            "--duration-log.disable", "true"
        ]
        traci.start(sumo_cmd)
    
    def _get_valid_traffic_lights(self) -> List[str]:
            """
            Get list of valid traffic lights by filtering out problematic ones
            
            Returns:
                List of valid traffic light IDs
            """
            valid_tls = []
            all_tls = traci.trafficlight.getIDList()
            
            # Create a mapping of node IDs to actual nodes
            node_map = {node.getID(): node for node in self.net.getNodes()}
            
            for tl_id in all_tls:
                try:
                    # Check if traffic light has valid programs
                    programs = traci.trafficlight.getAllProgramLogics(tl_id)
                    if not programs:
                        print(f"Skipping {tl_id}: No program logic found")
                        continue
                        
                    # Check if traffic light has phases
                    if not programs[0].phases:
                        print(f"Skipping {tl_id}: No phases found")
                        continue
                        
                    # Check if traffic light controls any lanes
                    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                    if not controlled_lanes:
                        print(f"Skipping {tl_id}: No controlled lanes")
                        continue
                    
                    # Try to find the corresponding node
                    node_found = False
                    
                    # First try: direct match
                    if tl_id in node_map:
                        node = node_map[tl_id]
                        if node.getType() == 'traffic_light':
                            valid_tls.append(tl_id)
                            node_found = True
                    
                    # Second try: partial match
                    if not node_found:
                        for node_id, node in node_map.items():
                            if (node.getType() == 'traffic_light' and 
                                (tl_id in node_id or any(part in node_id for part in tl_id.split('_')))):
                                valid_tls.append(tl_id)
                                node_found = True
                                break
                    
                    # Third try: use junction position
                    if not node_found:
                        try:
                            junction_pos = traci.junction.getPosition(tl_id)
                            if junction_pos:
                                valid_tls.append(tl_id)
                                node_found = True
                        except traci.exceptions.TraCIException:
                            pass
                    
                    if not node_found:
                        print(f"Skipping {tl_id}: No corresponding node found in network")
                    
                except traci.exceptions.TraCIException as e:
                    print(f"Error checking traffic light {tl_id}: {str(e)}")
                    continue
                    
            if not valid_tls:
                raise ValueError("No valid traffic lights found in the network!")
                
            print(f"Found {len(valid_tls)} valid traffic lights out of {len(all_tls)} total")
            return valid_tls
        
    def _create_regions(self) -> Dict[int, List[str]]:
            """
            Create regional groupings of traffic lights based on geographical proximity
            
            Returns:
                Dictionary mapping region IDs to lists of traffic light IDs
            """
            print("\nStarting region creation...")
            regions = {}
            tl_positions = {}
            node_map = {node.getID(): node for node in self.net.getNodes()}
            
            # Get positions for valid traffic lights
            for tl in self.traffic_lights:
                position_found = False
                
                # Try direct node lookup
                if tl in node_map:
                    node = node_map[tl]
                    if node.getType() == 'traffic_light':
                        tl_positions[tl] = node.getCoord()
                        position_found = True
                
                # Try partial matching
                if not position_found:
                    for node_id, node in node_map.items():
                        if (node.getType() == 'traffic_light' and 
                            (tl in node_id or any(part in node_id for part in tl.split('_')))):
                            tl_positions[tl] = node.getCoord()
                            position_found = True
                            break
                
                # Use junction position as fallback
                if not position_found:
                    try:
                        pos = traci.junction.getPosition(tl)
                        tl_positions[tl] = (pos[0], pos[1])
                        position_found = True
                    except traci.exceptions.TraCIException as e:
                        print(f"Warning: Could not find position for traffic light {tl}: {str(e)}")
                        # Use a default position as last resort
                        tl_positions[tl] = (0.0, 0.0)

            print(f"Found positions for {len(tl_positions)} traffic lights")
            
            if len(tl_positions) < self.num_regions:
                self.num_regions = max(1, len(tl_positions) // 2)
                print(f"Adjusting number of regions to {self.num_regions} due to limited traffic lights")
                
            # Use k-means clustering to group intersections into regions
            try:
                from sklearn.cluster import KMeans
                
                # Convert positions to numpy array
                tl_ids = list(tl_positions.keys())
                positions = np.array([tl_positions[tl] for tl in tl_ids])
                
                # Perform clustering
                kmeans = KMeans(n_clusters=self.num_regions, random_state=42)
                cluster_labels = kmeans.fit_predict(positions)
                
                # Create regions dictionary
                for i, tl in enumerate(tl_ids):
                    region_id = int(cluster_labels[i])
                    if region_id not in regions:
                        regions[region_id] = []
                    regions[region_id].append(tl)
                
                print(f"Successfully created {len(regions)} regions")
                for region_id, tls in regions.items():
                    print(f"Region {region_id}: {len(tls)} traffic lights")
                    
                return regions
                
            except Exception as e:
                print(f"Error during clustering: {str(e)}")
                # Fallback to simple division if clustering fails
                tl_list = list(tl_positions.keys())
                tls_per_region = max(1, len(tl_list) // self.num_regions)
                
                for i in range(self.num_regions):
                    start_idx = i * tls_per_region
                    end_idx = start_idx + tls_per_region if i < self.num_regions - 1 else len(tl_list)
                    regions[i] = tl_list[start_idx:end_idx]
                
                print("Used fallback region creation")
                return regions
    
    def get_state(self, tl_id: str) -> np.ndarray:
        """
        Get the state of a single intersection with fixed dimensionality
        
        Args:
            tl_id: Traffic light ID
            
        Returns:
            Fixed-size state array
        """
        max_lanes = 12  # Maximum number of lanes we'll consider
        
        # Get controlled lanes
        lanes = traci.trafficlight.getControlledLanes(tl_id)
        num_lanes = min(len(lanes), max_lanes)
        
        # Initialize state arrays with zeros
        queue_lengths = np.zeros(max_lanes)
        waiting_times = np.zeros(max_lanes)
        vehicle_counts = np.zeros(max_lanes)
        speeds = np.zeros(max_lanes)
        
        # Fill in actual values
        for i, lane in enumerate(lanes[:max_lanes]):
            queue_lengths[i] = traci.lane.getLastStepHaltingNumber(lane)
            waiting_times[i] = traci.lane.getWaitingTime(lane)
            vehicle_counts[i] = traci.lane.getLastStepVehicleNumber(lane)
            speeds[i] = traci.lane.getLastStepMeanSpeed(lane)
        
        # Get traffic light phase information
        current_phase = traci.trafficlight.getPhase(tl_id)
        phase_duration = traci.trafficlight.getPhaseDuration(tl_id)
        total_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
        
        # Normalize phase information
        normalized_phase = current_phase / max(1, total_phases)
        normalized_duration = phase_duration / 90.0  # Assuming max duration of 90 seconds
        
        # Combine all features into state vector
        state = np.concatenate([
            queue_lengths / 10.0,  # Normalize by typical maximum
            waiting_times / 60.0,  # Normalize by 1 minute
            vehicle_counts / 15.0, # Normalize by typical maximum
            speeds / 15.0,        # Normalize by typical maximum speed
            [normalized_phase, normalized_duration]
        ])
        
        return state
    
    def get_coordinator_state(self, region_id: int) -> np.ndarray:
        """
        Get the aggregate state for a regional coordinator
        
        Args:
            region_id: Region identifier
            
        Returns:
            Fixed-size aggregated state array for the region
        """
        region_tls = self.regions[region_id]
        
        if not region_tls:
            # Return zero state if region is empty
            return np.zeros(self.coordinator_state_size)
        
        # Collect states from all intersections in the region
        states = []
        for tl_id in region_tls:
            local_state = self.get_state(tl_id)
            states.append(local_state)
            
        # Convert to numpy array
        states = np.array(states)
        
        # Calculate regional statistics
        mean_state = np.mean(states, axis=0)
        max_state = np.max(states, axis=0)
        min_state = np.min(states, axis=0)
        std_state = np.std(states, axis=0)
        
        # Combine statistics into coordinator state
        coordinator_state = np.concatenate([
            mean_state,
            max_state,
            min_state,
            std_state
        ])
        
        # Ensure fixed size by padding or truncating
        if len(coordinator_state) > self.coordinator_state_size:
            coordinator_state = coordinator_state[:self.coordinator_state_size]
        elif len(coordinator_state) < self.coordinator_state_size:
            padding = np.zeros(self.coordinator_state_size - len(coordinator_state))
            coordinator_state = np.concatenate([coordinator_state, padding])
            
        return coordinator_state
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state values to [0, 1] range"""
        # Define normalization parameters (could be made dynamic)
        max_queue = 20
        max_wait = 300
        max_count = 30
        max_speed = 15
        
        # Split state into components
        queue_length = state[:len(state)//5]
        waiting_time = state[len(state)//5:2*len(state)//5]
        vehicle_count = state[2*len(state)//5:3*len(state)//5]
        speed = state[3*len(state)//5:4*len(state)//5]
        phase_info = state[4*len(state)//5:]
        
        # Normalize each component
        normalized_state = np.concatenate([
            queue_length / max_queue,
            waiting_time / max_wait,
            vehicle_count / max_count,
            speed / max_speed,
            phase_info  # Already normalized by SUMO
        ])
        
        return np.clip(normalized_state, 0, 1)
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], dict]:
        """
        Execute one time step in the environment
        
        Args:
            actions: Dictionary mapping traffic light IDs to actions
            
        Returns:
            Tuple containing:
            - Dictionary of next states for each traffic light
            - Dictionary of rewards for each traffic light
            - Dictionary of done flags for each traffic light
            - Dictionary containing additional information
        """
        # Apply actions
        for tl_id, action in actions.items():
            self._apply_action(tl_id, action)
            
        # Ensure delta_time is an integer for the range function
        delta_steps = int(self.delta_time)
        
        # Simulate for delta_time steps
        for _ in range(delta_steps):
            traci.simulationStep()
            
        # Collect new states and calculate rewards
        new_states = {tl_id: self.get_state(tl_id) for tl_id in self.traffic_lights}
        rewards = self._compute_rewards(actions)
        
        # Check if simulation is done
        simulation_done = traci.simulation.getTime() >= self.max_episode_steps
        dones = {tl_id: simulation_done for tl_id in self.traffic_lights}
        
        # Collect additional info
        info = self._collect_metrics()
        
        return new_states, rewards, dones, info
    
    def _apply_action(self, tl_id: str, action: int):
        """Apply the selected action to the traffic light"""
        # Define possible phase durations
        phase_durations = [10, 20, 30, 40]  # Example durations
        
        # Get current phase
        current_phase = traci.trafficlight.getPhase(tl_id)
        
        # Interpret action as combination of phase selection and duration
        new_phase = action % len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
        duration_idx = action // len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
        duration = phase_durations[duration_idx % len(phase_durations)]
        
        # Set new phase and duration
        traci.trafficlight.setPhase(tl_id, new_phase)
        traci.trafficlight.setPhaseDuration(tl_id, duration)
    
    def _compute_rewards(self, actions: Dict[str, int]) -> Dict[str, float]:
        """
        Compute rewards for each traffic light
        
        Args:
            actions: Dictionary of actions taken by each traffic light
            
        Returns:
            Dictionary mapping traffic light IDs to rewards
        """
        rewards = {}
        
        # Get weights from config
        weights = self.config['training']['reward_weights']
        coordination_weights = self.config['training']['coordination_weights']
        
        for tl_id in self.traffic_lights:
            # Get controlled lanes
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            if not controlled_lanes:
                rewards[tl_id] = 0.0
                continue
                
            # Calculate local metrics
            total_waiting_time = 0
            total_queue_length = 0
            total_throughput = 0
            vehicles = set()
            
            # Get all vehicles in controlled lanes
            for lane in controlled_lanes:
                lane_vehicles = set(traci.lane.getLastStepVehicleIDs(lane))
                vehicles.update(lane_vehicles)
                total_queue_length += traci.lane.getLastStepHaltingNumber(lane)
                
            if not vehicles:
                rewards[tl_id] = 0.0
                continue
                
            # Calculate metrics for vehicles
            for vehicle in vehicles:
                waiting_time = traci.vehicle.getWaitingTime(vehicle)  # In milliseconds
                total_waiting_time += waiting_time
                if traci.vehicle.getSpeed(vehicle) > 0.1:  # Moving vehicle
                    total_throughput += 1
                    
            # Normalize metrics
            num_lanes = len(controlled_lanes)
            avg_waiting_time = (total_waiting_time / len(vehicles)) / 1000.0  # Convert to seconds
            avg_queue_length = total_queue_length / num_lanes
            avg_throughput = total_throughput / len(vehicles)
            
            # Calculate local reward components with increased weight on waiting time
            local_reward = (
                weights['waiting_time'] * (avg_waiting_time / 60.0) +  # Normalize by 1 minute
                weights['queue_length'] * (avg_queue_length / 10.0) +  # Normalize by 10 vehicles
                weights['throughput'] * avg_throughput
            )
            
            # Calculate regional coordination reward
            regional_reward = self._compute_coordination_reward(tl_id)
            
            # Calculate global network state reward
            global_reward = self._compute_global_reward()
            
            # Combine rewards using coordination weights
            rewards[tl_id] = (
                coordination_weights['local'] * local_reward +
                coordination_weights['regional'] * regional_reward +
                coordination_weights['global'] * global_reward
            )
            
            # Scale reward to reasonable range
            rewards[tl_id] *= 100
            
        return rewards
        
    def _compute_global_reward(self) -> float:
        """Calculate reward based on global network state"""
        try:
            # Get all vehicles in the network
            all_vehicles = set(traci.vehicle.getIDList())
            if not all_vehicles:
                return 0.0
                
            # Calculate global metrics
            total_waiting_time = 0
            total_speed = 0
            total_stopped = 0
            
            for vehicle in all_vehicles:
                waiting_time = traci.vehicle.getWaitingTime(vehicle)
                speed = traci.vehicle.getSpeed(vehicle)
                
                total_waiting_time += waiting_time
                total_speed += speed
                if speed < 0.1:  # Stopped vehicle
                    total_stopped += 1
                    
            # Calculate averages
            avg_waiting_time = total_waiting_time / len(all_vehicles)
            avg_speed = total_speed / len(all_vehicles)
            stopped_ratio = total_stopped / len(all_vehicles)
            
            # Combine into global reward (negative because we want to minimize)
            global_reward = (
                -0.7 * (avg_waiting_time / 60.0) +    # Heavily weight waiting time reduction
                -0.2 * stopped_ratio +                # Penalize stopped vehicles
                0.1 * (avg_speed / 13.89)            # Small reward for speed (13.89 m/s = 50 km/h)
            )
            
            return global_reward
            
        except Exception as e:
            print(f"Error calculating global reward: {str(e)}")
            return 0.0
    
    def _compute_coordination_reward(self, tl_id: str) -> float:
            """
            Compute coordination reward based on neighboring intersections using lane positions
            
            Args:
                tl_id: Traffic light ID
                
            Returns:
                Coordination reward value
            """
            try:
                # Get neighboring intersections within the same region
                region_id = None
                for rid, tls in self.regions.items():
                    if tl_id in tls:
                        region_id = rid
                        break
                        
                if region_id is None:
                    return 0.0
                
                # Get controlled lanes for this traffic light
                try:
                    tl_lanes = traci.trafficlight.getControlledLanes(tl_id)
                    if not tl_lanes:
                        return 0.0
                        
                    # Get position of the intersection based on average of its lanes' positions
                    lane_positions = []
                    for lane in tl_lanes:
                        try:
                            # Get the center point of the lane
                            lane_shape = traci.lane.getShape(lane)
                            if lane_shape:
                                mid_point = (
                                    (lane_shape[0][0] + lane_shape[-1][0]) / 2,
                                    (lane_shape[0][1] + lane_shape[-1][1]) / 2
                                )
                                lane_positions.append(mid_point)
                        except traci.exceptions.TraCIException:
                            continue
                    
                    if not lane_positions:
                        return 0.0
                        
                    # Calculate intersection position as average of lane positions
                    tl_pos = (
                        sum(pos[0] for pos in lane_positions) / len(lane_positions),
                        sum(pos[1] for pos in lane_positions) / len(lane_positions)
                    )
                    
                    # Find neighbors in the same region
                    neighbors = []
                    for other_tl in self.regions[region_id]:
                        if other_tl != tl_id:
                            try:
                                # Get other intersection's lanes
                                other_lanes = traci.trafficlight.getControlledLanes(other_tl)
                                if not other_lanes:
                                    continue
                                    
                                # Calculate other intersection's position
                                other_lane_positions = []
                                for lane in other_lanes:
                                    try:
                                        lane_shape = traci.lane.getShape(lane)
                                        if lane_shape:
                                            mid_point = (
                                                (lane_shape[0][0] + lane_shape[-1][0]) / 2,
                                                (lane_shape[0][1] + lane_shape[-1][1]) / 2
                                            )
                                            other_lane_positions.append(mid_point)
                                    except traci.exceptions.TraCIException:
                                        continue
                                
                                if other_lane_positions:
                                    other_pos = (
                                        sum(pos[0] for pos in other_lane_positions) / len(other_lane_positions),
                                        sum(pos[1] for pos in other_lane_positions) / len(other_lane_positions)
                                    )
                                    
                                    # Calculate distance between intersections
                                    distance = np.sqrt(
                                        (tl_pos[0] - other_pos[0])**2 + 
                                        (tl_pos[1] - other_pos[1])**2
                                    )
                                    
                                    # Consider intersections within 200 meters as neighbors
                                    if distance < 200:
                                        neighbors.append((other_tl, other_lanes))
                                        
                            except traci.exceptions.TraCIException:
                                continue
                    
                    if not neighbors:
                        return 0.0
                    
                    # Calculate coordination score
                    coordination_score = 0.0
                    valid_comparisons = 0
                    
                    for neighbor_tl, neighbor_lanes in neighbors:
                        try:
                            # Compare traffic metrics between intersections
                            tl_metrics = []
                            neighbor_metrics = []
                            
                            # Collect metrics for this intersection
                            for lane in tl_lanes:
                                try:
                                    flow = traci.lane.getLastStepVehicleNumber(lane)
                                    speed = traci.lane.getLastStepMeanSpeed(lane)
                                    queue = traci.lane.getLastStepHaltingNumber(lane)
                                    tl_metrics.append((flow, speed, queue))
                                except traci.exceptions.TraCIException:
                                    continue
                                    
                            # Collect metrics for neighbor
                            for lane in neighbor_lanes:
                                try:
                                    flow = traci.lane.getLastStepVehicleNumber(lane)
                                    speed = traci.lane.getLastStepMeanSpeed(lane)
                                    queue = traci.lane.getLastStepHaltingNumber(lane)
                                    neighbor_metrics.append((flow, speed, queue))
                                except traci.exceptions.TraCIException:
                                    continue
                            
                            if tl_metrics and neighbor_metrics:
                                # Calculate average metrics
                                tl_avg = np.mean(tl_metrics, axis=0)
                                neighbor_avg = np.mean(neighbor_metrics, axis=0)
                                
                                # Calculate differences
                                flow_diff = abs(tl_avg[0] - neighbor_avg[0]) / max(1, max(tl_avg[0], neighbor_avg[0]))
                                speed_diff = abs(tl_avg[1] - neighbor_avg[1]) / max(1, max(tl_avg[1], neighbor_avg[1]))
                                queue_diff = abs(tl_avg[2] - neighbor_avg[2]) / max(1, max(tl_avg[2], neighbor_avg[2]))
                                
                                # Combine differences into coordination score
                                score = -(flow_diff + speed_diff + queue_diff) / 3
                                coordination_score += score
                                valid_comparisons += 1
                                
                        except Exception as e:
                            print(f"Warning: Error calculating metrics for {tl_id} and {neighbor_tl}: {str(e)}")
                            continue
                    
                    # Normalize coordination score
                    if valid_comparisons > 0:
                        coordination_score /= valid_comparisons
                        
                    # Scale to reasonable range (-1 to 0)
                    coordination_score = max(-1, min(0, coordination_score))
                    
                    return coordination_score
                    
                except traci.exceptions.TraCIException as e:
                    print(f"Warning: Error getting lanes for traffic light {tl_id}: {str(e)}")
                    return 0.0
                    
            except Exception as e:
                print(f"Warning: Error in coordination reward calculation for {tl_id}: {str(e)}")
                return 0.0
    
    def _collect_metrics(self) -> dict:
        """Collect performance metrics from the environment"""
        metrics = {}
        
        # Get current vehicles and update tracking
        current_vehicles = set(traci.vehicle.getIDList())
        self.vehicles_seen.update(current_vehicles)
        
        # Check for completed trips
        new_completed = self.vehicles_seen - current_vehicles - self.vehicles_completed
        self.vehicles_completed.update(new_completed)
        
        if not current_vehicles:
            waiting_time = 0
            throughput = 0
            avg_speed = 0
            waiting_time_per_vehicle = 0
            max_waiting_time = 0
        else:
            # Vectorized calculations using numpy for better performance
            vehicle_list = list(current_vehicles)
            # Get raw waiting times in milliseconds, just like baseline
            waiting_times = np.array([traci.vehicle.getWaitingTime(vid) for vid in vehicle_list])
            speeds = np.array([traci.vehicle.getSpeed(vid) for vid in vehicle_list])
            distances = np.array([traci.vehicle.getDistance(vid) for vid in vehicle_list])
            
            waiting_time = np.sum(waiting_times)  # Total waiting time
            waiting_time_per_vehicle = waiting_time / len(vehicle_list)  # Average waiting time per vehicle
            throughput = np.sum(distances > 0)
            avg_speed = np.mean(speeds) if len(speeds) > 0 else 0
            max_waiting_time = np.max(waiting_times)
            
        # Store current metrics in history
        self.metrics_history['total_waiting_time'].append(waiting_time)
        self.metrics_history['waiting_time_per_vehicle'].append(waiting_time_per_vehicle)
        self.metrics_history['max_waiting_time'].append(max_waiting_time)
        self.metrics_history['total_throughput'].append(throughput)
        self.metrics_history['average_speed'].append(avg_speed)
        self.metrics_history['vehicles_in_network'].append(len(current_vehicles))
        self.metrics_history['completed_trips'].append(len(self.vehicles_completed))
        
        # Calculate episode metrics using history
        metrics.update({
            'mean_waiting_time (s)': np.mean(self.metrics_history['waiting_time_per_vehicle']) / 1000,  # Convert ms to s
            'mean_throughput (vehicles)': float(np.mean(self.metrics_history['total_throughput'])),
            'mean_speed (km/h)': float(np.mean(self.metrics_history['average_speed']) * 3.6),
            'max_waiting_time (s)': float(np.max(self.metrics_history['max_waiting_time'])) / 1000,  # Convert ms to s
            'completed_trips': len(self.vehicles_completed),
            'total_vehicles': len(self.vehicles_seen),
            'completion_rate (%)': (len(self.vehicles_completed) / len(self.vehicles_seen) * 100) 
                if self.vehicles_seen else 0
        })
        
        return metrics
        
    def reset(self):
        """Reset the environment"""
        # Close existing SUMO connection if any
        try:
            traci.close()
        except:
            pass
            
        # Reset vehicle tracking
        self.vehicles_seen = set()
        self.vehicles_completed = set()
        
        # Reset metrics history
        self.metrics_history = {
            'total_waiting_time': [],
            'waiting_time_per_vehicle': [],
            'max_waiting_time': [],
            'total_throughput': [],
            'average_speed': [],
            'vehicles_in_network': [],
            'completed_trips': []
        }
        
        # Start new SUMO instance
        self._init_sumo()
        
        # Get initial states for all traffic lights
        states = {tl_id: self.get_state(tl_id) for tl_id in self.traffic_lights}
        
        return states
    
    def close(self):
        """Clean up environment"""
        traci.close()
        
    def get_default_state(self) -> np.ndarray:
        """Return a default state vector filled with zeros"""
        return np.zeros(self.state_size)
    
    def render(self, mode='human'):
        """
        Render the environment
        Note: SUMO-GUI is used when self.gui=True
        """
        pass  # SUMO-GUI handles visualization when enabled