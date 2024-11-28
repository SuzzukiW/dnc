# src/environment/multi_agent_sumo_env.py

import os
import sys
import traci
import numpy as np
from gymnasium import Env, spaces
import sumolib
from collections import defaultdict
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import logging
import traci

logger = logging.getLogger(__name__)

class MultiAgentSumoEnvironment(Env):
    """
    Multi-agent environment for traffic signal control
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
                logger.warning(f"No valid phases found for {tl_id}, using first phase")
            return valid_phases
        except Exception as e:
            logger.error(f"Error getting phases for {tl_id}: {str(e)}")
            return [0]
        
    @property
    def observation_space_size(self):
        """Get the size of the observation space"""
        # All agents have the same observation space size
        return next(iter(self.observation_spaces.values())).shape[0]
    
    @property
    def action_space_size(self):
        """Get the size of the action space"""
        # All agents have the same action space size
        return next(iter(self.action_spaces.values())).shape[0]
    
    def __init__(self,
                config: Dict = {}):
        """Initialize environment"""
        # Extract configuration
        full_config = config.get('config', config)
        
        # Handle use_gui separately
        self.use_gui = full_config.get('use_gui', False)
        
        # Set up configuration parameters with defaults
        self.net_file = full_config.get('net_file', os.path.join('Version1', '2024-11-05-18-42-37', 'osm.net.xml.gz'))
        self.route_file = full_config.get('route_file', os.path.join('Version1', '2024-11-05-18-42-37', 'osm.passenger.trips.xml'))
        self.out_csv_name = full_config.get('out_csv_name', 'metrics.csv')
        self.num_seconds = full_config.get('num_seconds', 1800)
        self.delta_time = full_config.get('delta_time', 8)
        self.yellow_time = full_config.get('yellow_time', 3)
        self.min_green = full_config.get('min_green', 12)
        self.max_green = full_config.get('max_green', 45)
        self.max_depart_delay = full_config.get('max_depart_delay', 100000)
        self.time_to_teleport = full_config.get('time_to_teleport', -1)
        self.neighbor_distance = full_config.get('neighbor_distance', 100)
        
        # Validate network and route files
        if not os.path.exists(self.net_file):
            logger.warning(f"Network file {self.net_file} not found. Using default configuration.")
            self.net_file = os.path.join('Version1', '2024-11-05-18-42-37', 'osm.net.xml.gz')
        
        # If network file is gzipped, decompress it
        if self.net_file.endswith('.gz'):
            import gzip
            import shutil
            
            # Create a decompressed version of the network file
            decompressed_net_file = self.net_file.replace('.gz', '')
            if not os.path.exists(decompressed_net_file):
                with gzip.open(self.net_file, 'rb') as f_in:
                    with open(decompressed_net_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            
            # Use the decompressed file
            self.net_file = decompressed_net_file
        
        if not os.path.exists(self.route_file):
            logger.warning(f"Route file {self.route_file} not found. Using default configuration.")
            self.route_file = os.path.join('Version1', '2024-11-05-18-42-37', 'osm.passenger.trips.xml')
        
        # Store full configuration
        self.config = full_config
        
        # SUMO initialization
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            raise ValueError("Please declare SUMO_HOME environment variable")
            
        # Close any existing SUMO connections
        try:
            traci.close()
        except:
            pass
            
        self.sumo_binary = 'sumo-gui' if self.use_gui else 'sumo'
        self.sumo_cmd = [
            self.sumo_binary,
            '-n', self.net_file,
            '-r', self.route_file,
            '--max-depart-delay', str(self.max_depart_delay),
            '--waiting-time-memory', '10000',
            '--time-to-teleport', str(self.time_to_teleport),
            '--no-warnings',  # Suppress warnings
            '--no-step-log',  # Suppress step logs
            '--collision.action', 'teleport',  # Handle collisions with teleport
            '--collision.mingap-factor', '0',  # Reduce minimum gap
            '--time-to-impatience', '30',  # Reduce time to impatience
            '--random',  # Add some randomness to vehicle behavior
            '--device.rerouting.probability', '0.8',  # Enable dynamic rerouting
            '--device.rerouting.period', '60',  # Rerouting check period
            '--lanechange.duration', '2',  # Faster lane changes
        ]
        
        # Initialize connection with sumo
        traci.start(self.sumo_cmd)
        
        # Get traffic light IDs and initialize agent data
        self.traffic_lights = traci.trafficlight.getIDList()
        self.num_agents = len(self.traffic_lights)
        
        # Create network graph for neighbor identification
        self.net = sumolib.net.readNet(self.net_file)
        self.neighbor_map = self._build_neighbor_map()
        
        # Initialize observation and action spaces
        self.observation_spaces = {}
        self.action_spaces = {}
        
        # Initialize valid phases for each traffic light
        self.valid_phases = {tl_id: self._init_valid_phases(tl_id) 
                            for tl_id in self.traffic_lights}
        
        # Initialize spaces based on valid phases
        max_lanes = max(len(traci.trafficlight.getControlledLanes(tl_id)) 
                       for tl_id in self.traffic_lights)
        
        # State space: [queue_length, waiting_time, density] for each lane
        # Use max_lanes to ensure consistent state size across all agents
        state_size = max_lanes * 3  # Fixed size for all agents
        
        # Find max number of valid phases across all traffic lights
        max_phases = max(len(self.valid_phases[tl_id]) for tl_id in self.traffic_lights)
        
        # Create observation and action spaces with consistent sizes
        for tl_id in self.traffic_lights:
            self.observation_spaces[tl_id] = spaces.Box(
                low=0,
                high=1,
                shape=(state_size,),
                dtype=np.float32
            )
            
            # Action space: continuous action space with fixed dimension for all agents
            self.action_spaces[tl_id] = spaces.Box(
                low=-1,
                high=1,
                shape=(max_phases,),  # Use max_phases for all agents
                dtype=np.float32
            )
        
        # Initialize current phase and yellow phase mapping
        self.current_phase = {tl: self.valid_phases[tl][0] if self.valid_phases[tl] else 0 
                            for tl in self.traffic_lights}
        self.yellow_phase_dict = self._init_yellow_phases()
        
        # Initialize reward calculation parameters
        self.max_queue_length = 10
        self.max_waiting_time = 100
        self.max_speed = 50
        self.max_throughput = 20
        self.max_pressure = 10
        
    def _build_neighbor_map(self):
        """Build a map of neighboring traffic lights based on distance"""
        neighbor_map = defaultdict(list)
        
        # Create a graph of the road network
        G = nx.Graph()
        for edge in self.net.getEdges():
            from_node = edge.getFromNode().getID()
            to_node = edge.getToNode().getID()
            length = edge.getLength()
            G.add_edge(from_node, to_node, weight=length)
        
        # Get junction information for each traffic light
        tl_junctions = {}
        for tl_id in self.traffic_lights:
            # Get controlled lanes for this traffic light
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            # Find the junction nodes connected to these lanes
            junctions = set()
            for lane in controlled_lanes:
                # Skip internal lanes (those starting with ':')
                if lane.startswith(':'):
                    continue
                    
                # Extract edge ID from lane ID (remove lane index)
                edge_id = lane.split('_')[0]
                try:
                    edge = self.net.getEdge(edge_id)
                    junctions.add(edge.getFromNode().getID())
                    junctions.add(edge.getToNode().getID())
                except KeyError:
                    # Skip if edge not found
                    continue
            
            # Store the first junction as the primary one for this traffic light
            if junctions:
                tl_junctions[tl_id] = list(junctions)[0]
        
        # Find neighbors based on network distance
        for tl1 in self.traffic_lights:
            if tl1 not in tl_junctions:
                continue
                
            junction1 = tl_junctions[tl1]
            for tl2 in self.traffic_lights:
                if tl2 not in tl_junctions or tl1 == tl2:
                    continue
                    
                junction2 = tl_junctions[tl2]
                try:
                    # Calculate shortest path distance between junctions
                    distance = nx.shortest_path_length(G, junction1, junction2, weight='weight')
                    if distance <= self.neighbor_distance:
                        neighbor_map[tl1].append(tl2)
                except nx.NetworkXNoPath:
                    continue
        
        # If any traffic light has no neighbors, assign nearest traffic lights
        for tl_id in self.traffic_lights:
            if not neighbor_map[tl_id]:
                # Find nearest traffic lights
                if tl_id in tl_junctions:
                    distances = []
                    junction1 = tl_junctions[tl_id]
                    for other_tl in self.traffic_lights:
                        if other_tl != tl_id and other_tl in tl_junctions:
                            junction2 = tl_junctions[other_tl]
                            try:
                                dist = nx.shortest_path_length(G, junction1, junction2, weight='weight')
                                distances.append((other_tl, dist))
                            except nx.NetworkXNoPath:
                                continue
                    
                    # Add the nearest traffic light as a neighbor
                    if distances:
                        nearest_tl = min(distances, key=lambda x: x[1])[0]
                        neighbor_map[tl_id].append(nearest_tl)
        
        return neighbor_map
    
    def _init_yellow_phases(self):
        """Initialize yellow phases for each traffic light"""
        yellow_dict = {}
        for tl in self.traffic_lights:
            phases = traci.trafficlight.getAllProgramLogics(tl)[0].phases
            yellow_dict[tl] = [i for i, p in enumerate(phases) if 'y' in p.state.lower()]
            if not yellow_dict[tl]:  # If no yellow phase found, use first phase
                yellow_dict[tl] = [0]
        return yellow_dict
    
    def _get_state(self, traffic_light_id):
        """Get state for a specific traffic light"""
        controlled_lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
        max_lanes = max(len(traci.trafficlight.getControlledLanes(tl_id)) 
                       for tl_id in self.traffic_lights)
        
        state = []
        # Add metrics for actual lanes
        for lane in controlled_lanes:
            queue_length = traci.lane.getLastStepHaltingNumber(lane)
            waiting_time = sum(traci.vehicle.getWaitingTime(veh) 
                             for veh in traci.lane.getLastStepVehicleIDs(lane))
            density = len(traci.lane.getLastStepVehicleIDs(lane)) / traci.lane.getLength(lane)
            
            state.extend([
                min(1.0, queue_length / 10.0),
                min(1.0, waiting_time / 100.0),
                min(1.0, density)
            ])
        
        # Pad state with zeros if needed
        padding_size = (max_lanes - len(controlled_lanes)) * 3
        if padding_size > 0:
            state.extend([0.0] * padding_size)
        
        return np.array(state, dtype=np.float32)
    
    def _apply_action(self, action_dict):
        """Apply actions to the traffic lights with robust connection handling"""
        try:
            if not traci.isLoaded():
                logger.warning("SUMO connection lost, attempting to reconnect...")
                traci.start(self.sumo_cmd)
            
            for tl_id, action in action_dict.items():
                try:
                    # Get current phase
                    current_phase = traci.trafficlight.getPhase(tl_id)
                    
                    # Get valid phases for this traffic light
                    valid_phases = self.valid_phases[tl_id]
                    if not valid_phases:
                        continue
                    
                    # Use only the first n components of the action vector where n is the number of valid phases
                    num_valid_phases = len(valid_phases)
                    action = action[:num_valid_phases]  # Truncate to actual number of phases
                    
                    # Find the phase with highest action value
                    phase_idx = np.argmax(action)
                    next_phase = valid_phases[phase_idx]
                    
                    # Only change phase if different
                    if current_phase != next_phase:
                        traci.trafficlight.setPhase(tl_id, next_phase)
                        self._update_traffic_pressure(tl_id)
                
                except traci.exceptions.TraCIException as e:
                    logger.error(f"Error applying action to traffic light {tl_id}: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in _apply_action: {str(e)}")
            logger.error("Action error details:", exc_info=True)
            raise
    
    def _update_traffic_pressure(self, tl_id):
        """Update traffic pressure for a traffic light after phase change"""
        try:
            incoming_lanes = traci.trafficlight.getControlledLanes(tl_id)
            outgoing_lanes = []
            
            # Get outgoing lanes
            for lane in incoming_lanes:
                links = traci.lane.getLinks(lane)
                outgoing_lanes.extend([link[0] for link in links])
            
            # Calculate pressure as difference between incoming and outgoing vehicles
            incoming_count = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in incoming_lanes)
            outgoing_count = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in outgoing_lanes)
            
            # Update pressure (store in instance variable if needed)
            pressure = incoming_count - outgoing_count
            if not hasattr(self, 'traffic_pressure'):
                self.traffic_pressure = {}
            self.traffic_pressure[tl_id] = pressure
            
        except traci.exceptions.TraCIException as e:
            logger.error(f"Error updating traffic pressure for {tl_id}: {str(e)}")
            self.traffic_pressure[tl_id] = 0
    
    def _calculate_reward(self, tls_id):
        """Calculate reward for a traffic light"""
        reward = 0.0
        
        # Get traffic light metrics
        queue_length = self._get_queue_length(tls_id)
        waiting_time = self._get_waiting_time(tls_id)
        mean_speed = self._get_mean_speed(tls_id)
        throughput = self._get_throughput(tls_id)
        pressure = self._get_traffic_pressure(tls_id)
        
        # Normalize metrics
        norm_queue = min(1.0, queue_length / self.max_queue_length)
        norm_waiting = min(1.0, waiting_time / self.max_waiting_time)
        norm_speed = mean_speed / self.max_speed if mean_speed > 0 else 0
        norm_throughput = min(1.0, throughput / self.max_throughput)
        norm_pressure = min(1.0, pressure / self.max_pressure)
        
        # Get reward weights from config, using defaults if not specified
        weights = self.config.get('reward_weights', {})
        queue_weight = weights.get('queue_length_weight', -0.4)
        waiting_weight = weights.get('waiting_time_weight', -0.3)
        speed_weight = weights.get('speed_reward_weight', 0.3)
        flow_weight = weights.get('flow_reward_weight', 0.4)
        phase_weight = weights.get('phase_efficiency_weight', 0.2)
        neighbor_weight = weights.get('neighbor_reward_weight', 0.2)
        
        # Calculate component rewards
        queue_reward = queue_weight * norm_queue
        waiting_reward = waiting_weight * norm_waiting
        speed_reward = speed_weight * norm_speed
        flow_reward = flow_weight * norm_throughput
        pressure_reward = -0.2 * norm_pressure
        
        # Phase efficiency reward
        current_phase = traci.trafficlight.getPhase(tls_id)
        phase_duration = traci.trafficlight.getPhaseDuration(tls_id)
        env_config = self.config.get('environment', {})
        max_green = env_config.get('max_green_time', self.max_green)
        min_green = env_config.get('min_green_time', self.min_green)
        
        if phase_duration > max_green:
            phase_reward = -0.2
        elif phase_duration < min_green:
            phase_reward = -0.1
        else:
            phase_reward = 0.1
            
        # Neighbor coordination reward
        neighbor_reward = 0.0
        neighbors = self._get_neighbor_lights(tls_id)
        for neighbor in neighbors:
            if self._phases_are_coordinated(tls_id, neighbor):
                neighbor_reward += 0.1
        
        # Combine rewards
        reward = (queue_reward + waiting_reward + speed_reward + 
                 flow_reward + pressure_reward + 
                 phase_weight * phase_reward + 
                 neighbor_weight * neighbor_reward)
        
        # Apply reward scaling
        reward_scale = self.config.get('reward_scaling', 0.1)
        reward *= reward_scale
        
        return reward
    
    def _get_queue_length(self, tls_id):
        """Get total queue length for a traffic light"""
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes)
    
    def _get_waiting_time(self, tls_id):
        """Get total waiting time for a traffic light"""
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        return sum(traci.lane.getWaitingTime(lane) for lane in controlled_lanes)
    
    def _get_mean_speed(self, tls_id):
        """Get mean speed for a traffic light"""
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        return sum(traci.lane.getLastStepMeanSpeed(lane) for lane in controlled_lanes) / len(controlled_lanes)
    
    def _get_throughput(self, tls_id):
        """Get throughput for a traffic light"""
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        return sum(traci.lane.getLastStepVehicleNumber(lane) for lane in controlled_lanes)
    
    def _get_traffic_pressure(self, tls_id):
        """Get traffic pressure for a traffic light"""
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        return sum(traci.lane.getLastStepOccupancy(lane) for lane in controlled_lanes)
    
    def _get_neighbor_lights(self, tls_id):
        """Get neighboring traffic lights"""
        return self.neighbor_map[tls_id]
    
    def _phases_are_coordinated(self, tls1, tls2):
        """Check if two traffic lights have coordinated phases"""
        phase1 = traci.trafficlight.getPhase(tls1)
        phase2 = traci.trafficlight.getPhase(tls2)
        
        # Get vehicle counts in shared lanes
        shared_lanes = self._get_shared_lanes(tls1, tls2)
        if not shared_lanes:
            return True  # No shared lanes means no coordination needed
            
        vehicles1 = sum(len(traci.lane.getLastStepVehicleIDs(lane)) 
                       for lane in shared_lanes[tls1])
        vehicles2 = sum(len(traci.lane.getLastStepVehicleIDs(lane)) 
                       for lane in shared_lanes[tls2])
                       
        # Check if phases complement each other based on traffic load
        return abs(vehicles1 - vehicles2) < 5  # Allow small difference
    
    def _get_shared_lanes(self, tls1, tls2):
        """Get shared lanes between two traffic lights"""
        controlled_lanes1 = traci.trafficlight.getControlledLanes(tls1)
        controlled_lanes2 = traci.trafficlight.getControlledLanes(tls2)
        
        shared_lanes = {}
        for lane in controlled_lanes1:
            if lane in controlled_lanes2:
                shared_lanes[tls1] = [lane]
                shared_lanes[tls2] = [lane]
                break
        
        return shared_lanes
    
    def reset(self, seed=None, return_info=False):
        """Reset the environment with connection handling"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        try:
            if traci.isLoaded():
                traci.close()
            traci.start(self.sumo_cmd)
            traci.simulationStep()
            
            states = {}
            for tl_id in self.traffic_lights:
                states[tl_id] = self._get_state(tl_id)
            
            if return_info:
                return states, {}
            return states
            
        except Exception as e:
            logger.error(f"Error during reset: {str(e)}")
            # Try to restart simulation
            try:
                if traci.isLoaded():
                    traci.close()
                traci.start(self.sumo_cmd)
                traci.simulationStep()
                
                states = {}
                for tl_id in self.traffic_lights:
                    states[tl_id] = self._get_state(tl_id)
                
                if return_info:
                    return states, {}
                return states
                
            except Exception as e:
                logger.error(f"Failed to reset environment: {str(e)}")
                # Return zero states as fallback
                states = {tl_id: np.zeros(self.observation_spaces[tl_id].shape[0]) 
                        for tl_id in self.traffic_lights}
                if return_info:
                    return states, {}
                return states
    
    def step(self, actions):
        """Execute actions for all agents and return results"""
        try:
            # Apply actions and simulate
            self._apply_action(actions)
            
            try:
                for _ in range(self.delta_time):
                    if not traci.isLoaded():
                        traci.start(self.sumo_cmd)
                    traci.simulationStep()
            except traci.exceptions.TraCIException as e:
                logger.error(f"Error during simulation step: {str(e)}")
                raise
            
            # Get next states and rewards
            next_states = {}
            rewards = {}
            info = {
                'waiting_times': {},
                'queue_lengths': {},
                'speeds': {}
            }
            
            # Calculate states and rewards for each traffic light
            for tl_id in self.traffic_lights:
                next_states[tl_id] = self._get_state(tl_id)
                rewards[tl_id] = self._calculate_reward(tl_id)
                
                # Update info
                waiting_times = {}
                queue_lengths = {}
                speeds = {}
                
                for lane in traci.trafficlight.getControlledLanes(tl_id):
                    waiting_times[lane] = traci.lane.getWaitingTime(lane)
                    queue_lengths[lane] = traci.lane.getLastStepHaltingNumber(lane)
                    speeds[lane] = traci.lane.getLastStepMeanSpeed(lane)
                
                info['waiting_times'].update(waiting_times)
                info['queue_lengths'].update(queue_lengths)
                info['speeds'].update(speeds)
            
            # Check if simulation is done
            done = traci.simulation.getMinExpectedNumber() <= 0
            dones = {tl_id: done for tl_id in self.traffic_lights}
            
            return next_states, rewards, dones, info
            
        except Exception as e:
            logger.error(f"Error during step: {str(e)}")
            logger.error("Step error details:", exc_info=True)
            raise
    
    def close(self):
        """Close the environment"""
        try:
            traci.close()
        except:
            pass
    
    def get_neighbor_map(self):
        """Return the neighbor map for cooperative learning"""
        return self.neighbor_map

    def get_agent_positions(self) -> Dict[str, Tuple[float, float]]:
        """Get positions of all traffic light agents"""
        positions = {}
        
        # Read SUMO network if not already done
        if not hasattr(self, 'net'):
            self.net = sumolib.net.readNet(self.net_file)
        
        for tl_id in self.traffic_lights:
            try:
                # Get controlled lanes for this traffic light
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                if not controlled_lanes:
                    continue
                    
                # Get position from the first controlled lane
                lane = controlled_lanes[0]
                try:
                    # Get the edge from the lane ID
                    edge = self.net.getLane(lane).getEdge()
                    # Use the center of the edge as the traffic light position
                    pos = edge.getCenter()
                    positions[tl_id] = (float(pos[0]), float(pos[1]))
                except:
                    # If we can't get position from lane, try junction
                    try:
                        junctions = self.net.getNode(tl_id)
                        if junctions:
                            pos = junctions.getCoord()
                            positions[tl_id] = (float(pos[0]), float(pos[1]))
                    except:
                        # Use default position if all else fails
                        positions[tl_id] = (0.0, 0.0)
                        
            except traci.exceptions.TraCIException:
                positions[tl_id] = (0.0, 0.0)
        
        return positions
    
    def calculate_hierarchical_rewards(self, traffic_lights, region_manager):
        """Calculate hierarchical rewards for traffic lights"""
        rewards = {}
        global_factor = 0.2
        regional_factor = 0.3
        local_factor = 0.5
        
        for tl_id in traffic_lights:
            # Local metrics
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            waiting_time = self._get_waiting_time(tl_id)
            queue_length = self._get_queue_length(tl_id)
            throughput = self._get_throughput(tl_id)
            
            # Normalize metrics
            num_lanes = len(controlled_lanes)
            normalized_wait = min(1.0, waiting_time / (180.0 * num_lanes))
            normalized_queue = min(1.0, queue_length / (10.0 * num_lanes))
            normalized_throughput = min(1.0, throughput / (20.0 * num_lanes))
            
            # Calculate local reward
            local_reward = (
                -0.4 * normalized_wait
                -0.3 * normalized_queue
                +0.3 * normalized_throughput
            )
            
            # Regional component
            region_ids = region_manager.agent_region_map.get(tl_id, set())
            if region_ids:
                region_rewards = []
                for region_id in region_ids:
                    coordinator = region_manager.coordinators.get(region_id)
                    if coordinator:
                        region_rewards.append(coordinator.get_performance_score())
                regional_reward = np.mean(region_rewards) if region_rewards else 0.0
            else:
                regional_reward = 0.0
                
            # Global metrics
            global_waiting = sum(traci.lane.getWaitingTime(lane) 
                            for tl in traffic_lights 
                            for lane in traci.trafficlight.getControlledLanes(tl))
            global_queue = sum(traci.lane.getLastStepHaltingNumber(lane)
                            for tl in traffic_lights
                            for lane in traci.trafficlight.getControlledLanes(tl))
            total_lanes = sum(len(traci.trafficlight.getControlledLanes(tl)) 
                            for tl in traffic_lights)
            
            global_reward = -(global_waiting + global_queue) / (total_lanes + 1)
            
            # Combine rewards
            rewards[tl_id] = (
                local_factor * local_reward +
                regional_factor * regional_reward +
                global_factor * global_reward
            )
            
            # Scale reward
            rewards[tl_id] *= 100  # Increase reward magnitude
        
        return rewards