# src/environment/multi_agent_sumo_env.py

import os
import sys
import traci
import numpy as np
from gymnasium import Env, spaces
import sumolib
from collections import defaultdict
import networkx as nx

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
                num_seconds=20000,
                max_depart_delay=100000,
                time_to_teleport=-1,
                delta_time=5,
                yellow_time=2,
                min_green=5,
                max_green=50,
                neighbor_distance=100):
        
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
        self.neighbor_distance = neighbor_distance
        
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
        
        # Debug output for phase initialization
        for tl_id in self.traffic_lights:
            print(f"\nTraffic light {tl_id}:")
            try:
                program = traci.trafficlight.getAllProgramLogics(tl_id)[0]
                print(f"Total phases: {len(program.phases)}")
                print(f"Valid phases: {self.valid_phases[tl_id]}")
                if self.valid_phases[tl_id]:
                    print(f"Phase states: {[program.phases[i].state for i in self.valid_phases[tl_id]]}")
            except Exception as e:
                print(f"Error getting program logic: {e}")
        
        # Initialize spaces based on valid phases
        for tl_id in self.traffic_lights:
            num_lanes = len(traci.trafficlight.getControlledLanes(tl_id))
            # State space: [queue_length, waiting_time, density] for each lane
            self.observation_spaces[tl_id] = spaces.Box(
                low=0,
                high=1,
                shape=(num_lanes * 3,),
                dtype=np.float32
            )
            
            # Action space: number of valid phases (minimum 1)
            num_actions = max(len(self.valid_phases[tl_id]), 1)
            self.action_spaces[tl_id] = spaces.Discrete(num_actions)
        
        # Initialize current phase and yellow phase mapping
        self.current_phase = {tl: self.valid_phases[tl][0] if self.valid_phases[tl] else 0 
                            for tl in self.traffic_lights}
        self.yellow_phase_dict = self._init_yellow_phases()
    
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
            for lane_id in controlled_lanes:
                # Skip internal lanes (those starting with ':')
                if lane_id.startswith(':'):
                    continue
                    
                # Extract edge ID from lane ID (remove lane index)
                edge_id = lane_id.split('_')[0]
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
        
        state = []
        for lane in controlled_lanes:
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
        try:
            # Get program logic and check if it exists
            program_logics = traci.trafficlight.getAllProgramLogics(traffic_light_id)
            if not program_logics:
                print(f"Warning: No program logic for traffic light {traffic_light_id}")
                return
            
            program = program_logics[0]
            if not program.phases:
                print(f"Warning: No phases for traffic light {traffic_light_id}")
                return
            
            # Convert action index to actual phase index
            valid_phases = self.valid_phases[traffic_light_id]
            if not valid_phases:
                # If no valid phases, try to find any usable phase
                for i, phase in enumerate(program.phases):
                    if any(c in 'Gg' for c in phase.state):
                        valid_phases = [i]
                        self.valid_phases[traffic_light_id] = valid_phases
                        break
            
            # If still no valid phases, use the first available phase
            if not valid_phases:
                phase_index = 0
            else:
                # Ensure action is within valid range
                action = min(action, len(valid_phases) - 1)
                phase_index = valid_phases[action]
            
            # Ensure phase index is valid
            if phase_index >= len(program.phases):
                print(f"Warning: Phase index {phase_index} out of range for traffic light {traffic_light_id}")
                phase_index = 0
            
            # Apply yellow phase if changing phases
            if self.current_phase[traffic_light_id] != phase_index:
                yellow_phases = self.yellow_phase_dict.get(traffic_light_id, [])
                if yellow_phases:
                    try:
                        yellow_phase = yellow_phases[0]
                        if yellow_phase < len(program.phases):
                            traci.trafficlight.setPhase(traffic_light_id, yellow_phase)
                            for _ in range(self.yellow_time):
                                traci.simulationStep()
                    except traci.exceptions.TraCIException as e:
                        print(f"Warning: Could not set yellow phase for {traffic_light_id}: {e}")
            
            # Set new phase
            try:
                traci.trafficlight.setPhase(traffic_light_id, phase_index)
                self.current_phase[traffic_light_id] = phase_index
            except traci.exceptions.TraCIException as e:
                print(f"Error setting phase {phase_index} for traffic light {traffic_light_id}: {e}")
                print(f"Available phases: {len(program.phases)}")
                print(f"Program phases: {[p.state for p in program.phases]}")
                # Try to find any working phase
                for i in range(len(program.phases)):
                    try:
                        traci.trafficlight.setPhase(traffic_light_id, i)
                        self.current_phase[traffic_light_id] = i
                        break
                    except:
                        continue
        
        except Exception as e:
            print(f"Error handling traffic light {traffic_light_id}: {e}")
            print(f"Current phases: {self.valid_phases.get(traffic_light_id, [])}")
    
    def _get_reward(self, traffic_light_id):
        """Calculate reward for a specific traffic light"""
        controlled_lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
        
        total_waiting_time = 0
        total_queue_length = 0
        
        for lane in controlled_lanes:
            total_waiting_time += traci.lane.getWaitingTime(lane)
            total_queue_length += traci.lane.getLastStepHaltingNumber(lane)
            
        reward = -(total_waiting_time + total_queue_length)
        return reward
    
    def _get_global_reward(self):
        """Calculate global reward across all traffic lights"""
        total_waiting_time = 0
        total_queue_length = 0
        
        for tl in self.traffic_lights:
            controlled_lanes = traci.trafficlight.getControlledLanes(tl)
            for lane in controlled_lanes:
                total_waiting_time += traci.lane.getWaitingTime(lane)
                total_queue_length += traci.lane.getLastStepHaltingNumber(lane)
        
        return -(total_waiting_time + total_queue_length)
    
    def reset(self, seed=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        if traci.isLoaded():
            traci.close()
        traci.start(self.sumo_cmd)
        
        self.current_phase = {tl: 0 for tl in self.traffic_lights}
        
        # Get initial states for all agents
        states = {}
        for tl in self.traffic_lights:
            states[tl] = self._get_state(tl)
        
        return states, {}
    
    def step(self, actions):
        """Execute actions for all agents and return results"""
        # Apply actions for all traffic lights
        for tl_id, action in actions.items():
            self._apply_action(tl_id, action)
        
        # Run simulation for delta_time steps
        for _ in range(self.delta_time):
            traci.simulationStep()
        
        # Get new states, rewards, and dones for all agents
        new_states = {}
        rewards = {}
        dones = {}
        
        # Calculate global reward
        global_reward = self._get_global_reward()
        
        for tl_id in self.traffic_lights:
            new_states[tl_id] = self._get_state(tl_id)
            rewards[tl_id] = self._get_reward(tl_id)
            dones[tl_id] = traci.simulation.getTime() >= self.num_seconds
        
        # Additional info
        info = {
            'time': traci.simulation.getTime(),
            'global_reward': global_reward,
            'total_waiting_time': -global_reward  # Negative since reward is negative of waiting time
        }
        
        # All agents are done when simulation ends
        done = traci.simulation.getTime() >= self.num_seconds
        
        return new_states, rewards, done, False, info
    
    def close(self):
        """Close the environment"""
        if traci.isLoaded():
            traci.close()
    
    def get_neighbor_map(self):
        """Return the neighbor map for cooperative learning"""
        return self.neighbor_map