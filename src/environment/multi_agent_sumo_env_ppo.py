# src/environment/multi_agent_sumo_env_ppo.py

import os
import sys
import numpy as np
import traci
import sumolib
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import gymnasium as gym

# SUMO imports
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

class MultiAgentSumoEnvironmentPPO(gym.Env):
    """
    Multi-agent environment wrapper for SUMO traffic simulation
    Implements the environment for PPO-based traffic signal control
    """
    def __init__(self, config: dict):
        """
        Initialize SUMO environment
        
        Args:
            config: Configuration dictionary containing SUMO and environment parameters
        """
        super().__init__()
        self.config = config
        
        # SUMO configuration
        self.net_file = config['env']['sumo_net_file']
        self.route_file = config['env']['sumo_route_file']
        self.gui = config['env']['sumo_gui']
        self.delta_time = config['env']['sumo_step_length']
        
        # Traffic light parameters
        self.min_green = config['env']['min_green_time']
        self.max_green = config['env']['max_green_time']
        self.yellow_time = config['env']['yellow_time']
        
        # Initialize SUMO
        self._initializeSumo()
        
        # Get traffic light IDs and initialize agent information
        self.traffic_lights = traci.trafficlight.getIDList()
        self.num_agents = len(self.traffic_lights)
        
        # Store traffic light programs and phases
        self.programs = self._getTrafficLightPrograms()
        
        # Initialize observation and action spaces
        self._initializeSpaces()
        
        # State tracking variables
        self.traffic_light_states = defaultdict(dict)
        self.vehicle_states = defaultdict(dict)
        self.current_phases = defaultdict(int)
        self.time_since_last_phase_change = defaultdict(int)
        self.already_yellow = defaultdict(bool)
        
        # Vehicle tracking for rewards
        self.vehicles_seen = set()
        self.vehicles_completed = set()
        self.last_vehicle_counts = defaultdict(int)
        
        # Episode step counter
        self.episode_step = 0
        
    def _initializeSumo(self):
        """Initialize SUMO simulation"""
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        
        sumo_cmd = [
            sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "10000",
            "--time-to-teleport", "300",  # Match baseline teleport time
            "--no-warnings", "true",
            "--step-length", str(self.delta_time),
            "--begin", "0"
        ]
        
        traci.start(sumo_cmd)
        
    def _initializeSpaces(self):
        """Initialize observation and action spaces for each agent"""
        # Base state size: 4 lanes × 4 features + 1 phase duration
        base_state_size = 4 * 4 + 1  # 17 features
        
        # Add neighbor features if enabled
        if self.config['multi_agent']['observation_radius'] > 0:
            num_neighbors = 4
            features_per_neighbor = 3
            neighbor_features = num_neighbors * features_per_neighbor
            self.observation_space_size = base_state_size + neighbor_features  # 17 + 12 = 29 features
        else:
            self.observation_space_size = base_state_size  # 17 features
            
        self.action_space_size = self.config['env']['num_actions']
        
    def _getTrafficLightPrograms(self) -> Dict:
        """Get traffic light programs for all intersections"""
        programs = {}
        for tl_id in self.traffic_lights:
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            print(f"Traffic Light {tl_id} has {len(logic.phases)} phases:")
            for i, phase in enumerate(logic.phases):
                print(f"Phase {i}: duration={phase.duration}, state={phase.state}")
            programs[tl_id] = {
                'phases': logic.phases,
                'current_phase_index': 0,
                'num_green_phases': len([p for p in logic.phases if 'y' not in p.state.lower()])
            }
        return programs
        
    def _computeState(self, tl_id: str) -> np.ndarray:
        """
        Compute state for a traffic light
        
        Args:
            tl_id: Traffic light ID
            
        Returns:
            State array containing traffic information
        """
        # Get controlled lanes
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        unique_lanes = list(set(controlled_lanes))
        
        # Initialize feature lists
        queue_lengths = []
        waiting_times = []
        speeds = []
        densities = []
        
        # Collect lane features
        for lane in unique_lanes:
            # Queue length
            queue_length = traci.lane.getLastStepHaltingNumber(lane)
            queue_lengths.append(queue_length)
            
            # Waiting time
            waiting_time = traci.lane.getWaitingTime(lane)
            waiting_times.append(waiting_time)
            
            # Average speed
            mean_speed = traci.lane.getLastStepMeanSpeed(lane)
            speeds.append(mean_speed)
            
            # Vehicle density
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane)
            lane_length = traci.lane.getLength(lane)
            density = vehicle_count / lane_length if lane_length > 0 else 0
            densities.append(density)
        
        # Normalize to fixed size (4 lanes)
        max_lanes = 4
        queue_lengths = queue_lengths[:max_lanes] + [0] * (max_lanes - len(queue_lengths))
        waiting_times = waiting_times[:max_lanes] + [0] * (max_lanes - len(waiting_times))
        speeds = speeds[:max_lanes] + [0] * (max_lanes - len(speeds))
        densities = densities[:max_lanes] + [0] * (max_lanes - len(densities))
        
        # Phase information
        phase_duration = self.time_since_last_phase_change[tl_id]
        normalized_phase_duration = min(phase_duration / 60.0, 1.0)  # Normalize to [0,1], cap at 60s
        
        # Create state vector
        state = np.array(
            queue_lengths +    # 4 values
            waiting_times +    # 4 values
            speeds +          # 4 values
            densities +       # 4 values
            [normalized_phase_duration]  # 1 value
        )
        
        # Add neighbor information if enabled
        if self.config['multi_agent']['observation_radius'] > 0:
            neighbor_info = self._getNeighborInfo(tl_id)
            state = np.concatenate([state, neighbor_info])
            
        return state.astype(np.float32)
        
    def _getNeighborInfo(self, tl_id: str) -> List[float]:
        """
        Get state information from neighboring intersections
        
        Args:
            tl_id: Traffic light ID
            
        Returns:
            List of neighbor state values
        """
        neighbor_info = []
        max_neighbors = 4  # Fixed number of closest neighbors to consider
        
        # If no observation radius specified, return empty list
        if self.config['multi_agent']['observation_radius'] <= 0:
            return [0.0] * (max_neighbors * self.config['multi_agent']['neighbor_features'])
            
        try:
            # Get neighboring traffic lights
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            neighbor_tls = set()
            
            # Find connected traffic lights through lanes
            for lane in controlled_lanes:
                # Get connected lanes
                links = traci.lane.getLinks(lane)
                for link in links:
                    to_lane = link[0]  # Get target lane
                    # Find traffic light controlling this lane
                    try:
                        tls = traci.lane.getControllingTLS(to_lane)
                        for tl in tls:
                            if tl[0] != tl_id:  # Don't include self
                                neighbor_tls.add(tl[0])
                    except:
                        continue
            
            # Sort neighbors by distance (using placeholder distance for now)
            neighbors_list = list(neighbor_tls)[:max_neighbors]
            
            # Get state information for each neighbor
            for neighbor_id in neighbors_list:
                try:
                    # Get basic neighbor state
                    phase = self.current_phases.get(neighbor_id, 0)
                    time_since_change = self.time_since_last_phase_change.get(neighbor_id, 0)
                    
                    # Add normalized neighbor information
                    neighbor_info.extend([
                        float(phase) / 4.0,  # Normalize phase
                        min(float(time_since_change) / 60.0, 1.0),  # Normalize time (cap at 60s)
                        1.0  # Placeholder for distance (normalized)
                    ])
                except:
                    # If error getting neighbor info, add zero padding
                    neighbor_info.extend([0.0] * self.config['multi_agent']['neighbor_features'])
            
            # Pad if we have fewer than max_neighbors
            while len(neighbor_info) < max_neighbors * self.config['multi_agent']['neighbor_features']:
                neighbor_info.extend([0.0] * self.config['multi_agent']['neighbor_features'])
                
        except Exception as e:
            print(f"Error getting neighbor info for {tl_id}: {str(e)}")
            # Return zero-padded neighbor info for max_neighbors
            return [0.0] * (max_neighbors * self.config['multi_agent']['neighbor_features'])
            
        return neighbor_info
        
    def _computeReward(self, tl_id: str) -> float:
        """
        Compute reward for a traffic light agent
        
        Args:
            tl_id: Traffic light ID
            
        Returns:
            Reward value combining various metrics
        """
        reward = 0
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        
        # Get all vehicles in controlled lanes
        current_vehicles = set()
        for lane in controlled_lanes:
            current_vehicles.update(traci.lane.getLastStepVehicleIDs(lane))
        
        if not current_vehicles:
            waiting_time = 0
            throughput = 0
            avg_speed = 0
        else:
            # Calculate metrics exactly like baseline
            waiting_times = np.array([traci.vehicle.getWaitingTime(vid) for vid in current_vehicles])
            speeds = np.array([traci.vehicle.getSpeed(vid) for vid in current_vehicles])
            distances = np.array([traci.vehicle.getDistance(vid) for vid in current_vehicles])
            
            waiting_time = np.sum(waiting_times)  # Keep in milliseconds like baseline
            throughput = np.sum(distances > 0)
            avg_speed = np.mean(speeds) * 3.6 if len(speeds) > 0 else 0  # Convert to km/h
        
        # Queue length penalty (normalized by number of lanes)
        queue_length = sum(traci.lane.getLastStepHaltingNumber(lane) 
                         for lane in controlled_lanes)
        reward += self.config['rewards']['queue_length'] * (queue_length / max(1, len(controlled_lanes)))
        
        # Waiting time penalty (in milliseconds)
        reward += self.config['rewards']['waiting_time'] * waiting_time
        
        # Speed reward (in km/h, normalized)
        reward += self.config['rewards']['speed'] * (avg_speed / 50.0)  # Normalize by typical urban speed limit
        
        # Emergency vehicle priority
        emergency_waiting = False
        for lane in controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for vehicle in vehicles:
                if traci.vehicle.getVehicleClass(vehicle) == "emergency":
                    if traci.vehicle.getWaitingTime(vehicle) > 0:
                        emergency_waiting = True
                        break
        if emergency_waiting:
            reward += self.config['rewards']['emergency']
        
        # Throughput reward based on completed trips and moving vehicles
        reward += self.config['rewards']['throughput'] * throughput
        
        return reward
        
    def _applyAction(self, tl_id: str, action: int):
        """
        Apply selected action to traffic light
        
        Args:
            tl_id: Traffic light ID
            action: Selected action index
        """
        num_phases = len(self.programs[tl_id]['phases'])
        if self.already_yellow[tl_id]:
            # Finish the yellow phase and switch to the target green phase
            green_phase = action % (num_phases // 2) * 2  # Map action to available green phases
            self.current_phases[tl_id] = green_phase
            traci.trafficlight.setPhase(tl_id, green_phase)
            self.already_yellow[tl_id] = False
            self.time_since_last_phase_change[tl_id] = 0
        else:
            if self.time_since_last_phase_change[tl_id] >= self.min_green:
                # Set yellow phase before changing
                current_phase = self.current_phases[tl_id]
                yellow_phase = current_phase + 1  # Yellow phases follow green phases
                if yellow_phase < num_phases:
                    traci.trafficlight.setPhase(tl_id, yellow_phase)
                    self.already_yellow[tl_id] = True
                    self.time_since_last_phase_change[tl_id] = 0
                
    def reset(self, seed: int = None) -> tuple[dict, dict]:
        """
        Reset environment and return initial state
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (observations, info)
        """
        # Close existing SUMO instance
        if traci.isLoaded():
            traci.close()
            
        # Start new SUMO instance
        self._initializeSumo()
        
        # Reset state variables
        self.traffic_light_states.clear()
        self.vehicle_states.clear()
        self.current_phases.clear()
        self.time_since_last_phase_change.clear()
        self.already_yellow.clear()
        
        # Reset vehicle tracking
        self.vehicles_seen = set()
        self.vehicles_completed = set()
        
        # Reset episode step counter
        self.episode_step = 0
        
        # Get initial states
        states = {}
        for tl_id in self.traffic_lights:
            states[tl_id] = self._computeState(tl_id)
            self.current_phases[tl_id] = 0
            self.time_since_last_phase_change[tl_id] = 0
            self.already_yellow[tl_id] = False
            
        return states, {}
        
    def step(self, actions: dict) -> tuple[dict, dict, dict, dict]:
        """
        Execute actions and return results
        
        Args:
            actions: Dictionary of actions for each agent
            
        Returns:
            Tuple of (next_states, rewards, dones, info)
        """
        # Apply actions for each agent
        for tl_id, action in actions.items():
            self._applyAction(tl_id, action)
            
        # Simulate one step
        traci.simulationStep()
        
        # Update time since last phase change
        for tl_id in self.traffic_lights:
            if not self.already_yellow[tl_id]:
                self.time_since_last_phase_change[tl_id] += self.delta_time
                
        # Increment episode step counter
        self.episode_step += 1
        
        # Compute next states and rewards
        next_states = {}
        rewards = {}
        info = defaultdict(dict)
        
        for tl_id in self.traffic_lights:
            next_states[tl_id] = self._computeState(tl_id)
            rewards[tl_id] = self._computeReward(tl_id)
            
            # Store additional information
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            info[tl_id]['queue_length'] = sum(traci.lane.getLastStepHaltingNumber(lane)
                                            for lane in controlled_lanes)
            info[tl_id]['waiting_time'] = sum(traci.lane.getWaitingTime(lane)
                                           for lane in controlled_lanes)
            info[tl_id]['average_speed'] = np.mean([traci.lane.getLastStepMeanSpeed(lane)
                                                  for lane in controlled_lanes])
            
        # Check if simulation should end
        done = traci.simulation.getMinExpectedNumber() <= 0
        dones = {tl_id: done for tl_id in self.traffic_lights}
        
        return next_states, rewards, dones, info
        
    def close(self):
        """Close SUMO simulation"""
        if traci.isLoaded():
            traci.close()
            
    def render(self, mode: str = 'human'):
        """
        Render environment
        For SUMO, this is only available when using the GUI
        """
        pass
        
    @property
    def agent_ids(self) -> List[str]:
        """Get list of agent IDs"""
        return self.traffic_lights