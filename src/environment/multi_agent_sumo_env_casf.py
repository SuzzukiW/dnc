# src/environment/multi_agent_sumo_env_casf.py

import os
import sys
import numpy as np
import traci
import sumolib
from gym import spaces
import logging  # Import logging module
from src.utils.logger import get_logger  # Import the logger utility
import time  # Import time module

class MACSFEnvironment:
    def __init__(self, config):
        # Initialize logger
        self.logger = get_logger('MACSFEnvironment', level=logging.INFO)
        self.logger.info("Initializing MACSFEnvironment")
        
        # Validate required configuration keys
        required_keys = [
            'observation_dim',
            'action_dim',
            'max_episode_steps',
            'max_neighbors',
            'communication_range',
            'max_phase_duration',
            'reward_weights'
        ]
        for key in required_keys:
            if key not in config:
                self.logger.error(f"Missing required configuration key: '{key}'")
                raise KeyError(f"Missing required configuration key: '{key}'")
        
        self.config = {
            'max_neighbors': 4,
            'communication_range': 1000,
            'reward_weights': {
                'waiting_time': 1.0,
                'queue_length': 1.0,
                'throughput': 1.0
            },
            **config
        }
        self.init_sumo()

        # Get initial state dimensions
        sample_state = self.get_state(traci.trafficlight.getIDList()[0])
        self.state_size = len(sample_state)

        self.init_spaces()
        self.n_agents = len(traci.trafficlight.getIDList())
        self.neighbor_map = self.build_neighbor_map()

        self.logger.info(f"Environment initialized with {self.n_agents} agents")

    def init_sumo(self):
        self.logger.info("Initializing SUMO simulation")
        if 'SUMO_HOME' not in os.environ:
            self.logger.error('SUMO_HOME not set')
            raise Exception('SUMO_HOME not set')
        
        sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
        self.sumo_cmd = [
            sumo_binary,
            '--net-file', os.path.join('baseline', 'osm.net.xml.gz'),
            '--route-files', os.path.join('baseline', 'osm.passenger.trips.xml'),
            '--no-step-log', 'true',
            '--no-warnings', 'true',
            '--ignore-junction-blocker', '60',
            '--time-to-teleport', '300',
            '--end', '200'  # Set simulation end time to match max_episode_steps
        ]
        traci.start(self.sumo_cmd)
        self.logger.info("SUMO simulation started")

    def init_spaces(self):
        self.logger.info("Initializing observation and action spaces")
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(28,),  # Updated to match the fixed state size
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.full(self.config['action_dim'], -1.0),
            high=np.full(self.config['action_dim'], 1.0),
            dtype=np.float32
        )
        self.logger.info(f"Observation space: {self.observation_space}")
        self.logger.info(f"Action space: {self.action_space}")

    def build_neighbor_map(self):
        self.logger.info("Building neighbor map")
        net = sumolib.net.readNet(os.path.join('baseline', 'osm.net.xml.gz'))
        neighbor_map = {}
        tls_nodes = {}

        # Get all traffic light nodes
        for tl_id in traci.trafficlight.getIDList():
            links = traci.trafficlight.getControlledLinks(tl_id)
            if links:
                from_lane = links[0][0][0]  # Get first controlled lane
                from_edge = traci.lane.getEdgeID(from_lane)
                node = net.getEdge(from_edge).getFromNode()
                tls_nodes[tl_id] = node

        # Build neighbor map
        for tl_id, node in tls_nodes.items():
            neighbors = []
            for other_tl, other_node in tls_nodes.items():
                if other_tl != tl_id:
                    dist = np.sqrt(
                        (node.getCoord()[0] - other_node.getCoord()[0])**2 + 
                        (node.getCoord()[1] - other_node.getCoord()[1])**2
                    )
                    if dist <= self.config.get('communication_range', 1000):
                        neighbors.append(other_tl)

            neighbor_map[tl_id] = neighbors[:self.config.get('max_neighbors', 4)]
            self.logger.debug(f"Traffic Light {tl_id} neighbors: {neighbor_map[tl_id]}")

        self.logger.info("Neighbor map built successfully")
        return neighbor_map

    def get_state(self, tl_id):
        lanes = traci.trafficlight.getControlledLanes(tl_id)

        # Fixed size representation per lane
        lane_states = []
        for lane in lanes:
            lane_state = [
                traci.lane.getLastStepHaltingNumber(lane),
                traci.lane.getWaitingTime(lane),
                traci.lane.getLastStepVehicleNumber(lane),
                traci.lane.getLastStepMeanSpeed(lane)
            ]
            lane_states.extend(lane_state)

        # Current phase info
        phase = traci.trafficlight.getPhase(tl_id)
        phase_one_hot = np.zeros(4)  # Assume 4 possible phases
        phase_one_hot[phase % 4] = 1

        # Combine all features into fixed-size state
        state = np.array(lane_states + list(phase_one_hot))

        # Pad or truncate to fixed size
        target_size = 28  # Set fixed size based on max possible state
        if len(state) < target_size:
            state = np.pad(state, (0, target_size - len(state)))
        else:
            state = state[:target_size]

        self.logger.debug(f"State for {tl_id}: {state}")
        return state

    def normalize_state(self, state):
        # Placeholder for state normalization if needed
        return state

    def get_neighbor_states(self, tl_id):
        neighbor_states = []
        for neighbor_id in self.neighbor_map[tl_id]:
            neighbor_state = self.get_state(neighbor_id)
            neighbor_states.append(neighbor_state)

        # Pad with zeros for missing neighbors
        while len(neighbor_states) < self.config['max_neighbors']:
            neighbor_states.append(np.zeros(28))  # Same fixed size

        self.logger.debug(f"Neighbor states for {tl_id}: {neighbor_states}")
        return np.array(neighbor_states)

    def step(self, actions):
        self.logger.debug("Applying actions to all traffic lights")
        # Apply actions
        for tl_id, action in zip(traci.trafficlight.getIDList(), actions):
            self.apply_action(tl_id, action)

        # Simulate one step
        traci.simulationStep()
        self.logger.debug("Simulation step completed")

        # Get states and rewards
        states = []
        neighbor_states = []
        rewards = []

        for tl_id in traci.trafficlight.getIDList():
            states.append(self.get_state(tl_id))
            neighbor_states.append(self.get_neighbor_states(tl_id))
            rewards.append(self.compute_reward(tl_id))

        # Check termination
        current_time = traci.simulation.getTime()
        done = current_time >= self.config['max_episode_steps']
        dones = [done] * self.n_agents

        info = {
            'time': current_time,
            'total_waiting_time': sum([traci.lane.getWaitingTime(lane) for tl_id in traci.trafficlight.getIDList() for lane in traci.trafficlight.getControlledLanes(tl_id)]),
            'avg_waiting_time': np.mean([traci.lane.getWaitingTime(lane) for tl_id in traci.trafficlight.getIDList() for lane in traci.trafficlight.getControlledLanes(tl_id)]),
            'avg_queue_length': np.mean([traci.lane.getLastStepHaltingNumber(lane) for tl_id in traci.trafficlight.getIDList() for lane in traci.trafficlight.getControlledLanes(tl_id)]),
            'throughput': sum([traci.lane.getLastStepVehicleNumber(lane) for tl_id in traci.trafficlight.getIDList() for lane in traci.trafficlight.getControlledLanes(tl_id)])
        }

        self.logger.debug(f"Step info: {info}")
        return states, neighbor_states, rewards, dones, info

    def apply_action(self, tl_id, action):
        # Ensure action is a scalar
        if isinstance(action, np.ndarray):
            action = action.item()
        
        # Rescale action from [-1, 1] to [0, 1]
        normalized_action = (action + 1) / 2
        
        # Convert normalized action to phase duration
        duration = int(normalized_action * self.config['max_phase_duration'])
        
        # Ensure minimum duration of 5 seconds
        duration = max(duration, 5)
        
        self.logger.debug(f"Applying action for {tl_id}: {action} (normalized: {normalized_action}), duration set to {duration}")
        
        # Set phase duration
        traci.trafficlight.setPhaseDuration(tl_id, duration)

    def compute_reward(self, tl_id):
        waiting_time = 0
        queue_length = 0
        throughput = 0

        for lane_id in traci.trafficlight.getControlledLanes(tl_id):
            waiting_time += traci.lane.getWaitingTime(lane_id)
            queue_length += traci.lane.getLastStepHaltingNumber(lane_id)
            throughput += traci.lane.getLastStepVehicleNumber(lane_id)

        # Get reward weights from config
        reward_weights = self.config.get('reward_weights', {
            'waiting_time': 1.0,
            'queue_length': 1.0,
            'throughput': 1.0
        })

        # Compute reward: penalize waiting time and queue length, reward throughput
        reward = (
            -reward_weights['waiting_time'] * waiting_time +
            -reward_weights['queue_length'] * queue_length +
            reward_weights['throughput'] * throughput
        )

        self.logger.debug(f"Reward for {tl_id}: {reward} (waiting_time: {waiting_time}, queue_length: {queue_length}, throughput: {throughput})")
        return reward

    def reset(self):
        self.logger.info("Resetting environment")
        traci.close()
        traci.start(self.sumo_cmd)

        states = []
        neighbor_states = []
        for tl_id in traci.trafficlight.getIDList():
            states.append(self.get_state(tl_id))
            neighbor_states.append(self.get_neighbor_states(tl_id))

        self.logger.info("Environment reset completed")
        return states, neighbor_states

    def close(self):
        """Cleanly close the environment and SUMO connection."""
        self.logger.info("Closing environment")
        try:
            if traci.isLoaded():
                traci.close()
                self.logger.debug("SUMO connection closed successfully")
        except Exception as e:
            self.logger.warning(f"Error while closing SUMO: {e}")
        
        # Wait a bit to ensure socket is fully closed
        time.sleep(0.1)

    def get_vehicle_data(self):
        """
        Collect vehicle performance metrics from the environment.
        Returns a list of dictionaries, where each dictionary contains metrics for a single vehicle:
        - vehicle_id: ID of the vehicle
        - waiting_time: Waiting time in seconds
        - speed: Speed in m/s
        - route_id: Route ID of the vehicle
        """
        # Get current vehicles
        current_vehicles = list(traci.vehicle.getIDList())
        
        if not current_vehicles:
            return []
        
        # Collect data for each vehicle
        vehicle_data = []
        for vid in current_vehicles:
            vehicle_info = {
                'vehicle_id': vid,
                'waiting_time': traci.vehicle.getWaitingTime(vid) / 1000.0,  # Convert ms to s
                'speed': traci.vehicle.getSpeed(vid),  # m/s
                'route_id': traci.vehicle.getRouteID(vid)
            }
            vehicle_data.append(vehicle_info)
        
        self.logger.debug(f"Collected data for {len(vehicle_data)} vehicles")
        return vehicle_data
