# src/environment/multi_agent_sumo_env_casf.py

import os
import sys
import numpy as np
import traci
import sumolib
from gym import spaces
import logging
from src.utils.logger import get_logger
import time
import contextlib

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
            'reward_weights',
            'files'  # Ensure 'files' section exists
        ]
        for key in required_keys:
            if key not in config:
                self.logger.error(f"Missing required configuration key: '{key}'")
                raise KeyError(f"Missing required configuration key: '{key}'")
        
        # Ensure 'net_file' and 'route_file' exist under 'files'
        if 'net_file' not in config['files']:
            self.logger.error("Missing 'net_file' in 'files' section of configuration.")
            raise KeyError("Missing 'net_file' in 'files' section of configuration.")
        if 'route_file' not in config['files']:
            self.logger.error("Missing 'route_file' in 'files' section of configuration.")
            raise KeyError("Missing 'route_file' in 'files' section of configuration.")
        
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
        
        # Initialize SUMO
        self.init_sumo()
        
        # Get controllable traffic light IDs (those with program logics and at least one phase)
        controllable_tl_ids = []
        for tl_id in traci.trafficlight.getIDList():
            program_logics = traci.trafficlight.getAllProgramLogics(tl_id)
            if not program_logics:
                self.logger.warning(f"Traffic light {tl_id} lacks program logics and will be excluded from control.")
                continue  # Skip uncontrollable traffic lights
            if len(program_logics[0].phases) == 0:
                self.logger.warning(f"No phases defined for traffic light {tl_id}. Skipping phase setting.")
                continue  # Skip traffic lights with no phases
            controllable_tl_ids.append(tl_id)
        
        if len(controllable_tl_ids) == 0:
            self.logger.error("No controllable traffic lights found in the SUMO simulation.")
            raise ValueError("No controllable traffic lights found in the SUMO simulation.")
        
        self.agent_ids = controllable_tl_ids  # Update agent_ids to controllable ones
        self.logger.info(f"Controllable Traffic Lights: {self.agent_ids}")
        
        # Sample state from the first controllable traffic light
        sample_state = self.get_state(self.agent_ids[0])
        self.state_size = len(sample_state)

        self.init_spaces()
        self.n_agents = len(self.agent_ids)
        self.neighbor_map = self.build_neighbor_map()

        self.logger.info(f"Environment initialized with {self.n_agents} agents")

    def init_sumo(self):
        """Initialize the SUMO simulation."""
        sumo_home = os.getenv('SUMO_HOME')
        if sumo_home:
            sumo_binary = os.path.join(sumo_home, "bin", "sumo")
        else:
            self.logger.error("SUMO_HOME environment variable not set.")
            raise EnvironmentError("SUMO_HOME environment variable not set.")
        
        sumo_config = self.config['files']
        
        sumo_cmd = [sumo_binary]
        
        if 'sumocfg' in sumo_config and sumo_config['sumocfg']:
            sumo_cmd += [
                "-c", sumo_config['sumocfg'],
                "--start",
                "--quit-on-end"
            ]
            self.logger.info("Using 'sumocfg' for SUMO configuration.")
        else:
            if 'net_file' not in sumo_config or 'route_file' not in sumo_config:
                self.logger.error("Either 'sumocfg' or both 'net_file' and 'route_file' must be provided in 'files' section.")
                raise KeyError("Either 'sumocfg' or both 'net_file' and 'route_file' must be provided in 'files' section.")
            sumo_cmd += [
                "-n", sumo_config['net_file'],
                "-r", sumo_config['route_file'],
                "--start",
                "--quit-on-end"
            ]
            self.logger.info("Using 'net_file' and 'route_file' for SUMO configuration.")

        # Add options to suppress warnings and logs
        sumo_cmd += [
            "--no-warnings",
            "--no-step-log",
            "--error-log", os.devnull,
            "--message-log", os.devnull
        ]

        self.sumo_cmd = sumo_cmd

        # Start SUMO with stdout and stderr redirected to suppress output
        try:
            with contextlib.redirect_stdout(open(os.devnull, 'w')), contextlib.redirect_stderr(open(os.devnull, 'w')):
                traci.start(self.sumo_cmd)
            self.logger.info("SUMO simulation started successfully.")
        except Exception as e:
            self.logger.error(f"Failed to start SUMO simulation: {e}")
            raise e

    def build_neighbor_map(self):
        self.logger.info("Building neighbor map")
        net = sumolib.net.readNet(self.config['files']['net_file'])
        neighbor_map = {}
        tls_nodes = {}

        # Get all controllable traffic light nodes
        for tl_id in self.agent_ids:
            links = traci.trafficlight.getControlledLinks(tl_id)
            if links:
                from_lane = links[0][0][0]  # Get first controlled lane
                from_edge = traci.lane.getEdgeID(from_lane)
                node = net.getEdge(from_edge).getFromNode()
                tls_nodes[tl_id] = node
            else:
                self.logger.warning(f"Traffic light {tl_id} has no controlled links and will be excluded from neighbor map.")
        
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

    def init_spaces(self):
        """Initialize action and observation spaces."""
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.config['action_dim'],),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config['observation_dim'],),
            dtype=np.float32
        )
        self.logger.info("Action and observation spaces initialized.")

    def get_state(self, tl_id):
        lane_states = []
        for lane in traci.trafficlight.getControlledLanes(tl_id):
            waiting_time = traci.lane.getWaitingTime(lane)
            queue_length = traci.lane.getLastStepHaltingNumber(lane)
            throughput = traci.lane.getLastStepVehicleNumber(lane)
            lane_states.extend([waiting_time, queue_length, throughput])

        phase = traci.trafficlight.getPhase(tl_id)
        program_logics = traci.trafficlight.getAllProgramLogics(tl_id)
        if not program_logics or len(program_logics[0].phases) == 0:
            num_phases = 0
        else:
            num_phases = len(program_logics[0].phases)
        phase_one_hot = np.zeros(num_phases)  # Dynamic based on actual number of phases
        if num_phases > 0:
            phase_one_hot[phase % num_phases] = 1

        # Combine all features into fixed-size state
        state = np.array(lane_states + list(phase_one_hot))

        # Pad or truncate to fixed size
        target_size = self.config['observation_dim']  # Set fixed size based on max possible state
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
        while len(neighbor_states) < self.config.get('max_neighbors', 4):
            neighbor_states.append(np.zeros(self.config['observation_dim'], dtype=np.float32))  # Same fixed size

        self.logger.debug(f"Neighbor states for {tl_id}: {neighbor_states}")
        return np.array(neighbor_states)

    def compute_global_reward(self):
        """
        Compute the global reward based on overall grid performance.
        For example, you can use negative average waiting time and negative average queue length plus total throughput.
        """
        total_waiting_time = 0
        total_queue_length = 0
        total_throughput = 0
        num_lanes = 0

        for tl_id in self.agent_ids:
            for lane_id in traci.trafficlight.getControlledLanes(tl_id):
                total_waiting_time += traci.lane.getWaitingTime(lane_id)
                total_queue_length += traci.lane.getLastStepHaltingNumber(lane_id)
                total_throughput += traci.lane.getLastStepVehicleNumber(lane_id)
                num_lanes += 1

        if num_lanes == 0:
            self.logger.warning("No lanes controlled by traffic lights. Global reward set to 0.")
            avg_waiting_time = 0
            avg_queue_length = 0
        else:
            avg_waiting_time = total_waiting_time / num_lanes
            avg_queue_length = total_queue_length / num_lanes

        # Get reward weights from config
        reward_weights = self.config.get('reward_weights', {
            'waiting_time': 1.0,
            'queue_length': 1.0,
            'throughput': 1.0
        })

        # Compute global reward: similar to individual rewards
        global_reward = (
            -reward_weights['waiting_time'] * avg_waiting_time +
            -reward_weights['queue_length'] * avg_queue_length +
            reward_weights['throughput'] * total_throughput
        )

        self.logger.debug(f"Global Reward: {global_reward} (Avg Waiting Time: {avg_waiting_time}, "
                          f"Avg Queue Length: {avg_queue_length}, Total Throughput: {total_throughput})")

        return global_reward

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

        self.logger.debug(f"Reward for {tl_id}: {reward} (waiting_time: {waiting_time}, "
                          f"queue_length: {queue_length}, throughput: {throughput})")
        return reward

    def step(self, actions):
        self.logger.debug("Applying actions to all traffic lights")
        # Apply actions
        for tl_id, action in zip(self.agent_ids, actions):
            self.apply_action(tl_id, action)

        # Simulate one step
        traci.simulationStep()
        self.logger.debug("Simulation step completed")

        # Get states and rewards using list comprehensions
        try:
            states = [self.get_state(tl_id) for tl_id in self.agent_ids]
            neighbor_states = [self.get_neighbor_states(tl_id) for tl_id in self.agent_ids]
            local_rewards = [self.compute_reward(tl_id) for tl_id in self.agent_ids]
        except Exception as e:
            self.logger.error(f"Error while collecting states or rewards: {e}")
            raise e

        # Compute global reward
        global_reward = self.compute_global_reward()

        # Check termination
        current_time = traci.simulation.getTime()
        done = current_time >= self.config['max_episode_steps']
        dones = [done] * self.n_agents

        info = {
            'time': current_time,
            'total_waiting_time': sum([traci.lane.getWaitingTime(lane) for tl_id in self.agent_ids for lane in traci.trafficlight.getControlledLanes(tl_id)]),
            'avg_waiting_time': np.mean([traci.lane.getWaitingTime(lane) for tl_id in self.agent_ids for lane in traci.trafficlight.getControlledLanes(tl_id)]),
            'avg_queue_length': np.mean([traci.lane.getLastStepHaltingNumber(lane) for tl_id in self.agent_ids for lane in traci.trafficlight.getControlledLanes(tl_id)]),
            'throughput': sum([traci.lane.getLastStepVehicleNumber(lane) for tl_id in self.agent_ids for lane in traci.trafficlight.getControlledLanes(tl_id)]),
            'global_reward': global_reward  # Add global reward to info
        }

        self.logger.debug(f"Step info: {info}")
        return states, neighbor_states, local_rewards, dones, info

    def apply_action(self, tl_id, actions):
        """
        Applies a multi-dimensional action to the traffic light.
        actions: array-like with shape (action_dim,)
        """
        # Ensure actions is a numpy array
        actions = np.array(actions)
        
        # Rescale first action from [-1, 1] to [0, num_phases - 1]
        normalized_phase = (actions[0] + 1) / 2
        program_logics = traci.trafficlight.getAllProgramLogics(tl_id)
        if not program_logics:
            self.logger.warning(f"No program logics found for traffic light {tl_id}. Skipping phase setting.")
            return
        num_phases = len(program_logics[0].phases)
        if num_phases == 0:
            self.logger.warning(f"No phases defined for traffic light {tl_id}. Skipping phase setting.")
            return
        phase = int(normalized_phase * (num_phases - 1))
        phase = min(max(phase, 0), num_phases - 1)  # Ensure phase is within valid range
        
        # Rescale second action from [-1, 1] to [5, max_phase_duration]
        normalized_duration = (actions[1] + 1) / 2
        duration = int(normalized_duration * (self.config['max_phase_duration'] - 5)) + 5  # Minimum duration of 5 seconds
        
        self.logger.debug(f"Applying actions for {tl_id}: phase {phase}, duration {duration}")
        
        # Set the new phase and duration
        traci.trafficlight.setPhase(tl_id, phase)
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

        self.logger.debug(f"Reward for {tl_id}: {reward} (waiting_time: {waiting_time}, "
                          f"queue_length: {queue_length}, throughput: {throughput})")
        return reward

    def compute_global_reward(self):
        """
        Compute the global reward based on overall grid performance.
        For example, you can use negative average waiting time and negative average queue length plus total throughput.
        """
        total_waiting_time = 0
        total_queue_length = 0
        total_throughput = 0
        num_lanes = 0

        for tl_id in self.agent_ids:
            for lane_id in traci.trafficlight.getControlledLanes(tl_id):
                total_waiting_time += traci.lane.getWaitingTime(lane_id)
                total_queue_length += traci.lane.getLastStepHaltingNumber(lane_id)
                total_throughput += traci.lane.getLastStepVehicleNumber(lane_id)
                num_lanes += 1

        if num_lanes == 0:
            self.logger.warning("No lanes controlled by traffic lights. Global reward set to 0.")
            avg_waiting_time = 0
            avg_queue_length = 0
        else:
            avg_waiting_time = total_waiting_time / num_lanes
            avg_queue_length = total_queue_length / num_lanes

        # Get reward weights from config
        reward_weights = self.config.get('reward_weights', {
            'waiting_time': 1.0,
            'queue_length': 1.0,
            'throughput': 1.0
        })

        # Compute global reward: similar to individual rewards
        global_reward = (
            -reward_weights['waiting_time'] * avg_waiting_time +
            -reward_weights['queue_length'] * avg_queue_length +
            reward_weights['throughput'] * total_throughput
        )

        self.logger.debug(f"Global Reward: {global_reward} (Avg Waiting Time: {avg_waiting_time}, "
                          f"Avg Queue Length: {avg_queue_length}, Total Throughput: {total_throughput})")

        return global_reward

    def reset(self):
        self.logger.info("Resetting environment")
        try:
            traci.close()
        except Exception as e:
            self.logger.warning(f"Error while closing SUMO during reset: {e}")
        
        # Start SUMO with stderr redirected to suppress warnings
        try:
            with contextlib.redirect_stderr(open(os.devnull, 'w')):
                traci.start(self.sumo_cmd)
            self.logger.info("SUMO simulation started successfully on reset.")
        except Exception as e:
            self.logger.error(f"Failed to start SUMO simulation on reset: {e}")
            raise e

        try:
            states = [self.get_state(tl_id) for tl_id in self.agent_ids]
            neighbor_states = [self.get_neighbor_states(tl_id) for tl_id in self.agent_ids]
            local_rewards = [self.compute_reward(tl_id) for tl_id in self.agent_ids]
        except Exception as e:
            self.logger.error(f"Error while collecting states or neighbor states during reset: {e}")
            raise e

        # Compute global reward
        global_reward = self.compute_global_reward()

        # Ensure the lists have the correct length
        if len(states) != self.n_agents:
            self.logger.error(f"Mismatch in number of states during reset: Expected {self.n_agents}, got {len(states)}")
            raise IndexError(f"Mismatch in number of states during reset: Expected {self.n_agents}, got {len(states)}")
        if len(neighbor_states) != self.n_agents:
            self.logger.error(f"Mismatch in number of neighbor_states during reset: Expected {self.n_agents}, got {len(neighbor_states)}")
            raise IndexError(f"Mismatch in number of neighbor_states during reset: Expected {self.n_agents}, got {len(neighbor_states)}")
        if len(local_rewards) != self.n_agents:
            self.logger.error(f"Mismatch in number of local_rewards during reset: Expected {self.n_agents}, got {len(local_rewards)}")
            raise IndexError(f"Mismatch in number of local_rewards during reset: Expected {self.n_agents}, got {len(local_rewards)}")

        info = {
            'global_reward': global_reward
        }

        self.logger.info("Environment reset completed")
        return states, neighbor_states, local_rewards, info

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
                'waiting_time': traci.vehicle.getWaitingTime(vid),
                'speed': traci.vehicle.getSpeed(vid),  # m/s
                'route_id': traci.vehicle.getRouteID(vid)
            }
            vehicle_data.append(vehicle_info)
        
        self.logger.debug(f"Collected data for {len(vehicle_data)} vehicles")
        return vehicle_data
