# src/environment/multi_agent_sumo_env_independent.py

import os
import sys
import numpy as np
from pathlib import Path
import traci
import sumolib

class MultiAgentSumoIndependentEnvironment:
    """
    Multi-agent environment for SUMO traffic light control.
    Each traffic light is treated as an independent agent.
    """
    
    def __init__(
        self,
        net_file,
        route_file,
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
        sumo_cmd = []
        
        # Binary selection
        if self.use_gui:
            sumo_cmd.extend(['sumo-gui'])
        else:
            sumo_cmd.extend(['sumo'])

        # Basic options
        sumo_cmd.extend([
            '-n', self.net_file,
            '-r', self.route_file,
            '--max-depart-delay', '0',
            '--waiting-time-memory', '10000',
            '--time-to-teleport', '-1',
        ])

        # Optional flags
        if not sumo_warnings:
            sumo_cmd.extend(['--no-warnings'])
        
        if sumo_seed is not None:
            sumo_cmd.extend(['--seed', str(sumo_seed)])
            
        # Step length and logging options
        sumo_cmd.extend([
            '--step-length', str(self.delta_time),
            '--no-step-log',
            '--quit-on-end'
        ])
        
        self.sumo_cmd = sumo_cmd
        
        # Traffic light settings
        self.traffic_lights = {}
        # State size: 4 lanes * (queue + wait + speed) + phase_duration + is_yellow
        self.observation_space_size = 4 * 3 + 2  # 14 values total
        self.action_space_size = 2  # Binary: keep/change phase
        
        # Episode statistics
        self.episode_step = 0
        self.episode_length = num_seconds // delta_time
        
        # Start SUMO
        self.start_simulation()
        
    def start_simulation(self):
        """Start or restart the SUMO simulation"""
        traci.start(self.sumo_cmd)
        self._get_traffic_lights()
        
    def _get_traffic_lights(self):
            """Get all traffic lights and their properties"""
            self.traffic_lights = {}
            for tl_id in traci.trafficlight.getIDList():
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                # Convert tuple of lanes to a set of unique lanes
                unique_lanes = set(controlled_lanes)
                
                self.traffic_lights[tl_id] = {
                    'phases': traci.trafficlight.getAllProgramLogics(tl_id)[0].phases,
                    'links': traci.trafficlight.getControlledLinks(tl_id),
                    'lanes': unique_lanes
                }
            
    def get_traffic_light_ids(self):
        """Return list of traffic light IDs"""
        return list(self.traffic_lights.keys())
        
    def _get_state(self, tl_id):
            """
            Get state for a specific traffic light.
            State includes:
            - Queue lengths for incoming lanes
            - Waiting times for incoming lanes
            - Current phase duration
            - Is current phase yellow
            """
            lanes = self.traffic_lights[tl_id]['lanes']
            
            # Get queue lengths and waiting times
            queue_lengths = []
            waiting_times = []
            speeds = []
            
            for lane in lanes:
                # Queue length (number of stopped vehicles)
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                queue = 0
                for veh_id in vehicle_ids:
                    try:
                        if traci.vehicle.getSpeed(veh_id) < 0.1:
                            queue += 1
                    except traci.exceptions.TraCIException:
                        continue
                queue_lengths.append(queue)
                
                # Waiting time
                wait_time = traci.lane.getWaitingTime(lane)
                waiting_times.append(wait_time)
                
                # Average speed
                speed = traci.lane.getLastStepMeanSpeed(lane)
                speeds.append(speed)

            # Normalize lengths to fixed size (e.g., take max 4 lanes or pad with zeros)
            max_lanes = 4
            queue_lengths = queue_lengths[:max_lanes] + [0] * (max_lanes - len(queue_lengths))
            waiting_times = waiting_times[:max_lanes] + [0] * (max_lanes - len(waiting_times))
            speeds = speeds[:max_lanes] + [0] * (max_lanes - len(speeds))
            
            # Current phase info
            try:
                phase_id = traci.trafficlight.getPhase(tl_id)
                current_program = traci.trafficlight.getAllProgramLogics(tl_id)[0]
                phase_duration = traci.trafficlight.getPhaseDuration(tl_id)
                
                if phase_id < len(current_program.phases):
                    is_yellow = 'y' in current_program.phases[phase_id].state.lower()
                else:
                    is_yellow = False
            except:
                phase_duration = 0
                is_yellow = False
            
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
        """
        Calculate reward for a traffic light based on:
        - Queue lengths
        - Waiting times
        - Number of stops
        """
        lanes = self.traffic_lights[tl_id]['lanes']
        
        # Calculate total waiting time and queue length
        total_wait = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
        total_queue = sum(1 for lane in lanes
                         for veh in traci.lane.getLastStepVehicleIDs(lane)
                         if traci.vehicle.getSpeed(veh) < 0.1)
        
        # Calculate reward (negative sum of waiting time and queue length)
        reward = -(total_wait + total_queue)
        
        return reward
        
    def step(self, actions):
            """
            Execute actions for all traffic lights and return results.
            
            Args:
                actions: dict with traffic light IDs as keys and actions as values
                
            Returns:
                states: dict of new states
                rewards: dict of rewards
                done: bool indicating if episode is done
                info: dict of additional information
            """
            # Execute actions
            for tl_id, action in actions.items():
                try:
                    current_phase = traci.trafficlight.getPhase(tl_id)
                    current_program = traci.trafficlight.getAllProgramLogics(tl_id)[0]
                    
                    if action == 1:  # Change phase
                        if current_phase < len(current_program.phases):
                            current_state = current_program.phases[current_phase].state
                            if 'y' not in current_state.lower():
                                # Not yellow, so change to next green phase
                                next_phase = (current_phase + 2) % len(current_program.phases)
                                traci.trafficlight.setPhase(tl_id, next_phase)
                except:
                    continue  # Skip this traffic light if there's any error
            
            # Run simulation step
            traci.simulationStep()
            self.episode_step += 1
            
            # Collect states and rewards
            states = {tl_id: self._get_state(tl_id) for tl_id in self.traffic_lights}
            rewards = {tl_id: self._get_reward(tl_id) for tl_id in self.traffic_lights}
            
            # Check if episode is done
            done = self.episode_step >= self.episode_length
            
            # Additional info
            info = {}
            
            return states, rewards, done, info
        
    def reset(self):
        """Reset the environment"""
        # Close existing simulation
        if traci.isLoaded():
            traci.close()
            
        # Start new simulation
        self.start_simulation()
        self.episode_step = 0
        
        # Get initial states
        states = {tl_id: self._get_state(tl_id) for tl_id in self.traffic_lights}
        
        # Update observation space size based on actual state size
        if len(states) > 0:
            self.observation_space_size = len(next(iter(states.values())))
            
        return states
        
    def close(self):
        """Close the environment"""
        if traci.isLoaded():
            traci.close()