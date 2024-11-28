# src/environment/multi_agent_sumo_env_maddpg.py

import os
import sys
import numpy as np
from typing import List, Tuple, Dict, Optional
import traci
import sumolib
from gymnasium import Env
from gymnasium import spaces
import traci

# Set up SUMO_HOME environment variable
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

class MultiAgentSumoEnvironmentMADDPG(Env):
    """Multi-agent SUMO environment for traffic signal control using MADDPG."""
    
    def __init__(self, 
                 net_file: str = "osm.net.xml.gz",  # Match baseline default
                 route_file: str = "osm.passenger.trips.xml",  # Match baseline default
                 use_gui: bool = False,
                 num_seconds: int = 3600,
                 delta_time: int = 1,  # Changed to 1 to match baseline
                 yellow_time: int = 3,  # SUMO default
                 min_green: int = 5,  # SUMO default
                 max_green: int = 60,  # SUMO default
                 num_agents: int = 1,
                 port: int = 8813):
        """Initialize the environment."""
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.num_agents = num_agents
        self.port = port
        
        # Initialize episode step counter
        self.episode_step = 0
        
        # Track vehicles like baseline
        self.vehicles_seen = set()
        self.vehicles_completed = set()
        
        # Suppress SUMO output
        import os
        os.environ['SUMO_QUIET'] = '1'
        
        # Debug: Print minimal configuration
        print("\nSUMO Environment Configuration:")
        print(f"Network file: {os.path.abspath(self.net_file)}")
        print(f"Route file: {os.path.abspath(self.route_file)}")
        print(f"GUI enabled: {self.use_gui}")
        
        # Set up SUMO
        if 'SUMO_HOME' not in os.environ:
            os.environ['SUMO_HOME'] = '/usr/local/opt/sumo/share/sumo'
            print(f"Setting SUMO_HOME to: {os.environ['SUMO_HOME']}")
        
        # Get directory containing network file
        net_dir = os.path.dirname(self.net_file)
        
        # Use only the specified route file
        if not os.path.exists(self.route_file):
            raise FileNotFoundError(f"Route file not found: {self.route_file}")
        
        print(f"\nUsing route file: {os.path.basename(self.route_file)}")
        
        # Construct SUMO command with minimal logging
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        self.sumo_cmd = [
            sumo_binary,
            '--net-file', self.net_file,
            '--route-files', self.route_file,
            '--step-length', '1.0',
            '--no-step-log',
            '--no-warnings',
            '--begin', '0',
            '--time-to-teleport', '300',  # Match baseline: Teleport after 300s of waiting
            '--log-file', '/dev/null',  # Redirect logs to null
            '--verbose', '0'  # Minimal verbosity
        ]
        
        print(f"\nSUMO command: {' '.join(self.sumo_cmd)}")
        
        try:
            # Start SUMO to get initial network structure
            traci.start(self.sumo_cmd, port=self.port)
            
            # Get all traffic lights and filter valid ones
            self.traffic_lights = list(traci.trafficlight.getIDList())
            print(f"\nFound {len(self.traffic_lights)} total traffic lights")
            
            self.traffic_lights = self._filter_valid_traffic_lights()
            print(f"\nUsing {len(self.traffic_lights)} valid traffic lights")
            
            if not self.traffic_lights:
                raise ValueError("No valid traffic lights found in the network!")
            
            # Initialize observation and action spaces
            self.observation_spaces = {}
            self.action_spaces = {}
            
            for tl_id in self.traffic_lights:
                # State space: queue lengths, waiting times, speeds for each lane
                num_lanes = len(traci.trafficlight.getControlledLanes(tl_id))
                self.observation_spaces[tl_id] = spaces.Box(
                    low=0,
                    high=float('inf'),
                    shape=(num_lanes * 3,),  # queue, wait time, speed for each lane
                    dtype=np.float32
                )
                
                # Action space: continuous action for each phase
                num_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
                self.action_spaces[tl_id] = spaces.Box(
                    low=-1,
                    high=1,
                    shape=(num_phases,),
                    dtype=np.float32
                )
            
            traci.close()
        except Exception as e:
            print(f"Error during SUMO initialization: {str(e)}")
            raise
        
        print(f"\nUsing {len(self.traffic_lights)} valid traffic lights as agents")
        
    def _filter_valid_traffic_lights(self) -> List[str]:
        """Filter out traffic lights that don't have valid programs."""
        valid_tls = []
        for tl_id in self.traffic_lights:
            try:
                programs = traci.trafficlight.getAllProgramLogics(tl_id)
                if not programs:
                    print(f"Traffic light {tl_id} has no programs")
                    continue
                    
                phases = programs[0].phases
                if not phases:
                    print(f"Traffic light {tl_id} has no phases")
                    continue
                    
                # Check if the traffic light controls any lanes
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                if not controlled_lanes:
                    print(f"Traffic light {tl_id} controls no lanes")
                    continue
                
                # Verify that the phases have valid states
                valid_phases = True
                for phase in phases:
                    if len(phase.state) != len(controlled_lanes):
                        print(f"Traffic light {tl_id} has mismatched phase state length")
                        valid_phases = False
                        break
                
                if not valid_phases:
                    continue
                
                valid_tls.append(tl_id)
                print(f"Traffic light {tl_id} is valid:")
                print(f"  - Controls {len(controlled_lanes)} lanes")
                print(f"  - Has {len(phases)} phases")
                print(f"  - Phase state length: {len(phases[0].state)}")
                
            except Exception as e:
                print(f"Error checking traffic light {tl_id}: {str(e)}")
                continue
        
        return valid_tls
    
    def reset(self) -> Dict:
        """Reset the environment."""
        # Reset vehicle tracking
        self.vehicles_seen = set()
        self.vehicles_completed = set()
        
        # Reset SUMO simulation
        if traci.isLoaded():
            traci.close()
        
        print("\nStarting new episode...")
        traci.start(self.sumo_cmd, port=self.port)
        
        # Reset episode step counter
        self.episode_step = 0
        
        # Wait for vehicles to populate the network
        print("\nWarming up simulation...")
        warmup_steps = 100  # Increased warm-up period
        vehicles_per_step = []
        
        for step in range(warmup_steps):
            traci.simulationStep()
            vehicles = traci.vehicle.getIDList()
            vehicles_per_step.append(len(vehicles))
            
            # Track all vehicles in the network
            self.vehicles_seen.update(vehicles)
            
            if step % 10 == 0:  # Print every 10 steps
                print(f"Warm-up step {step}: {len(vehicles)} vehicles")
                if vehicles:
                    print(f"Sample vehicles: {list(vehicles)[:5]}")
                    print(f"Vehicle speeds: {[traci.vehicle.getSpeed(v) for v in list(vehicles)[:5]]}")
        
        print("\nWarm-up complete!")
        print(f"Vehicle count progression: {vehicles_per_step[::10]}")  # Show every 10th step
        
        self.simulation_time = 0
        self.yellow_phase_countdown = {tl: 0 for tl in self.traffic_lights}
        self.current_phases = {tl: 0 for tl in self.traffic_lights}
        
        # Get initial observations
        observations = {}
        total_vehicles = len(self.vehicles_seen)  # Count all vehicles seen
        total_waiting = sum(traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList())
        
        for tl_id in self.traffic_lights:
            observations[tl_id] = self._get_observation(tl_id)
        
        return observations
    
    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        """Execute one step in the environment."""
        # Track vehicles like baseline
        current_vehicles = set(traci.vehicle.getIDList())
        self.vehicles_seen.update(current_vehicles)
        new_completed = self.vehicles_seen - current_vehicles - self.vehicles_completed
        self.vehicles_completed.update(new_completed)
        
        # Apply actions
        for tl_id, action in actions.items():
            self._apply_action(tl_id, action)
        
        # Run simulation for one timestep
        traci.simulationStep()
        self.simulation_time += self.delta_time
        
        # Increment episode step counter
        self.episode_step += 1
        
        # Get observations and rewards
        observations = {}
        rewards = {}
        info = {'traffic_lights': {}}
        
        for tl_id in self.traffic_lights:
            observations[tl_id] = self._get_observation(tl_id)
            rewards[tl_id] = self._compute_reward(tl_id)
            
            # Collect traffic light info without printing
            lanes = traci.trafficlight.getControlledLanes(tl_id)
            tl_info = {
                'num_vehicles': sum(len(traci.lane.getLastStepVehicleIDs(lane)) for lane in lanes),
                'total_waiting_time': sum(traci.lane.getWaitingTime(lane) for lane in lanes),
                'avg_speed': np.mean([traci.lane.getLastStepMeanSpeed(lane) for lane in lanes]),
                'phase': self.current_phases[tl_id],
                'yellow_countdown': self.yellow_phase_countdown[tl_id]
            }
            info['traffic_lights'][tl_id] = tl_info
        
        # Check if simulation is done
        done = self.simulation_time >= self.num_seconds
        dones = {tl_id: done for tl_id in self.traffic_lights}
        
        # Add global metrics to info
        info.update(self._get_info())
        
        return observations, rewards, dones, info
    
    def _get_observation(self, tl_id: str) -> np.ndarray:
        """Get observation for a traffic light.
        
        Args:
            tl_id: Traffic light ID
            
        Returns:
            Numpy array containing the observation
        """
        lanes = traci.trafficlight.getControlledLanes(tl_id)
        observation = []
        
        for lane in lanes:
            # Queue length
            queue_length = traci.lane.getLastStepHaltingNumber(lane)
            
            # Waiting time
            waiting_time = traci.lane.getWaitingTime(lane)
            
            # Average speed
            mean_speed = traci.lane.getLastStepMeanSpeed(lane)
            
            observation.extend([queue_length, waiting_time, mean_speed])
        
        return np.array(observation, dtype=np.float32)
    
    def _compute_reward(self, tl_id: str) -> float:
        """Compute reward for a traffic light."""
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        vehicles = set()
        
        # Get all vehicles in controlled lanes
        for lane in controlled_lanes:
            vehicles.update(traci.lane.getLastStepVehicleIDs(lane))
        
        num_vehicles = len(vehicles)
        if num_vehicles > 0:
            total_waiting_time = 0
            total_speed = 0
            total_queue_length = 0

            for v in vehicles:
                # Get waiting time and time loss
                waiting_time = traci.vehicle.getWaitingTime(v)
                time_loss = traci.vehicle.getTimeLoss(v)
                speed = traci.vehicle.getSpeed(v)
                
                # Accumulate metrics, converting waiting time to seconds
                total_waiting_time += (waiting_time + time_loss) / 1000.0  # Convert ms to seconds
                total_speed += speed
                if speed < 0.1:  # Vehicle is considered stopped
                    total_queue_length += 1
            
            # Calculate averages
            avg_waiting_time = total_waiting_time / num_vehicles
            avg_speed = total_speed / num_vehicles
            avg_queue_length = total_queue_length / num_vehicles
            
            # Normalize metrics (same as baseline)
            waiting_time_factor = -min(avg_waiting_time / 180.0, 1.0)  # Cap at 3 minutes
            queue_length_factor = -min(avg_queue_length / 10.0, 1.0)   # Cap at 10 vehicles
            speed_factor = avg_speed / traci.vehicle.getMaxSpeed(list(vehicles)[0])  # Normalize by max speed
            
            # Combine rewards with same weights as baseline
            reward = (
                0.4 * waiting_time_factor +    # Penalize waiting time
                0.3 * queue_length_factor +    # Penalize queue length
                0.3 * speed_factor            # Reward higher speeds
            )
            
            return reward * 100  # Scale reward same as baseline
        
        return 0  # Return 0 if no vehicles
    
    def _apply_action(self, tl_id: str, action: np.ndarray):
        """Apply action to traffic light.
        
        Args:
            tl_id: Traffic light ID
            action: Action array
        """
        # Check if yellow phase is active
        if self.yellow_phase_countdown[tl_id] > 0:
            self.yellow_phase_countdown[tl_id] -= self.delta_time
            return
            
        # Get current program and phases
        program = traci.trafficlight.getAllProgramLogics(tl_id)[0]
        phases = program.phases
        
        # Convert normalized action to phase index
        action = (action + 1) / 2  # Convert from [-1, 1] to [0, 1]
        phase_index = int(np.argmax(action) % len(phases))
        
        # Get current phase index
        current_phase = self.current_phases[tl_id]
        
        # If changing phase, set yellow phase
        if phase_index != current_phase:
            # Create yellow state by replacing all 'g' with 'y' in current state
            current_state = phases[current_phase].state
            yellow_state = ''.join(['y' if c in 'gG' else c for c in current_state])
            
            # Apply yellow state
            traci.trafficlight.setRedYellowGreenState(tl_id, yellow_state)
            self.yellow_phase_countdown[tl_id] = self.yellow_time
            
            # Store the next phase to apply after yellow
            self.current_phases[tl_id] = phase_index
        else:
            # Apply the current phase state directly
            traci.trafficlight.setRedYellowGreenState(tl_id, phases[phase_index].state)
            
    def _get_info(self) -> Dict:
        """Get additional information about the environment."""
        vehicles = traci.vehicle.getIDList()
        num_vehicles = len(vehicles)
        
        if num_vehicles > 0:
            # Calculate metrics for all vehicles
            total_waiting_time = 0
            total_speed = 0
            total_queue_length = 0

            for v in vehicles:
                # Get waiting time and time loss
                waiting_time = traci.vehicle.getWaitingTime(v)
                total_waiting_time += waiting_time
                
                speed = traci.vehicle.getSpeed(v)
                total_speed += speed
                if speed < 0.1:  # Vehicle is considered stopped
                    total_queue_length += 1
            
            # Calculate averages
            avg_waiting_time = total_waiting_time / (num_vehicles * 1000.0)  # Convert ms to seconds
            avg_speed = total_speed / num_vehicles
            avg_queue_length = total_queue_length / num_vehicles
            
            return {
                'average_waiting_time': avg_waiting_time,
                'average_queue_length': avg_queue_length,
                'average_speed': avg_speed,
                'throughput': traci.simulation.getArrivedNumber(),
                'total_vehicles': num_vehicles,
                'simulation_time': self.simulation_time,
                'episode_step': self.episode_step
            }
        else:
            return {
                'average_waiting_time': 0,
                'average_queue_length': 0,
                'average_speed': 0,
                'throughput': 0,
                'total_vehicles': 0,
                'simulation_time': self.simulation_time,
                'episode_step': self.episode_step
            }
    
    def close(self):
        """Close SUMO simulation."""
        traci.close()
        
    def render(self):
        """Render method is not implemented as SUMO-GUI can be used instead."""
        pass
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)