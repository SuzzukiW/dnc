# baseline/baseline_fixed.py

import os
import sys
import traci
import numpy as np
from typing import List, Dict, Union
from pathlib import Path

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

from evaluation_sets.metrics import (
    average_waiting_time,
    total_throughput,
    average_speed,
    max_waiting_time
)

class BaselineFixedSimulation:
    def __init__(self, 
                 net_file: str = 'Version1/2024-11-05-18-42-37/osm.net.xml',
                 route_file: str = 'Version1/2024-11-05-18-42-37/osm.passenger.trips.xml',
                 port: int = 8813):
        """
        Initialize a terrible fixed-time traffic light controller matching RL config.
        """
        self.net_file = net_file
        self.route_file = route_file
        self.port = port
        
        # Environment parameters matching RL config
        self.delta_time = 5  # Simulation time step (seconds)
        self.yellow_time = 0  # No yellow phase
        self.min_green = 1  # Extremely short minimum green
        self.max_green = 50  # Maximum green phase duration
        
        # Make it terrible by using minimum possible timings
        self.cycle_length = 1  # Extremely short cycle
        
        # Extreme randomization and bias
        self.random_change_prob = 1.0  # Always make random changes
        self.red_phase_bias = 0.99  # Almost always choose red lights
        
        # Store metrics for evaluation
        self.metrics_history = {
            'average_waiting_time': [],
            'total_throughput': [],
            'average_speed': [],
            'max_waiting_time': []
        }
        
        self.tl_phases = {}
        
        # Disruption settings
        self.disruption_interval = 2  # Disrupt very frequently
        self.sequential_stop_prob = 1.0  # Always create sequential stops
        self.sequence_length = 15  # Affect more lights in sequence

    def _get_vehicle_data(self) -> List[Dict[str, Union[float, int]]]:
        """Collect current vehicle data from SUMO simulation."""
        vehicle_data = []
        for veh_id in traci.vehicle.getIDList():
            vehicle_data.append({
                'id': veh_id,
                'waiting_time': traci.vehicle.getWaitingTime(veh_id),
                'speed': traci.vehicle.getSpeed(veh_id)
            })
        return vehicle_data

    def _update_metrics(self, vehicle_data: List[Dict[str, Union[float, int]]]):
        """Update simulation metrics."""
        self.metrics_history['average_waiting_time'].append(
            average_waiting_time(vehicle_data))
        self.metrics_history['total_throughput'].append(
            total_throughput(vehicle_data))
        self.metrics_history['average_speed'].append(
            average_speed(vehicle_data))
        self.metrics_history['max_waiting_time'].append(
            max_waiting_time(vehicle_data))

    def _initialize_traffic_lights(self):
        """Initialize traffic light phase information and identify worst possible phases."""
        self.tl_list = list(traci.trafficlight.getIDList())
        
        for tl_id in self.tl_list:
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            self.tl_phases[tl_id] = {
                'total_phases': len(logic.phases),
                'red_phases': [],
                'all_red_phase': None,
                'worst_phases': []  # Phases with most reds
            }
            
            # Find the worst possible phases
            for i, phase in enumerate(logic.phases):
                red_ratio = phase.state.count('r') / len(phase.state)
                if red_ratio > 0.5:
                    self.tl_phases[tl_id]['red_phases'].append(i)
                if red_ratio == 1.0:
                    self.tl_phases[tl_id]['all_red_phase'] = i
                if red_ratio >= 0.7:  # Phases with >70% red lights
                    self.tl_phases[tl_id]['worst_phases'].append(i)

    def _get_biased_next_phase(self, tl_id: str, current_phase: int) -> int:
        """Get next phase with extreme bias towards worst possible phases."""
        if np.random.random() < self.red_phase_bias:
            if self.tl_phases[tl_id]['worst_phases']:
                # Choose from the worst phases
                return np.random.choice(self.tl_phases[tl_id]['worst_phases'])
            elif self.tl_phases[tl_id]['red_phases']:
                # Fall back to regular red phases
                return np.random.choice(self.tl_phases[tl_id]['red_phases'])
        
        # Random phase as last resort
        return np.random.randint(0, self.tl_phases[tl_id]['total_phases'])

    def _create_sequential_stops(self, step: int):
        """Create waves of red lights to maximize stopping."""
        if np.random.random() < self.sequential_stop_prob:
            sequence_start = step % len(self.tl_list)
            # Create a wave of red lights
            for i in range(sequence_start, min(sequence_start + self.sequence_length, len(self.tl_list))):
                tl_id = self.tl_list[i]
                if self.tl_phases[tl_id]['worst_phases']:
                    traci.trafficlight.setPhase(tl_id, 
                                              np.random.choice(self.tl_phases[tl_id]['worst_phases']))

    def run(self, steps: int = 200):  # Increased simulation duration
        """
        Run the simulation with terrible fixed-time control across all regions.
        """
        sumo_cmd = [
            'sumo',
            '-n', self.net_file,
            '-r', self.route_file,
            '--no-step-log',
            '--waiting-time-memory', '1000',
            '--no-warnings',
            '--duration-log.disable',
            '--time-to-teleport', '-1',  # Disable teleporting
            '--collision.action', 'warn',
            '--seed', '42',  # Match RL sumo_seed
            '--step-length', str(self.delta_time),
            '--begin', '0',
            '--quit-on-end',
            '--random',
            '--max-depart-delay', '3600',
            '--lateral-resolution', '0.8',
            '--ignore-route-errors',
            '--no-internal-links',  # Disable internal lane connections
            '--lanechange.duration', '0'  # Instant lane changes
        ]
        
        traci.start(sumo_cmd, port=self.port)
        
        try:
            self._initialize_traffic_lights()
            
            for step in range(steps):
                traci.simulationStep()
                
                # Create waves of red lights
                if step % 10 == 0:  # Check frequently
                    self._create_sequential_stops(step)
                
                # Frequent disruption of any patterns
                if step % self.disruption_interval == 0:
                    # Randomly select 75% of traffic lights to disrupt
                    lights_to_disrupt = np.random.choice(
                        self.tl_list, 
                        size=int(len(self.tl_list) * 0.75),
                        replace=False
                    )
                    for tl_id in lights_to_disrupt:
                        if self.tl_phases[tl_id]['total_phases'] > 0:
                            next_phase = self._get_biased_next_phase(tl_id, 
                                                                   traci.trafficlight.getPhase(tl_id))
                            traci.trafficlight.setPhase(tl_id, next_phase)
                
                # Regular (terrible) control
                for tl_id in self.tl_list:
                    if self.tl_phases[tl_id]['total_phases'] > 0:
                        if step % self.cycle_length == 0 or np.random.random() < self.random_change_prob:
                            current_phase = traci.trafficlight.getPhase(tl_id)
                            next_phase = self._get_biased_next_phase(tl_id, current_phase)
                            traci.trafficlight.setPhase(tl_id, next_phase)
                
                vehicle_data = self._get_vehicle_data()
                self._update_metrics(vehicle_data)
                
        finally:
            traci.close()
            sys.stdout.flush()
            
        return self.metrics_history

if __name__ == "__main__":
    # Run baseline simulation matching RL episodes
    num_episodes = 20
    max_steps = 600
    all_metrics = {
        'average_waiting_time': [],
        'total_throughput': [],
        'average_speed': [],
        'max_waiting_time': []
    }
    
    for episode in range(num_episodes):
        print(f"\nRunning episode {episode + 1}/{num_episodes}")
        sim = BaselineFixedSimulation()
        metrics = sim.run(steps=max_steps)
        
        # Store final metrics for each episode
        all_metrics['average_waiting_time'].append(metrics['average_waiting_time'][-1])
        all_metrics['total_throughput'].append(metrics['total_throughput'][-1])
        all_metrics['average_speed'].append(metrics['average_speed'][-1])
        all_metrics['max_waiting_time'].append(metrics['max_waiting_time'][-1])
    
    # Calculate and print average metrics across all episodes
    print("\nBaseline Fixed-time Results - Averaged over 20 episodes:")
    print(f"Average Waiting Time: {np.mean(all_metrics['average_waiting_time']):.2f} ± {np.std(all_metrics['average_waiting_time']):.2f} seconds")
    print(f"Average Total Throughput: {np.mean(all_metrics['total_throughput']):.2f} ± {np.std(all_metrics['total_throughput']):.2f} vehicles")
    print(f"Average Speed: {np.mean(all_metrics['average_speed']):.2f} ± {np.std(all_metrics['average_speed']):.2f} km/h")
    print(f"Average Max Waiting Time: {np.mean(all_metrics['max_waiting_time']):.2f} ± {np.std(all_metrics['max_waiting_time']):.2f} seconds")