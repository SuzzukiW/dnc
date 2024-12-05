#!/usr/bin/env python

# baseline_rush_hour.py

import os
import sys
import traci
import numpy as np
import random
from typing import List, Dict, Union
from pathlib import Path
from tqdm import tqdm
import time
import logging

# Configure logging
logging.basicConfig(
    filename='rush_hour_simulation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add SUMO_HOME to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Import metrics
try:
    from evaluation_sets.metrics import (
        average_waiting_time,
        max_waiting_time
    )
except ImportError:
    logging.error("Failed to import metrics. Ensure 'evaluation_sets.metrics' is accessible.")
    sys.exit("Failed to import metrics.")

class BaselineRushHourSimulation:
    def __init__(self, 
                 net_file: str = 'Version1/2024-11-05-18-42-37/osm.net.xml',
                 route_file: str = 'Version1/2024-11-05-18-42-37/osm.passenger.trips.xml',
                 port: int = 8813):
        """
        Initialize a rush-hour traffic simulation with poorly timed signals.
        """
        self.net_file = net_file
        self.route_file = route_file
        self.port = port
        
        # Environment parameters
        self.delta_time = 8  # Simulation time step (seconds)
        self.yellow_time = 0  # No yellow phase
        self.min_green = 1  # Very short minimum green
        self.max_green = 35  # Limited maximum green phase duration
        
        # Poor timing parameters
        self.cycle_length = 2  # Very short cycle
        self.random_change_prob = 0.8  # High probability of random changes
        self.red_phase_bias = 0.95  # Strong bias towards red phases
        
        # Store metrics
        self.metrics_history = {
            'average_waiting_time': [],
            'max_waiting_time': []
        }
        
        self.tl_phases = {}
        
        # Disruption settings (less aggressive than incident-induced)
        self.disruption_interval = 3  # Disrupt every 3 steps
        self.sequential_stop_prob = 0.8  # High probability of sequential stops
        self.sequence_length = 20  # Affect multiple lights in sequence

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
        self.metrics_history['max_waiting_time'].append(
            max_waiting_time(vehicle_data))

    def _initialize_traffic_lights(self):
        """Initialize traffic light phase information and identify inefficient phases."""
        self.tl_list = list(traci.trafficlight.getIDList())
        
        for tl_id in self.tl_list:
            try:
                logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
                self.tl_phases[tl_id] = {
                    'total_phases': len(logic.phases),
                    'red_phases': [],
                    'all_red_phase': None,
                    'inefficient_phases': []  # Phases with high red ratios
                }
                
                # Identify inefficient phases
                for i, phase in enumerate(logic.phases):
                    red_ratio = phase.state.count('r') / len(phase.state)
                    if red_ratio > 0.4:  # More lenient threshold than incident-induced
                        self.tl_phases[tl_id]['red_phases'].append(i)
                    if red_ratio == 1.0:
                        self.tl_phases[tl_id]['all_red_phase'] = i
                    if red_ratio >= 0.6:  # More lenient threshold
                        self.tl_phases[tl_id]['inefficient_phases'].append(i)
            except Exception as e:
                logging.error(f"Error initializing traffic light {tl_id}: {e}")

    def _get_biased_next_phase(self, tl_id: str, current_phase: int) -> int:
        """Get next phase with bias towards red phases."""
        if np.random.random() < self.red_phase_bias:
            if self.tl_phases[tl_id]['inefficient_phases']:
                return np.random.choice(self.tl_phases[tl_id]['inefficient_phases'])
            elif self.tl_phases[tl_id]['red_phases']:
                return np.random.choice(self.tl_phases[tl_id]['red_phases'])
        
        return np.random.randint(0, self.tl_phases[tl_id]['total_phases'])

    def _create_sequential_stops(self, step: int):
        """Create waves of red lights to simulate rush hour congestion."""
        if np.random.random() < self.sequential_stop_prob:
            sequence_start = step % len(self.tl_list)
            for i in range(sequence_start, min(sequence_start + self.sequence_length, len(self.tl_list))):
                tl_id = self.tl_list[i]
                if self.tl_phases[tl_id]['inefficient_phases']:
                    try:
                        phase = np.random.choice(self.tl_phases[tl_id]['inefficient_phases'])
                        traci.trafficlight.setPhase(tl_id, phase)
                    except Exception as e:
                        logging.error(f"Error setting sequential stop phase for traffic light {tl_id}: {e}")

    def run_episode(self, steps: int = 600, pbar_steps: tqdm = None):
        """Run a single simulation episode."""
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
            '--seed', str(random.randint(0, 10000)),
            '--step-length', str(self.delta_time),
            '--begin', '0',
            '--quit-on-end',
            '--random',
            '--max-depart-delay', '3600',
            '--lateral-resolution', '0.8',
            '--ignore-route-errors',
            '--no-internal-links',
            '--lanechange.duration', '0'
        ]
        
        try:
            traci.start(sumo_cmd, port=self.port)
            logging.info(f"SUMO started on port {self.port} for episode.")
            
            self._initialize_traffic_lights()
            
            for step in range(steps):
                try:
                    traci.simulationStep()
                except Exception as e:
                    logging.error(f"Error during simulation step {step}: {e}")
                    break
                
                # Create waves of red lights periodically
                if step % self.disruption_interval == 0:
                    self._create_sequential_stops(step)
                
                # Regular inefficient control
                for tl_id in self.tl_list:
                    if self.tl_phases[tl_id]['total_phases'] > 0:
                        if step % self.cycle_length == 0 or np.random.random() < self.random_change_prob:
                            try:
                                current_phase = traci.trafficlight.getPhase(tl_id)
                                next_phase = self._get_biased_next_phase(tl_id, current_phase)
                                traci.trafficlight.setPhase(tl_id, next_phase)
                            except Exception as e:
                                logging.error(f"Error setting phase for traffic light {tl_id}: {e}")
                
                vehicle_data = self._get_vehicle_data()
                self._update_metrics(vehicle_data)
                
                if pbar_steps is not None:
                    pbar_steps.update(1)
                
        except Exception as e:
            logging.error(f"Simulation error: {e}")
        finally:
            try:
                traci.close()
                logging.info(f"SUMO closed for episode.")
            except Exception as e:
                logging.error(f"Error closing SUMO: {e}")
            sys.stdout.flush()

    def run(self, episodes: int = 1, steps: int = 600):
        """Run multiple simulation episodes."""
        all_metrics = {
            'average_waiting_time': [],
            'max_waiting_time': []
        }
        
        with tqdm(total=episodes, desc="Episodes", unit="episode") as pbar_episodes:
            for episode in range(episodes):
                print(f"\nRunning episode {episode + 1}/{episodes}")
                start_time = time.time()
                self.metrics_history = {
                    'average_waiting_time': [],
                    'max_waiting_time': []
                }
                
                with tqdm(total=steps, desc=f"Episode {episode + 1} Steps", leave=False, unit="step") as pbar_steps:
                    self.run_episode(steps=steps, pbar_steps=pbar_steps)
                
                if self.metrics_history['average_waiting_time']:
                    all_metrics['average_waiting_time'].append(self.metrics_history['average_waiting_time'][-1])
                else:
                    logging.warning(f"No data for Average Waiting Time in episode {episode + 1}")
                
                if self.metrics_history['max_waiting_time']:
                    all_metrics['max_waiting_time'].append(self.metrics_history['max_waiting_time'][-1])
                else:
                    logging.warning(f"No data for Max Waiting Time in episode {episode + 1}")
                
                end_time = time.time()
                print(f"Episode {episode + 1} completed in {end_time - start_time:.2f} seconds")
                
                if self.metrics_history['average_waiting_time']:
                    avg_wait = self.metrics_history['average_waiting_time'][-1]
                    print(f"Episode {episode + 1} - Average Waiting Time: {avg_wait:.2f} seconds")
                else:
                    print(f"Episode {episode + 1} - No data for Average Waiting Time.")
                
                if self.metrics_history['max_waiting_time']:
                    max_wait = self.metrics_history['max_waiting_time'][-1]
                    print(f"Episode {episode + 1} - Max Waiting Time: {max_wait:.2f} seconds")
                else:
                    print(f"Episode {episode + 1} - No data for Max Waiting Time.")
                
                pbar_episodes.update(1)
        
        print("\nBaseline Rush Hour Results:")
        if all_metrics['average_waiting_time']:
            print(f"Average Waiting Time: {np.mean(all_metrics['average_waiting_time']):.2f} ± {np.std(all_metrics['average_waiting_time']):.2f} seconds")
        else:
            print("No data for Average Waiting Time.")
        
        if all_metrics['max_waiting_time']:
            print(f"Average Max Waiting Time: {np.mean(all_metrics['max_waiting_time']):.2f} ± {np.std(all_metrics['max_waiting_time']):.2f} seconds")
        else:
            print("No data for Max Waiting Time.")


if __name__ == "__main__":
    # Run baseline rush hour simulation
    sim = BaselineRushHourSimulation()
    
    # Run the simulation with 1 episode and 600 steps
    sim.run(episodes=1, steps=600)