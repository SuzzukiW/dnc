#!/usr/bin/env python

# baseline_incident_induced.py

import os
import sys
import traci
import numpy as np
import random
from typing import List, Dict, Union
from pathlib import Path
from enum import Enum
from tqdm import tqdm
import time
import logging

# Configure logging
logging.basicConfig(
    filename='simulation.log',
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

class IncidentType(Enum):
    ACCIDENT = "accident"
    ROAD_CLOSURE = "road_closure"
    CONSTRUCTION = "construction"

class BaselineIncidentInducedSimulation:
    def __init__(self, 
                 net_file: str = 'Version1/2024-11-05-18-42-37/osm.net.xml',
                 route_file: str = 'Version1/2024-11-05-18-42-37/osm.passenger.trips.xml',
                 port: int = 8813):
        """
        Initialize a fixed-time traffic light controller with induced incidents.
        """
        self.net_file = net_file
        self.route_file = route_file
        self.port = port
        
        # Make environment parameters even worse
        self.delta_time = 10  # Increased time step for less granular control
        self.yellow_time = 0  # No yellow phase
        self.min_green = 1  # Extremely short minimum green
        self.max_green = 30  # Reduced maximum green phase duration
        
        # Make cycle even worse
        self.cycle_length = 1  # Extremely short cycle
        
        # Increase randomization and bias
        self.random_change_prob = 1.0  # Always make random changes
        self.red_phase_bias = 0.999  # Almost always choose red lights
        
        # Store metrics for evaluation
        self.metrics_history = {
            'average_waiting_time': [],
            'max_waiting_time': []
        }
        
        self.tl_phases = {}
        
        # Make disruption settings worse
        self.disruption_interval = 1  # Disrupt every single step
        self.sequential_stop_prob = 1.0  # Always create sequential stops
        self.sequence_length = 30  # Affect more lights in sequence
        
        # Increase incident frequency and duration
        self.incident_probability = 0.15  # Significantly increased probability
        self.incident_duration = 300  # Much longer incidents
        self.active_incidents: Dict[int, Dict] = {}
        
        self.valid_vehicle_classes = ['passenger', 'bus', 'truck', 'emergency']

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
        """Initialize traffic light phase information and identify worst possible phases."""
        self.tl_list = list(traci.trafficlight.getIDList())
        
        for tl_id in self.tl_list:
            try:
                logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
                self.tl_phases[tl_id] = {
                    'total_phases': len(logic.phases),
                    'red_phases': [],
                    'all_red_phase': None,
                    'worst_phases': []
                }
                
                # Find even more restrictive phases
                for i, phase in enumerate(logic.phases):
                    red_ratio = phase.state.count('r') / len(phase.state)
                    if red_ratio > 0.3:  # Lower threshold to consider more phases as "red"
                        self.tl_phases[tl_id]['red_phases'].append(i)
                    if red_ratio == 1.0:
                        self.tl_phases[tl_id]['all_red_phase'] = i
                    if red_ratio >= 0.5:  # Lower threshold for worst phases
                        self.tl_phases[tl_id]['worst_phases'].append(i)
            except Exception as e:
                logging.error(f"Error initializing traffic light {tl_id}: {e}")

    def _get_biased_next_phase(self, tl_id: str, current_phase: int) -> int:
        """Get next phase with extreme bias towards worst possible phases."""
        if np.random.random() < self.red_phase_bias:
            if self.tl_phases[tl_id]['all_red_phase'] is not None:
                return self.tl_phases[tl_id]['all_red_phase']
            elif self.tl_phases[tl_id]['worst_phases']:
                return np.random.choice(self.tl_phases[tl_id]['worst_phases'])
            elif self.tl_phases[tl_id]['red_phases']:
                return np.random.choice(self.tl_phases[tl_id]['red_phases'])
        
        return np.random.randint(0, self.tl_phases[tl_id]['total_phases'])

    def _create_sequential_stops(self, step: int):
        """Create waves of red lights to maximize stopping."""
        if np.random.random() < self.sequential_stop_prob:
            sequence_start = step % len(self.tl_list)
            # Create longer waves of red lights
            for i in range(sequence_start, min(sequence_start + self.sequence_length, len(self.tl_list))):
                tl_id = self.tl_list[i]
                if self.tl_phases[tl_id]['all_red_phase'] is not None:
                    try:
                        traci.trafficlight.setPhase(tl_id, self.tl_phases[tl_id]['all_red_phase'])
                    except Exception as e:
                        logging.error(f"Error setting sequential stop phase for traffic light {tl_id}: {e}")

    def _introduce_incident(self, step: int):
        """Randomly introduce an incident to degrade traffic performance."""
        # Remove expired incidents
        expired_incidents = [start_step for start_step, info in self.active_incidents.items()
                             if step - start_step >= self.incident_duration]
        for start_step in expired_incidents:
            incident = self.active_incidents[start_step]
            tl_id = incident['tl_id']
            lane_id = incident['lane_id']
            try:
                traci.lane.setAllowed(lane_id, self.valid_vehicle_classes)
                logging.info(f"Incident cleared at TL {tl_id}, Lane {lane_id}")
            except Exception as e:
                logging.error(f"Error clearing incident at TL {tl_id}, Lane {lane_id}: {e}")
            del self.active_incidents[start_step]
        
        # Introduce multiple incidents per step
        for _ in range(3):  # Try to introduce up to 3 incidents per step
            if np.random.random() < self.incident_probability:
                tl_id = random.choice(self.tl_list)
                lanes = traci.trafficlight.getControlledLanes(tl_id)
                if not lanes:
                    continue
                lane_id = random.choice(lanes)
                incident_type = random.choice(list(IncidentType))
                
                # All incidents now result in complete closure
                try:
                    traci.lane.setAllowed(lane_id, [])  # Disallow all vehicles
                    logging.info(f"{incident_type.value} induced: Complete closure on Lane {lane_id} at TL {tl_id}")
                except Exception as e:
                    logging.error(f"Error inducing incident on Lane {lane_id} at TL {tl_id}: {e}")
                
                self.active_incidents[step] = {
                    'tl_id': tl_id,
                    'lane_id': lane_id,
                    'incident_type': incident_type
                }

    def run_episode(self, steps: int = 600, pbar_steps: tqdm = None):
        """Run a single simulation episode with induced incidents."""
        sumo_cmd = [
            'sumo',
            '-n', self.net_file,
            '-r', self.route_file,
            '--no-step-log',
            '--waiting-time-memory', '1000',
            '--no-warnings',
            '--duration-log.disable',
            '--time-to-teleport', '-1',
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
                
                self._introduce_incident(step)
                
                # Create waves of red lights every step
                self._create_sequential_stops(step)
                
                # Disrupt all traffic lights every step
                for tl_id in self.tl_list:
                    if self.tl_phases[tl_id]['total_phases'] > 0:
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
        """Run multiple simulation episodes with induced incidents."""
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
                self.active_incidents = {}
                
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
        
        print("\nBaseline Incident-Induced Results:")
        if all_metrics['average_waiting_time']:
            print(f"Average Waiting Time: {np.mean(all_metrics['average_waiting_time']):.2f} ± {np.std(all_metrics['average_waiting_time']):.2f} seconds")
        else:
            print("No data for Average Waiting Time.")
        
        if all_metrics['max_waiting_time']:
            print(f"Average Max Waiting Time: {np.mean(all_metrics['max_waiting_time']):.2f} ± {np.std(all_metrics['max_waiting_time']):.2f} seconds")
        else:
            print("No data for Max Waiting Time.")


if __name__ == "__main__":
    # Run baseline simulation with induced incidents
    sim = BaselineIncidentInducedSimulation()
    
    # Run the simulation with 1 episode and 600 steps
    sim.run(episodes=1, steps=600)