#!/usr/bin/env python

# baseline_special_event.py

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
    filename='special_event_simulation.log',
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

class BaselineSpecialEventSimulation:
    def __init__(self, 
                 net_file: str = 'Version1/2024-11-05-18-42-37/osm.net.xml',
                 route_file: str = 'Version1/2024-11-05-18-42-37/osm.passenger.trips.xml',
                 port: int = 8813):
        """
        Initialize a special event traffic simulation (e.g., TD Garden event).
        Focuses on localized congestion around venue area with moderate overall impact.
        """
        self.net_file = net_file
        self.route_file = route_file
        self.port = port
        
        # Environment parameters (more moderate than rush hour)
        self.delta_time = 6  # Simulation time step (seconds)
        self.yellow_time = 2  # Short yellow phase
        self.min_green = 5  # Reasonable minimum green
        self.max_green = 40  # Moderate maximum green phase duration
        
        # Timing parameters
        self.cycle_length = 5  # Moderate cycle length
        self.random_change_prob = 0.4  # Lower probability of random changes
        self.red_phase_bias = 0.7  # Moderate bias towards red phases
        
        # Store metrics
        self.metrics_history = {
            'average_waiting_time': [],
            'max_waiting_time': []
        }
        
        self.tl_phases = {}
        
        # Event-specific parameters
        self.event_area_radius = 5  # Number of intersections considered "near venue"
        self.venue_impact_factor = 0.8  # Higher congestion near venue
        self.disruption_interval = 8  # Less frequent disruptions
        self.sequential_stop_prob = 0.5  # Moderate probability of sequential stops
        self.sequence_length = 10  # Affect fewer lights in sequence
        
        # TD Garden area coordinates (approximate - adjust based on actual network)
        self.venue_location = {
            'x': 0,  # Will be set during initialization
            'y': 0   # Will be set during initialization
        }

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
        """Initialize traffic light phase information and identify venue area lights."""
        self.tl_list = list(traci.trafficlight.getIDList())
        
        # Find positions of traffic lights for venue location
        x_coords = []
        y_coords = []
        for tl_id in self.tl_list:
            try:
                # Get one of the controlled lanes
                lanes = traci.trafficlight.getControlledLanes(tl_id)
                if lanes:
                    # Get the position of the first controlled lane
                    lane_shape = traci.lane.getShape(lanes[0])
                    if lane_shape:
                        x, y = lane_shape[0]  # Get the start position of the lane
                        x_coords.append(x)
                        y_coords.append(y)
            except Exception:
                continue
        
        if x_coords and y_coords:
            # Set venue location to network center (adjust if needed)
            self.venue_location['x'] = np.mean(x_coords)
            self.venue_location['y'] = np.mean(y_coords)
        
        for tl_id in self.tl_list:
            try:
                logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
                # Calculate distance to venue using lane positions
                try:
                    lanes = traci.trafficlight.getControlledLanes(tl_id)
                    if lanes:
                        lane_shape = traci.lane.getShape(lanes[0])
                        if lane_shape:
                            x, y = lane_shape[0]
                            distance = np.sqrt((x - self.venue_location['x'])**2 + 
                                            (y - self.venue_location['y'])**2)
                        else:
                            distance = float('inf')
                    else:
                        distance = float('inf')
                except Exception:
                    distance = float('inf')
                
                self.tl_phases[tl_id] = {
                    'total_phases': len(logic.phases),
                    'red_phases': [],
                    'venue_distance': distance,
                    'near_venue': distance <= self.event_area_radius,
                    'congested_phases': []
                }
                
                # Identify congested phases
                for i, phase in enumerate(logic.phases):
                    red_ratio = phase.state.count('r') / len(phase.state)
                    if red_ratio > 0.5:
                        self.tl_phases[tl_id]['red_phases'].append(i)
                    if red_ratio >= 0.7:
                        self.tl_phases[tl_id]['congested_phases'].append(i)
            except Exception as e:
                logging.error(f"Error initializing traffic light {tl_id}: {e}")

    def _get_event_adjusted_phase(self, tl_id: str, current_phase: int) -> int:
        """Get next phase with bias based on proximity to venue."""
        near_venue = self.tl_phases[tl_id]['near_venue']
        bias = self.red_phase_bias * (1 + self.venue_impact_factor) if near_venue else self.red_phase_bias
        
        if np.random.random() < bias:
            if near_venue and self.tl_phases[tl_id]['congested_phases']:
                return np.random.choice(self.tl_phases[tl_id]['congested_phases'])
            elif self.tl_phases[tl_id]['red_phases']:
                return np.random.choice(self.tl_phases[tl_id]['red_phases'])
        
        return np.random.randint(0, self.tl_phases[tl_id]['total_phases'])

    def _create_venue_area_congestion(self, step: int):
        """Create concentrated congestion near venue area."""
        if np.random.random() < self.sequential_stop_prob:
            # Sort traffic lights by distance to venue
            sorted_lights = sorted(
                [(tl_id, self.tl_phases[tl_id]['venue_distance']) 
                 for tl_id in self.tl_list],
                key=lambda x: x[1]
            )
            
            # Affect nearby lights more severely
            for i in range(min(self.sequence_length, len(sorted_lights))):
                tl_id = sorted_lights[i][0]
                if self.tl_phases[tl_id]['congested_phases']:
                    try:
                        # Higher chance of congested phase for closer lights
                        if np.random.random() < (1 - i/self.sequence_length):
                            phase = np.random.choice(self.tl_phases[tl_id]['congested_phases'])
                            traci.trafficlight.setPhase(tl_id, phase)
                    except Exception as e:
                        logging.error(f"Error setting venue congestion phase for traffic light {tl_id}: {e}")

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
            '--time-to-teleport', '300',  # Allow teleporting after 5 minutes
            '--collision.action', 'warn',
            '--seed', str(random.randint(0, 10000)),
            '--step-length', str(self.delta_time),
            '--begin', '0',
            '--quit-on-end',
            '--random',
            '--max-depart-delay', '1800',  # 30 minutes max delay
            '--lateral-resolution', '0.8',
            '--ignore-route-errors',
            '--no-internal-links'
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
                
                # Create venue area congestion periodically
                if step % self.disruption_interval == 0:
                    self._create_venue_area_congestion(step)
                
                # Regular control with event-specific adjustments
                for tl_id in self.tl_list:
                    if self.tl_phases[tl_id]['total_phases'] > 0:
                        if step % self.cycle_length == 0 or np.random.random() < self.random_change_prob:
                            try:
                                current_phase = traci.trafficlight.getPhase(tl_id)
                                next_phase = self._get_event_adjusted_phase(tl_id, current_phase)
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
        
        print("\nBaseline Special Event Results:")
        if all_metrics['average_waiting_time']:
            print(f"Average Waiting Time: {np.mean(all_metrics['average_waiting_time']):.2f} ± {np.std(all_metrics['average_waiting_time']):.2f} seconds")
        else:
            print("No data for Average Waiting Time.")
        
        if all_metrics['max_waiting_time']:
            print(f"Average Max Waiting Time: {np.mean(all_metrics['max_waiting_time']):.2f} ± {np.std(all_metrics['max_waiting_time']):.2f} seconds")
        else:
            print("No data for Max Waiting Time.")


if __name__ == "__main__":
    # Run baseline special event simulation
    sim = BaselineSpecialEventSimulation()
    
    # Run the simulation with 1 episode and 600 steps
    sim.run(episodes=1, steps=600)