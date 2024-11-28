#!/usr/bin/env python

import os
import sys
import traci
import numpy as np
from typing import List, Dict, Union
from pathlib import Path

# Add SUMO_HOME to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Import metrics
from evaluation_sets.metrics import (
    average_waiting_time,
    total_throughput,
    average_speed,
    max_waiting_time
)

class UniformTrafficSimulation:
    def __init__(self, 
                 net_file: str = 'baseline/osm.net.xml.gz',
                 route_file: str = 'baseline/osm.passenger.trips.xml',
                 port: int = 8813):
        """
        Initialize adaptive controller for uniform traffic distribution.
        Assumes relatively consistent traffic flow throughout the network.
        """
        self.net_file = net_file
        self.route_file = route_file
        self.port = port
        
        # Parameters tuned for uniform distribution
        self.min_green_time = 20    # Longer minimum for consistent flow
        self.max_green_time = 45    # Moderate maximum for uniform distribution
        self.yellow_time = 4
        self.queue_threshold = 10    # Moderate threshold for uniform flow
        self.detection_range = 80    # Standard detection range
        
        # Store metrics for evaluation
        self.metrics_history = {
            'average_waiting_time': [],
            'total_throughput': [],
            'average_speed': [],
            'max_waiting_time': []
        }
        
        # Store traffic light information
        self.tl_phases = {}
        self.phase_start_times = {}
        
        # Uniform traffic specific parameters
        self.flow_rate_threshold = 0.7  # Expected consistent flow rate
        self.min_phase_splits = {       # Minimum phase splits for uniform flow
            'NS': 0.4,
            'EW': 0.4,
            'LT': 0.2  # Left turns
        }

    def _get_vehicle_data(self) -> List[Dict[str, Union[float, int]]]:
        """Collect current vehicle data from SUMO simulation."""
        vehicle_data = []
        for veh_id in traci.vehicle.getIDList():
            vehicle_data.append({
                'id': veh_id,
                'waiting_time': traci.vehicle.getWaitingTime(veh_id),
                'speed': traci.vehicle.getSpeed(veh_id),
                'position': traci.vehicle.getPosition(veh_id)
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
        """Initialize traffic light phase information."""
        for tl_id in traci.trafficlight.getIDList():
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            self.tl_phases[tl_id] = len(logic.phases)
            self.phase_start_times[tl_id] = 0

    def _get_queue_length(self, tl_id: str) -> Dict[str, int]:
        """
        Get directional queue lengths for uniform traffic.
        Returns queue lengths for different directions.
        """
        queues = {'NS': 0, 'EW': 0, 'LT': 0}
        
        for veh_id in traci.vehicle.getIDList():
            if traci.vehicle.getSpeed(veh_id) < 0.1:  # Stopped vehicle
                try:
                    # Check if vehicle is within detection range
                    if tl_id in traci.vehicle.getNextTLS(veh_id)[0][0]:
                        # Simplified direction determination
                        angle = traci.vehicle.getAngle(veh_id)
                        if 45 <= angle < 135 or 225 <= angle < 315:
                            queues['EW'] += 1
                        elif 0 <= angle < 45 or 135 <= angle < 225 or 315 <= angle <= 360:
                            queues['NS'] += 1
                        # Assume left turn for certain angles (simplified)
                        if abs(angle - traci.vehicle.getRoute(veh_id)[-1]) > 60:
                            queues['LT'] += 1
                except:
                    continue
        
        return queues

    def _should_change_phase(self, tl_id: str, current_time: int) -> bool:
        """
        Determine if phase should change based on uniform traffic patterns.
        """
        phase_duration = current_time - self.phase_start_times[tl_id]
        queues = self._get_queue_length(tl_id)
        
        # Minimum green time check
        if phase_duration < self.min_green_time:
            return False
        
        # Check if current phase has served its minimum split
        current_phase = traci.trafficlight.getPhase(tl_id)
        if current_phase % 2 == 0:  # Assuming even phases are main phases
            phase_type = 'NS' if current_phase == 0 else 'EW'
            min_time = self.min_phase_splits[phase_type] * self.max_green_time
            if phase_duration < min_time and queues[phase_type] > 0:
                return False
        
        # Queue-based decision for uniform flow
        total_queue = sum(queues.values())
        if total_queue < self.queue_threshold:
            return True
            
        # Maximum green time check
        if phase_duration >= self.max_green_time:
            return True
        
        return False

    def run(self, steps: int = 1000):
        """
        Run the simulation with uniform traffic distribution.
        """
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
            '--start',
            '--quit-on-end',
        ]
        
        traci.start(sumo_cmd, port=self.port)
        
        try:
            self._initialize_traffic_lights()
            
            for step in range(steps):
                traci.simulationStep()
                
                traffic_lights = traci.trafficlight.getIDList()
                
                # Adaptive control for uniform traffic
                for tl_id in traffic_lights:
                    if self.tl_phases[tl_id] > 0:
                        if self._should_change_phase(tl_id, step):
                            current_phase = traci.trafficlight.getPhase(tl_id)
                            next_phase = (current_phase + 1) % self.tl_phases[tl_id]
                            traci.trafficlight.setPhase(tl_id, next_phase)
                            self.phase_start_times[tl_id] = step
                
                vehicle_data = self._get_vehicle_data()
                self._update_metrics(vehicle_data)
                
        finally:
            traci.close()
            sys.stdout.flush()
            
        return self.metrics_history

if __name__ == "__main__":
    # Run uniform traffic simulation
    sim = UniformTrafficSimulation()
    metrics = sim.run(steps=1000)
    
    print("\nUniform Traffic Distribution Results:")
    print(f"Final Average Waiting Time: {metrics['average_waiting_time'][-1]:.2f} seconds")
    print(f"Final Total Throughput: {metrics['total_throughput'][-1]} vehicles")
    print(f"Final Average Speed: {metrics['average_speed'][-1]:.2f} km/h")
    print(f"Final Max Waiting Time: {metrics['max_waiting_time'][-1]:.2f} seconds")