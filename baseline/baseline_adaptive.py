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

class BaselineAdaptiveSimulation:
    def __init__(self, 
                 net_file: str = 'baseline/osm.net.xml.gz',
                 route_file: str = 'baseline/osm.passenger.trips.xml',
                 port: int = 8813):
        """
        Initialize an adaptive traffic light controller for Greater Boston area.
        Uses basic traffic-responsive rules across the entire network.
        """
        self.net_file = net_file
        self.route_file = route_file
        self.port = port
        
        # Basic adaptive control parameters
        self.min_green_time = 20    # Minimum for pedestrian crossings
        self.max_green_time = 60    # Maximum to prevent excessive waiting
        self.yellow_time = 4        # Standard yellow time
        self.queue_threshold = 12    # Queue length threshold
        self.detection_range = 100   # Standard detection range
        
        # Additional adaptive thresholds
        self.congestion_threshold = 0.3  # Speed ratio indicating congestion
        self.waiting_time_threshold = 45  # Maximum waiting time threshold
        
        # Store metrics for evaluation
        self.metrics_history = {
            'average_waiting_time': [],
            'total_throughput': [],
            'average_speed': [],
            'max_waiting_time': []
        }
        
        # Traffic light information
        self.tl_phases = {}
        self.phase_start_times = {}
        self.phase_durations = {}

    def _get_vehicle_data(self) -> List[Dict[str, Union[float, int]]]:
        """Collect current vehicle data from SUMO simulation."""
        vehicle_data = []
        for veh_id in traci.vehicle.getIDList():
            vehicle_data.append({
                'id': veh_id,
                'waiting_time': traci.vehicle.getWaitingTime(veh_id),
                'speed': traci.vehicle.getSpeed(veh_id),
                'lane': traci.vehicle.getLaneID(veh_id)
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
            self.phase_durations[tl_id] = self.min_green_time

    def _get_intersection_state(self, tl_id: str) -> Dict[str, float]:
        """
        Get current state of intersection including queues and speeds.
        """
        state = {'queue_length': 0, 'avg_speed': 0, 'max_wait': 0}
        vehicles_counted = 0
        
        for veh_id in traci.vehicle.getIDList():
            try:
                if tl_id in traci.vehicle.getNextTLS(veh_id)[0][0]:
                    # Count queued vehicles
                    if traci.vehicle.getSpeed(veh_id) < 0.1:
                        state['queue_length'] += 1
                    
                    # Track speeds and waiting times
                    state['avg_speed'] += traci.vehicle.getSpeed(veh_id)
                    state['max_wait'] = max(
                        state['max_wait'],
                        traci.vehicle.getWaitingTime(veh_id)
                    )
                    vehicles_counted += 1
            except:
                continue
        
        if vehicles_counted > 0:
            state['avg_speed'] /= vehicles_counted
        
        return state

    def _should_change_phase(self, tl_id: str, current_time: int) -> bool:
        """
        Determine if phase should change based on current traffic conditions.
        """
        phase_duration = current_time - self.phase_start_times[tl_id]
        state = self._get_intersection_state(tl_id)
        
        # Minimum green time check
        if phase_duration < self.min_green_time:
            return False
        
        # Congestion-based decision
        if state['avg_speed'] > 0:
            if state['avg_speed'] < self.congestion_threshold:
                if state['queue_length'] < self.queue_threshold:
                    return True
        
        # Queue-based decision
        if state['queue_length'] < self.queue_threshold and phase_duration >= self.min_green_time:
            return True
        
        # Waiting time check
        if state['max_wait'] > self.waiting_time_threshold:
            return True
        
        # Maximum green time check
        if phase_duration >= self.max_green_time:
            return True
        
        return False

    def run(self, steps: int = 1000):
        """
        Run the adaptive simulation for the entire Greater Boston network.
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
            '--start',
            '--quit-on-end',
        ]
        
        traci.start(sumo_cmd, port=self.port)
        
        try:
            self._initialize_traffic_lights()
            
            for step in range(steps):
                traci.simulationStep()
                
                # Adaptive control for all traffic lights
                for tl_id in traci.trafficlight.getIDList():
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
    # Run adaptive baseline for entire Greater Boston area
    sim = BaselineAdaptiveSimulation()
    metrics = sim.run(steps=1000)
    
    print("\nBaseline Adaptive Results (Greater Boston Area):")
    print(f"Final Average Waiting Time: {metrics['average_waiting_time'][-1]:.2f} seconds")
    print(f"Final Total Throughput: {metrics['total_throughput'][-1]} vehicles")
    print(f"Final Average Speed: {metrics['average_speed'][-1]:.2f} km/h")
    print(f"Final Max Waiting Time: {metrics['max_waiting_time'][-1]:.2f} seconds")