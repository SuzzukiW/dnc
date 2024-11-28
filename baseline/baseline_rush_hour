#!/usr/bin/env python

import os
import sys
import traci
import numpy as np
from typing import List, Dict, Union
from pathlib import Path
from enum import Enum

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

class RushHourPeriod(Enum):
    MORNING = "morning"    # 7-9:30 AM, heavy inbound traffic
    EVENING = "evening"    # 4-6:30 PM, heavy outbound traffic

class RushHourSimulation:
    def __init__(self, 
                 net_file: str = 'baseline/osm.net.xml.gz',
                 route_file: str = 'baseline/osm.passenger.trips.xml',
                 port: int = 8813,
                 rush_hour: RushHourPeriod = RushHourPeriod.MORNING):
        """
        Initialize adaptive controller for rush hour traffic.
        Handles both morning (inbound) and evening (outbound) peaks.
        """
        self.net_file = net_file
        self.route_file = route_file
        self.port = port
        self.rush_hour = rush_hour
        
        # Rush hour specific parameters
        self.rush_hour_params = {
            RushHourPeriod.MORNING: {
                'min_green_main': 30,    # Longer minimum for main routes
                'min_green_side': 15,    # Shorter for side streets
                'max_green_main': 90,    # Very long maximum for main routes
                'max_green_side': 30,    # Short maximum for side streets
                'queue_threshold_main': 20,  # Higher threshold for main routes
                'queue_threshold_side': 8,   # Lower for side streets
            },
            RushHourPeriod.EVENING: {
                'min_green_main': 35,    # Even longer for evening rush
                'min_green_side': 15,
                'max_green_main': 100,   # Very long for outbound traffic
                'max_green_side': 25,
                'queue_threshold_main': 25,
                'queue_threshold_side': 6,
            }
        }
        
        # Set parameters based on rush hour period
        params = self.rush_hour_params[rush_hour]
        self.min_green_main = params['min_green_main']
        self.min_green_side = params['min_green_side']
        self.max_green_main = params['max_green_main']
        self.max_green_side = params['max_green_side']
        self.queue_threshold_main = params['queue_threshold_main']
        self.queue_threshold_side = params['queue_threshold_side']
        
        self.yellow_time = 4
        self.detection_range = 100  # Longer detection for rush hour
        
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
        self.main_corridors = set()  # Will store IDs of main corridor lights
        
        # Heavy direction preferences (based on rush hour)
        self.heavy_direction = 'inbound' if rush_hour == RushHourPeriod.MORNING else 'outbound'

    def _get_vehicle_data(self) -> List[Dict[str, Union[float, int]]]:
        """Collect current vehicle data from SUMO simulation."""
        vehicle_data = []
        for veh_id in traci.vehicle.getIDList():
            vehicle_data.append({
                'id': veh_id,
                'waiting_time': traci.vehicle.getWaitingTime(veh_id),
                'speed': traci.vehicle.getSpeed(veh_id),
                'position': traci.vehicle.getPosition(veh_id),
                'route': traci.vehicle.getRoute(veh_id)
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

    def _identify_main_corridors(self):
        """
        Identify main traffic corridors based on rush hour direction.
        For Boston, identifies major arteries like Mass Ave, Comm Ave, etc.
        """
        # This is a simplified version - in practice, would need actual corridor data
        for tl_id in traci.trafficlight.getIDList():
            try:
                # Basic check for major intersections
                num_lanes = len(traci.trafficlight.getControlledLanes(tl_id))
                if num_lanes > 4:  # Assuming larger intersections are on main corridors
                    self.main_corridors.add(tl_id)
            except:
                continue

    def _initialize_traffic_lights(self):
        """Initialize traffic light phase information."""
        for tl_id in traci.trafficlight.getIDList():
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            self.tl_phases[tl_id] = len(logic.phases)
            self.phase_start_times[tl_id] = 0
        
        self._identify_main_corridors()

    def _get_queue_length(self, tl_id: str) -> Dict[str, int]:
        """
        Get directional queue lengths with rush hour consideration.
        Weighs queues differently based on direction relative to rush hour flow.
        """
        queues = {'main': 0, 'side': 0}
        
        for veh_id in traci.vehicle.getIDList():
            if traci.vehicle.getSpeed(veh_id) < 0.1:  # Stopped vehicle
                try:
                    if tl_id in traci.vehicle.getNextTLS(veh_id)[0][0]:
                        # Determine if vehicle is on main corridor
                        if tl_id in self.main_corridors:
                            queues['main'] += 1
                        else:
                            queues['side'] += 1
                except:
                    continue
        
        return queues

    def _should_change_phase(self, tl_id: str, current_time: int) -> bool:
        """
        Determine if phase should change based on rush hour patterns.
        Prioritizes main corridor movement during peak periods.
        """
        phase_duration = current_time - self.phase_start_times[tl_id]
        queues = self._get_queue_length(tl_id)
        
        # Check if this is a main corridor light
        is_main_corridor = tl_id in self.main_corridors
        
        # Get appropriate thresholds based on corridor type
        min_green = self.min_green_main if is_main_corridor else self.min_green_side
        max_green = self.max_green_main if is_main_corridor else self.max_green_side
        queue_threshold = self.queue_threshold_main if is_main_corridor else self.queue_threshold_side
        
        # Minimum green time check
        if phase_duration < min_green:
            return False
        
        # Queue-based decision
        if is_main_corridor:
            # For main corridors, be more patient during rush hour
            if queues['main'] > queue_threshold:
                return False
        else:
            # For side streets, be more aggressive about changing
            if queues['side'] < queue_threshold:
                return True
        
        # Maximum green time check
        if phase_duration >= max_green:
            return True
        
        return False

    def run(self, steps: int = 1000):
        """
        Run the rush hour simulation.
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
                
                # Rush hour adaptive control
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
    # Run both morning and evening rush hour simulations
    for period in RushHourPeriod:
        print(f"\nRunning {period.value} rush hour simulation...")
        sim = RushHourSimulation(rush_hour=period)
        metrics = sim.run(steps=1000)
        
        print(f"\n{period.value.capitalize()} Rush Hour Results:")
        print(f"Final Average Waiting Time: {metrics['average_waiting_time'][-1]:.2f} seconds")
        print(f"Final Total Throughput: {metrics['total_throughput'][-1]} vehicles")
        print(f"Final Average Speed: {metrics['average_speed'][-1]:.2f} km/h")
        print(f"Final Max Waiting Time: {metrics['max_waiting_time'][-1]:.2f} seconds")