#!/usr/bin/env python

import os
import sys
import traci
import numpy as np
from typing import List, Dict, Union, Set, Tuple
from pathlib import Path
from enum import Enum

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

class IncidentType(Enum):
    ACCIDENT = "accident"        # Temporary blockage
    ROAD_CLOSURE = "closure"     # Complete closure
    CONSTRUCTION = "construction"# Partial lane closure

class IncidentSimulation:
    def __init__(self, 
                 net_file: str = 'baseline/osm.net.xml.gz',
                 route_file: str = 'baseline/osm.passenger.trips.xml',
                 port: int = 8813):
        """
        Initialize adaptive controller for incident-affected traffic.
        Handles accidents, road closures, and construction scenarios.
        """
        self.net_file = net_file
        self.route_file = route_file
        self.port = port
        
        # Incident-specific parameters
        self.min_green_time = 20
        self.max_green_time = 60
        self.yellow_time = 4
        self.detection_range = 120  # Longer detection for incident backup
        
        # Incident thresholds
        self.congestion_threshold = 0.8  # Speed ratio indicating congestion
        self.queue_threshold = 15
        self.backup_threshold = 200  # meters of backup to trigger adaptation
        
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
        
        # Incident tracking
        self.incident_locations: Set[str] = set()  # Affected lanes
        self.affected_intersections: Dict[str, Dict] = {}  # TL IDs with incident info
        self.detour_routes: Dict[str, List[str]] = {}  # Alternative routes
        
        # Speed monitoring for incident detection
        self.normal_speeds: Dict[str, float] = {}  # Baseline speeds per edge
        self.speed_history: Dict[str, List[float]] = {}  # Recent speed readings

    def _get_vehicle_data(self) -> List[Dict[str, Union[float, int]]]:
        """Collect current vehicle data from SUMO simulation."""
        vehicle_data = []
        for veh_id in traci.vehicle.getIDList():
            vehicle_data.append({
                'id': veh_id,
                'waiting_time': traci.vehicle.getWaitingTime(veh_id),
                'speed': traci.vehicle.getSpeed(veh_id),
                'position': traci.vehicle.getPosition(veh_id),
                'lane': traci.vehicle.getLaneID(veh_id),
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

    def _initialize_traffic_lights(self):
        """Initialize traffic light phase information."""
        for tl_id in traci.trafficlight.getIDList():
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            self.tl_phases[tl_id] = len(logic.phases)
            self.phase_start_times[tl_id] = 0

    def _detect_incidents(self, vehicle_data: List[Dict]) -> List[Tuple[str, IncidentType]]:
        """
        Detect potential incidents based on traffic patterns.
        Returns list of (location, incident_type) tuples.
        """
        incidents = []
        lane_speeds = {}
        
        # Calculate average speeds per lane
        for veh in vehicle_data:
            lane = veh['lane']
            if lane not in lane_speeds:
                lane_speeds[lane] = []
            lane_speeds[lane].append(veh['speed'])
        
        # Check for incident patterns
        for lane, speeds in lane_speeds.items():
            avg_speed = np.mean(speeds) if speeds else 0
            
            # Update speed history
            if lane not in self.speed_history:
                self.speed_history[lane] = []
            self.speed_history[lane].append(avg_speed)
            if len(self.speed_history[lane]) > 60:  # Keep last minute
                self.speed_history[lane].pop(0)
            
            # Detect incidents based on speed patterns
            if len(self.speed_history[lane]) >= 30:  # Need 30s of data
                recent_avg = np.mean(self.speed_history[lane][-30:])
                if lane not in self.normal_speeds:
                    self.normal_speeds[lane] = np.mean(self.speed_history[lane])
                
                speed_ratio = recent_avg / self.normal_speeds[lane]
                
                if speed_ratio < 0.2:  # Severe slowdown
                    incidents.append((lane, IncidentType.ACCIDENT))
                elif speed_ratio < 0.4:  # Major slowdown
                    incidents.append((lane, IncidentType.CONSTRUCTION))
        
        return incidents

    def _identify_affected_intersections(self, incidents: List[Tuple[str, IncidentType]]):
        """Identify intersections affected by incidents."""
        self.affected_intersections.clear()
        
        for lane, incident_type in incidents:
            # Find nearby traffic lights
            try:
                tls = traci.lane.getClosingTrafficLights(lane)
                for tl_id in tls:
                    self.affected_intersections[tl_id] = {
                        'incident_type': incident_type,
                        'affected_lane': lane,
                        'distance': 0  # Could calculate actual distance
                    }
            except:
                continue

    def _adjust_timings_for_incident(self, tl_id: str) -> Dict:
        """
        Get adjusted timing parameters based on incident type and location.
        """
        if tl_id not in self.affected_intersections:
            return {
                'min_green': self.min_green_time,
                'max_green': self.max_green_time,
                'queue_threshold': self.queue_threshold
            }
        
        incident = self.affected_intersections[tl_id]
        
        if incident['incident_type'] == IncidentType.ACCIDENT:
            return {
                'min_green': int(self.min_green_time * 1.5),  # Longer minimum
                'max_green': int(self.max_green_time * 1.5),  # Longer maximum
                'queue_threshold': int(self.queue_threshold * 0.7)  # More sensitive
            }
        elif incident['incident_type'] == IncidentType.CONSTRUCTION:
            return {
                'min_green': int(self.min_green_time * 1.3),
                'max_green': int(self.max_green_time * 1.3),
                'queue_threshold': int(self.queue_threshold * 0.8)
            }
        
        return {
            'min_green': self.min_green_time,
            'max_green': self.max_green_time,
            'queue_threshold': self.queue_threshold
        }

    def _get_queue_length(self, tl_id: str) -> Dict[str, int]:
        """
        Get directional queue lengths with incident consideration.
        """
        queues = {'affected': 0, 'other': 0}
        
        # Get affected approaches if this intersection is impacted
        affected_lane = None
        if tl_id in self.affected_intersections:
            affected_lane = self.affected_intersections[tl_id]['affected_lane']
        
        for veh_id in traci.vehicle.getIDList():
            if traci.vehicle.getSpeed(veh_id) < 0.1:  # Stopped vehicle
                try:
                    if tl_id in traci.vehicle.getNextTLS(veh_id)[0][0]:
                        lane = traci.vehicle.getLaneID(veh_id)
                        if lane == affected_lane:
                            queues['affected'] += 1
                        else:
                            queues['other'] += 1
                except:
                    continue
        
        return queues

    def _should_change_phase(self, tl_id: str, current_time: int) -> bool:
        """
        Determine if phase should change based on incident conditions.
        """
        phase_duration = current_time - self.phase_start_times[tl_id]
        queues = self._get_queue_length(tl_id)
        
        # Get adjusted timings based on incident impact
        timings = self._adjust_timings_for_incident(tl_id)
        
        # Minimum green time check
        if phase_duration < timings['min_green']:
            return False
        
        # Queue-based decision
        if tl_id in self.affected_intersections:
            # More aggressive changes if queues building on non-affected approaches
            if queues['other'] > timings['queue_threshold']:
                return True
            # More patient with affected approaches
            if queues['affected'] > timings['queue_threshold'] * 1.5:
                return False
        else:
            # Normal queue check for unaffected intersections
            if sum(queues.values()) < timings['queue_threshold']:
                return True
        
        # Maximum green time check
        if phase_duration >= timings['max_green']:
            return True
        
        return False

    def run(self, steps: int = 1000):
        """
        Run the incident-affected simulation.
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
                
                # Collect vehicle data
                vehicle_data = self._get_vehicle_data()
                
                # Detect incidents and identify affected intersections
                incidents = self._detect_incidents(vehicle_data)
                self._identify_affected_intersections(incidents)
                
                # Adaptive control for all traffic lights
                for tl_id in traci.trafficlight.getIDList():
                    if self.tl_phases[tl_id] > 0:
                        if self._should_change_phase(tl_id, step):
                            current_phase = traci.trafficlight.getPhase(tl_id)
                            next_phase = (current_phase + 1) % self.tl_phases[tl_id]
                            traci.trafficlight.setPhase(tl_id, next_phase)
                            self.phase_start_times[tl_id] = step
                
                self._update_metrics(vehicle_data)
                
        finally:
            traci.close()
            sys.stdout.flush()
            
        return self.metrics_history

if __name__ == "__main__":
    # Run incident simulation
    sim = IncidentSimulation()
    metrics = sim.run(steps=1000)
    
    print("\nIncident-affected Simulation Results:")
    print(f"Final Average Waiting Time: {metrics['average_waiting_time'][-1]:.2f} seconds")
    print(f"Final Total Throughput: {metrics['total_throughput'][-1]} vehicles")
    print(f"Final Average Speed: {metrics['average_speed'][-1]:.2f} km/h")
    print(f"Final Max Waiting Time: {metrics['max_waiting_time'][-1]:.2f} seconds")