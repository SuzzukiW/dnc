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

class EventType(Enum):
    SPORTS = "sports"          # Fenway, TD Garden events
    CONCERT = "concert"        # Arena events, outdoor concerts
    GRADUATION = "graduation"  # University events
    PARADE = "parade"         # Marathons, victory parades
    CONVENTION = "convention" # Convention center events

class EventPhase(Enum):
    PRE_EVENT = "pre"    # Building crowd
    DURING = "during"    # Event in progress
    POST_EVENT = "post"  # Mass exodus

class SpecialEventSimulation:
    def __init__(self, 
                 net_file: str = 'baseline/osm.net.xml.gz',
                 route_file: str = 'baseline/osm.passenger.trips.xml',
                 port: int = 8813,
                 event_type: EventType = EventType.SPORTS,
                 event_phase: EventPhase = EventPhase.PRE_EVENT):
        """
        Initialize adaptive controller for special event traffic.
        Handles different types of events and their phases.
        """
        self.net_file = net_file
        self.route_file = route_file
        self.port = port
        self.event_type = event_type
        self.event_phase = event_phase
        
        # Event venue locations (simplified - would need actual coordinates)
        self.venue_locations = {
            EventType.SPORTS: {
                'fenway': (42.3467, -71.0972),
                'tdgarden': (42.3662, -71.0621)
            },
            EventType.CONCERT: {
                'tdgarden': (42.3662, -71.0621),
                'pavilion': (42.3543, -71.0444)
            },
            EventType.GRADUATION: {
                'bu': (42.3505, -71.1054),
                'harvard': (42.3744, -71.1182),
                'mit': (42.3601, -71.0942)
            },
            EventType.CONVENTION: {
                'hynes': (42.3486, -71.0827),
                'bcec': (42.3449, -71.0444)
            }
        }
        
        # Phase-specific parameters
        self.phase_params = {
            EventPhase.PRE_EVENT: {
                'min_green': 25,
                'max_green': 75,
                'queue_weight': 1.2,  # Weight more towards venue
            },
            EventPhase.DURING: {
                'min_green': 20,
                'max_green': 45,
                'queue_weight': 1.0,  # Normal operations
            },
            EventPhase.POST_EVENT: {
                'min_green': 30,
                'max_green': 90,
                'queue_weight': 1.5,  # Heavy weight to clear venue area
            }
        }
        
        # Set current phase parameters
        self.current_params = self.phase_params[event_phase]
        self.min_green_time = self.current_params['min_green']
        self.max_green_time = self.current_params['max_green']
        self.queue_weight = self.current_params['queue_weight']
        
        # Event-specific settings
        self.yellow_time = 4
        self.detection_range = 150  # Extended for event crowds
        self.venue_radius = 500  # meters around venue
        
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
        self.venue_area_lights = set()  # TLs near venue
        
        # Pedestrian consideration
        self.ped_crossing_time = 15  # Minimum for large crowds
        self.max_ped_wait = 60      # Maximum wait for pedestrians

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

    def _identify_venue_area_lights(self):
        """Identify traffic lights in venue area."""
        venue_coords = self.venue_locations[self.event_type]
        for tl_id in traci.trafficlight.getIDList():
            try:
                # Get traffic light position
                tl_pos = traci.junction.getPosition(
                    traci.trafficlight.getControlledJunctions(tl_id)[0]
                )
                
                # Check if within venue radius
                for venue_pos in venue_coords.values():
                    dist = np.sqrt(
                        (tl_pos[0] - venue_pos[0])**2 + 
                        (tl_pos[1] - venue_pos[1])**2
                    )
                    if dist <= self.venue_radius:
                        self.venue_area_lights.add(tl_id)
                        break
            except:
                continue

    def _initialize_traffic_lights(self):
        """Initialize traffic light phase information."""
        for tl_id in traci.trafficlight.getIDList():
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            self.tl_phases[tl_id] = len(logic.phases)
            self.phase_start_times[tl_id] = 0
        
        self._identify_venue_area_lights()

    def _get_queue_length(self, tl_id: str) -> Dict[str, float]:
        """
        Get weighted queue lengths based on event phase and direction.
        """
        queues = {'venue': 0, 'other': 0}
        is_venue_area = tl_id in self.venue_area_lights
        
        for veh_id in traci.vehicle.getIDList():
            if traci.vehicle.getSpeed(veh_id) < 0.1:  # Stopped vehicle
                try:
                    if tl_id in traci.vehicle.getNextTLS(veh_id)[0][0]:
                        # Apply venue area weighting
                        if is_venue_area:
                            queues['venue'] += 1 * self.queue_weight
                        else:
                            queues['other'] += 1
                except:
                    continue
        
        return queues

    def _adjust_for_pedestrians(self, tl_id: str, current_time: int) -> bool:
        """
        Adjust timing for heavy pedestrian volumes during events.
        """
        is_venue_area = tl_id in self.venue_area_lights
        phase_duration = current_time - self.phase_start_times[tl_id]
        
        if is_venue_area:
            # Ensure minimum pedestrian crossing time
            if phase_duration < self.ped_crossing_time:
                return False
            
            # Force phase change if pedestrians waiting too long
            if phase_duration > self.max_ped_wait:
                return True
        
        return None

    def _should_change_phase(self, tl_id: str, current_time: int) -> bool:
        """
        Determine if phase should change based on event conditions.
        """
        phase_duration = current_time - self.phase_start_times[tl_id]
        queues = self._get_queue_length(tl_id)
        
        # Check pedestrian requirements first
        ped_decision = self._adjust_for_pedestrians(tl_id, current_time)
        if ped_decision is not None:
            return ped_decision
        
        # Minimum green time check
        if phase_duration < self.min_green_time:
            return False
        
        # Queue-based decision with event considerations
        if tl_id in self.venue_area_lights:
            # More patient with venue area queues during events
            if queues['venue'] > 0 and phase_duration < self.max_green_time * 0.8:
                return False
            
            # More aggressive for other directions to prevent gridlock
            if queues['other'] > queues['venue'] * 1.5:
                return True
        else:
            # Normal queue check for non-venue areas
            if sum(queues.values()) < self.current_params['queue_weight'] * 5:
                return True
        
        # Maximum green time check
        if phase_duration >= self.max_green_time:
            return True
        
        return False

    def run(self, steps: int = 1000):
        """
        Run the special event simulation.
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
                
                # Handle all traffic lights
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
    # Run simulation for different event types and phases
    events = [
        (EventType.SPORTS, "Red Sox Game"),
        (EventType.CONCERT, "TD Garden Concert"),
        (EventType.GRADUATION, "University Graduation"),
        (EventType.CONVENTION, "Convention Center Event")
    ]
    
    for event_type, event_name in events:
        print(f"\nSimulating {event_name}:")
        for phase in EventPhase:
            print(f"\nEvent Phase: {phase.value}")
            sim = SpecialEventSimulation(
                event_type=event_type,
                event_phase=phase
            )
            metrics = sim.run(steps=1000)
            
            print(f"Results:")
            print(f"Average Waiting Time: {metrics['average_waiting_time'][-1]:.2f} seconds")
            print(f"Total Throughput: {metrics['total_throughput'][-1]} vehicles")
            print(f"Average Speed: {metrics['average_speed'][-1]:.2f} km/h")
            print(f"Max Waiting Time: {metrics['max_waiting_time'][-1]:.2f} seconds")