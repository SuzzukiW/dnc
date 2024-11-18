# experiments/scenarios/traffic/special_event.py
#!/usr/bin/env python3

import os
import sys
import random
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import sumolib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class EventLocation:
    """Represents a special event location in the network"""
    
    def __init__(self,
                location_id: str,
                edge_id: str,
                capacity: int,
                parking_edges: List[str],
                access_points: List[str]):
        """
        Initialize event location
        
        Args:
            location_id: Unique identifier for the location
            edge_id: Network edge where the event is located
            capacity: Maximum number of attendees
            parking_edges: List of edges with parking facilities
            access_points: List of edges serving as entry/exit points
        """
        self.id = location_id
        self.edge_id = edge_id
        self.capacity = capacity
        self.parking_edges = parking_edges
        self.access_points = access_points
        
        # Current state
        self.current_occupancy = 0
        self.arrival_rate = 0
        self.departure_rate = 0
        
        # Track vehicles
        self.arriving_vehicles = set()
        self.parked_vehicles = set()
        self.departing_vehicles = set()

class EventType:
    """Defines characteristics of different event types"""
    
    TYPES = {
        'sports_game': {
            'arrival_pattern': 'concentrated',  # Concentrated before start
            'departure_pattern': 'mass_exodus', # Most leave right after
            'arrival_window': 2,   # Hours before event
            'departure_window': 1,  # Hours after event
            'peak_arrival_rate': 0.6,  # Peak arrival rate (fraction of capacity/hour)
            'early_arrival_ratio': 0.7,  # Fraction arriving before event
            'transport_modes': {
                'passenger': 0.70,
                'bus': 0.25,
                'bicycle': 0.05
            }
        },
        'concert': {
            'arrival_pattern': 'gradual',     # More spread out arrivals
            'departure_pattern': 'staged',     # Staged departure
            'arrival_window': 3,
            'departure_window': 1.5,
            'peak_arrival_rate': 0.4,
            'early_arrival_ratio': 0.5,
            'transport_modes': {
                'passenger': 0.75,
                'bus': 0.20,
                'bicycle': 0.05
            }
        },
        'festival': {
            'arrival_pattern': 'distributed',  # Spread throughout event
            'departure_pattern': 'continuous', # Continuous departure
            'arrival_window': 4,
            'departure_window': 4,
            'peak_arrival_rate': 0.3,
            'early_arrival_ratio': 0.3,
            'transport_modes': {
                'passenger': 0.65,
                'bus': 0.25,
                'bicycle': 0.10
            }
        },
        'exhibition': {
            'arrival_pattern': 'uniform',      # Uniform distribution
            'departure_pattern': 'balanced',   # Balanced throughout
            'arrival_window': 8,
            'departure_window': 8,
            'peak_arrival_rate': 0.2,
            'early_arrival_ratio': 0.4,
            'transport_modes': {
                'passenger': 0.60,
                'bus': 0.30,
                'bicycle': 0.10
            }
        }
    }
    
    @classmethod
    def get_config(cls, event_type: str) -> dict:
        """Get configuration for event type"""
        return cls.TYPES.get(event_type, cls.TYPES['sports_game'])

class EventScheduler:
    """Schedule and manage special events in the network"""
    
    def __init__(self):
        """Initialize event scheduler"""
        self.events = {}
        self.current_events = set()
        self.completed_events = set()
        
        # Set up logging
        self.logger = logging.getLogger('EventScheduler')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def add_event(self,
                 event_id: str,
                 location: EventLocation,
                 event_type: str,
                 start_time: int,
                 duration: int,
                 attendees: int):
        """
        Add a new event to the scheduler
        
        Args:
            event_id: Unique identifier for the event
            location: EventLocation object
            event_type: Type of event (sports_game, concert, etc.)
            start_time: Event start time (seconds)
            duration: Event duration (seconds)
            attendees: Expected number of attendees
        """
        if attendees > location.capacity:
            self.logger.warning(
                f"Event {event_id} attendees ({attendees}) exceed location "
                f"capacity ({location.capacity}). Capping at capacity.")
            attendees = location.capacity
        
        event_config = EventType.get_config(event_type)
        
        self.events[event_id] = {
            'location': location,
            'type': event_type,
            'config': event_config,
            'start_time': start_time,
            'duration': duration,
            'end_time': start_time + duration,
            'attendees': attendees,
            'arrival_window_start': start_time - event_config['arrival_window'] * 3600,
            'departure_window_end': start_time + duration + event_config['departure_window'] * 3600,
            'status': 'scheduled',
            'vehicles': set()
        }
        
        self.logger.info(f"Added event {event_id} at location {location.id}")
    
    def calculate_arrival_rate(self,
                            event_id: str,
                            current_time: int) -> float:
        """
        Calculate arrival rate for an event at current time
        
        Args:
            event_id: Event identifier
            current_time: Current simulation time
            
        Returns:
            Expected arrival rate (vehicles/hour)
        """
        event = self.events[event_id]
        config = event['config']
        
        # Get time windows
        arrival_start = event['arrival_window_start']
        event_start = event['start_time']
        
        # If outside arrival window, return 0
        if current_time < arrival_start or current_time > event_start:
            return 0.0
        
        # Calculate base rate based on pattern
        pattern = config['arrival_pattern']
        time_ratio = (current_time - arrival_start) / (event_start - arrival_start)
        
        if pattern == 'concentrated':
            # Peak closer to event start
            rate = np.exp(-(1 - time_ratio) * 2)
        elif pattern == 'gradual':
            # More spread out distribution
            rate = np.sin(time_ratio * np.pi)
        elif pattern == 'distributed':
            # Multiple smaller peaks
            rate = 0.5 + 0.5 * np.sin(time_ratio * 2 * np.pi)
        else:  # uniform
            rate = 1.0
        
        # Scale by peak rate and attendees
        max_rate = config['peak_arrival_rate'] * event['attendees']
        return rate * max_rate
    
    def calculate_departure_rate(self,
                              event_id: str,
                              current_time: int) -> float:
        """
        Calculate departure rate for an event at current time
        
        Args:
            event_id: Event identifier
            current_time: Current simulation time
            
        Returns:
            Expected departure rate (vehicles/hour)
        """
        event = self.events[event_id]
        config = event['config']
        
        # Get time windows
        event_end = event['end_time']
        departure_end = event['departure_window_end']
        
        # If outside departure window, return 0
        if current_time < event_end or current_time > departure_end:
            return 0.0
        
        # Calculate base rate based on pattern
        pattern = config['departure_pattern']
        time_ratio = (current_time - event_end) / (departure_end - event_end)
        
        if pattern == 'mass_exodus':
            # High initial rate, dropping quickly
            rate = np.exp(-time_ratio * 3)
        elif pattern == 'staged':
            # Multiple waves of departures
            rate = 0.5 + 0.5 * np.cos(time_ratio * 2 * np.pi)
        elif pattern == 'continuous':
            # Steady departure rate
            rate = 1 - time_ratio
        else:  # balanced
            rate = 1.0
        
        # Scale by remaining attendees
        remaining = len(event['location'].parked_vehicles)
        max_rate = config['peak_arrival_rate'] * remaining
        return rate * max_rate
    
    def select_parking_edge(self,
                         event_id: str,
                         preferred_access_point: str) -> str:
        """
        Select parking edge for an arriving vehicle
        
        Args:
            event_id: Event identifier
            preferred_access_point: Preferred access point edge
            
        Returns:
            Selected parking edge ID
        """
        event = self.events[event_id]
        location = event['location']
        
        # Filter available parking edges
        available_edges = [
            edge for edge in location.parking_edges
            if sum(1 for vid in location.parked_vehicles
                  if vid.startswith(f"parking_{edge}")) < 100  # Assume capacity
        ]
        
        if not available_edges:
            return random.choice(location.parking_edges)
        
        if preferred_access_point in location.access_points:
            # Select closest available parking to preferred access
            # (simplified - in practice would use network distance)
            return random.choice(available_edges)
        
        return random.choice(available_edges)
    
    def update(self, current_time: int):
        """
        Update event states
        
        Args:
            current_time: Current simulation time
        """
        # Check for events starting/ending
        for event_id, event in self.events.items():
            # Update event status
            if event['status'] == 'scheduled':
                if current_time >= event['arrival_window_start']:
                    event['status'] = 'active'
                    self.current_events.add(event_id)
                    self.logger.info(f"Event {event_id} is now active")
            
            elif event['status'] == 'active':
                if current_time >= event['departure_window_end']:
                    event['status'] = 'completed'
                    self.current_events.remove(event_id)
                    self.completed_events.add(event_id)
                    self.logger.info(f"Event {event_id} is now completed")
            
            # Update arrival/departure rates for active events
            if event_id in self.current_events:
                location = event['location']
                location.arrival_rate = self.calculate_arrival_rate(event_id, current_time)
                location.departure_rate = self.calculate_departure_rate(event_id, current_time)
    
    def get_active_events(self) -> List[str]:
        """Get list of currently active events"""
        return list(self.current_events)
    
    def get_event_status(self, event_id: str) -> dict:
        """Get current status of an event"""
        if event_id not in self.events:
            return {}
        
        event = self.events[event_id]
        location = event['location']
        
        return {
            'status': event['status'],
            'current_occupancy': len(location.parked_vehicles),
            'arriving_vehicles': len(location.arriving_vehicles),
            'departing_vehicles': len(location.departing_vehicles),
            'arrival_rate': location.arrival_rate,
            'departure_rate': location.departure_rate
        }

# Part B

class VehicleGenerator:
    """Generate vehicle trips for event attendees"""
    
    def __init__(self, net_file: str):
        """
        Initialize vehicle generator
        
        Args:
            net_file: Path to SUMO network file
        """
        self.net = sumolib.net.readNet(net_file)
        self.edges = [edge.getID() for edge in self.net.getEdges()]
        
        # Vehicle counters
        self.vehicle_count = 0
        self.bus_count = 0
        self.bicycle_count = 0
        
        # Vehicle tracking
        self.active_vehicles = set()
        self.completed_trips = set()
        self.route_info = {}
    
    def _get_distributed_origins(self,
                              num_points: int,
                              center_edge: str,
                              min_distance: float = 2000,
                              max_distance: float = 10000) -> List[str]:
        """
        Get distributed origin points for event traffic
        
        Args:
            num_points: Number of origin points needed
            center_edge: Event location edge
            min_distance: Minimum distance from event
            max_distance: Maximum distance from event
            
        Returns:
            List of edge IDs for origin points
        """
        try:
            center_edge_obj = self.net.getEdge(center_edge)
            center_x, center_y = center_edge_obj.getFromNode().getCoord()
            
            # Find edges within distance range
            valid_edges = []
            for edge in self.net.getEdges():
                x, y = edge.getFromNode().getCoord()
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                if min_distance <= distance <= max_distance:
                    valid_edges.append(edge.getID())
            
            # If not enough valid edges, expand search
            if len(valid_edges) < num_points:
                return self._get_distributed_origins(
                    num_points, center_edge,
                    min_distance * 0.8, max_distance * 1.2
                )
            
            # Select distributed points
            selected = set()
            angle_step = 2 * np.pi / num_points
            
            for i in range(num_points):
                target_angle = i * angle_step
                best_edge = None
                best_angle_diff = float('inf')
                
                for edge_id in valid_edges:
                    if edge_id not in selected:
                        edge = self.net.getEdge(edge_id)
                        x, y = edge.getFromNode().getCoord()
                        angle = np.arctan2(y - center_y, x - center_x)
                        angle_diff = abs(angle - target_angle)
                        
                        if angle_diff < best_angle_diff:
                            best_edge = edge_id
                            best_angle_diff = angle_diff
                
                if best_edge:
                    selected.add(best_edge)
            
            return list(selected)
        
        except Exception as e:
            self.logger.error(f"Error getting distributed origins: {e}")
            return random.sample(self.edges, min(num_points, len(self.edges)))
    
    def generate_event_trips(self,
                          event: dict,
                          current_time: int) -> List[dict]:
        """
        Generate trips for event attendees
        
        Args:
            event: Event configuration dictionary
            current_time: Current simulation time
            
        Returns:
            List of trip definitions
        """
        location = event['location']
        config = event['config']
        trips = []
        
        try:
            # Calculate number of new arrivals
            arrival_rate = event['location'].arrival_rate
            num_arrivals = np.random.poisson(arrival_rate / 3600)  # Convert hourly rate to per-second
            
            if num_arrivals > 0:
                # Get distributed origin points
                origins = self._get_distributed_origins(
                    num_arrivals,
                    location.edge_id
                )
                
                # Generate trips
                for i in range(num_arrivals):
                    # Select transport mode
                    mode = random.choices(
                        list(config['transport_modes'].keys()),
                        list(config['transport_modes'].values())
                    )[0]
                    
                    # Generate vehicle ID
                    if mode == 'bus':
                        vid = f"bus_{self.bus_count}"
                        self.bus_count += 1
                    elif mode == 'bicycle':
                        vid = f"bicycle_{self.bicycle_count}"
                        self.bicycle_count += 1
                    else:
                        vid = f"veh_{self.vehicle_count}"
                        self.vehicle_count += 1
                    
                    # Select origin and parking
                    origin = origins[i % len(origins)]
                    parking = location.select_parking_edge(
                        origin if len(location.parking_edges) > 1 else None
                    )
                    
                    # Create trip
                    trip = {
                        'id': vid,
                        'type': mode,
                        'depart': current_time,
                        'from': origin,
                        'to': parking,
                        'event_id': event['id'],
                        'arrival': True
                    }
                    
                    trips.append(trip)
                    self.active_vehicles.add(vid)
                    self.route_info[vid] = trip
            
            # Calculate departures
            departure_rate = location.departure_rate
            num_departures = np.random.poisson(departure_rate / 3600)
            
            if num_departures > 0:
                # Select vehicles to depart
                parked_vehicles = list(location.parked_vehicles)
                num_departures = min(num_departures, len(parked_vehicles))
                
                if num_departures > 0:
                    departing = random.sample(parked_vehicles, num_departures)
                    destinations = self._get_distributed_origins(
                        num_departures,
                        location.edge_id
                    )
                    
                    for i, vid in enumerate(departing):
                        # Create return trip
                        trip = {
                            'id': f"{vid}_return",
                            'type': self.route_info[vid]['type'],
                            'depart': current_time,
                            'from': location.get_vehicle_parking(vid),
                            'to': destinations[i % len(destinations)],
                            'event_id': event['id'],
                            'arrival': False
                        }
                        
                        trips.append(trip)
                        location.parked_vehicles.remove(vid)
                        self.completed_trips.add(vid)
            
            return trips
        
        except Exception as e:
            self.logger.error(f"Error generating event trips: {e}")
            return []

class SpecialEventGenerator:
    """Generate traffic patterns for special events"""
    
    def __init__(self,
                net_file: str,
                output_dir: str,
                random_seed: Optional[int] = None):
        """
        Initialize special event generator
        
        Args:
            net_file: Path to SUMO network file
            output_dir: Directory for output files
            random_seed: Random seed for reproducibility
        """
        self.net_file = net_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Initialize components
        self.scheduler = EventScheduler()
        self.vehicle_gen = VehicleGenerator(net_file)
        
        # Track generated data
        self.trips = []
        self.vehicles = set()
        self.statistics = defaultdict(dict)
    
    def _create_vehicle_types(self) -> ET.Element:
        """Create vehicle type definitions"""
        vtypes = ET.Element('vtypes')
        
        # Passenger car
        vtype = ET.SubElement(vtypes, 'vType')
        vtype.set('id', 'passenger')
        vtype.set('length', '5.0')
        vtype.set('maxSpeed', '50.0')
        vtype.set('accel', '2.6')
        vtype.set('decel', '4.5')
        vtype.set('sigma', '0.5')
        
        # Bus
        vtype = ET.SubElement(vtypes, 'vType')
        vtype.set('id', 'bus')
        vtype.set('length', '12.0')
        vtype.set('maxSpeed', '30.0')
        vtype.set('accel', '1.2')
        vtype.set('decel', '4.0')
        vtype.set('sigma', '0.3')
        
        # Bicycle
        vtype = ET.SubElement(vtypes, 'vType')
        vtype.set('id', 'bicycle')
        vtype.set('length', '1.8')
        vtype.set('maxSpeed', '20.0')
        vtype.set('accel', '1.0')
        vtype.set('decel', '3.0')
        vtype.set('sigma', '0.5')
        
        return vtypes
    
    def add_event(self,
                 event_id: str,
                 location: EventLocation,
                 event_type: str,
                 start_time: int,
                 duration: int,
                 attendees: int):
        """Add event to the scheduler"""
        self.scheduler.add_event(
            event_id, location, event_type,
            start_time, duration, attendees
        )
    
    def generate(self,
                simulation_duration: int,
                step_length: int = 1) -> Tuple[str, dict]:
        """
        Generate special event traffic patterns
        
        Args:
            simulation_duration: Total simulation time (seconds)
            step_length: Simulation step length (seconds)
            
        Returns:
            Tuple of (output file path, statistics)
        """
        # Create output file
        routes_file = self.output_dir / 'event_routes.rou.xml'
        root = ET.Element('routes')
        
        # Add vehicle types
        root.append(self._create_vehicle_types())
        
        # Generate trips for each time step
        current_time = 0
        while current_time < simulation_duration:
            # Update event states
            self.scheduler.update(current_time)
            
            # Generate trips for active events
            for event_id in self.scheduler.get_active_events():
                event = self.scheduler.events[event_id]
                new_trips = self.vehicle_gen.generate_event_trips(
                    event, current_time
                )
                
                # Add trips to XML
                for trip in new_trips:
                    trip_element = ET.SubElement(root, 'trip')
                    trip_element.set('id', trip['id'])
                    trip_element.set('type', trip['type'])
                    trip_element.set('depart', str(trip['depart']))
                    trip_element.set('from', trip['from'])
                    trip_element.set('to', trip['to'])
                    
                    self.trips.append(trip)
                    self.vehicles.add(trip['id'])
                    
                    # Update statistics
                    self.statistics[event_id][current_time] = \
                        self.scheduler.get_event_status(event_id)
            
            current_time += step_length
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(routes_file, encoding='utf-8', xml_declaration=True)
        
        # Save statistics
        stats_file = self.output_dir / 'event_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(self.statistics, f, indent=4)
        
        return str(routes_file), self.statistics

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate special event traffic')
    parser.add_argument('--net-file', required=True, help='Input network file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create generator
    generator = SpecialEventGenerator(
        args.net_file,
        args.output_dir,
        args.seed
    )
    
    # Add example event
    location = EventLocation(
        'stadium',
        'main_edge',
        capacity=50000,
        parking_edges=['p1', 'p2', 'p3'],
        access_points=['a1', 'a2']
    )
    
    generator.add_event(
        'game_1',
        location,
        'sports_game',
        start_time=3600,  # 1 hour into simulation
        duration=7200,    # 2 hour game
        attendees=30000
    )
    
    # Generate traffic
    routes_file, stats = generator.generate(
        simulation_duration=14400  # 4 hours
    )
    
    print(f"\nGenerated files:")
    print(f"- Routes: {routes_file}")
    print(f"- Statistics: {args.output_dir}/event_statistics.json")
    
    print("\nEvent Statistics:")
    for event_id, timesteps in stats.items():
        print(f"\nEvent: {event_id}")
        peak_occupancy = max(
            status['current_occupancy']
            for status in timesteps.values()
        )
        print(f"Peak occupancy: {peak_occupancy}")

if __name__ == "__main__":
    main()