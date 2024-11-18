# experiments/scenarios/traffic/rush_hour.py
#!/usr/bin/env python3

import os
import sys
import random
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class RushHourGenerator:
    """Generate rush hour traffic patterns for SUMO simulation"""
    
    def __init__(self,
                simulation_start=0,
                simulation_end=86400,  # 24 hours in seconds
                morning_peak_start=25200,  # 7:00 AM
                morning_peak_end=32400,   # 9:00 AM
                evening_peak_start=57600,  # 4:00 PM
                evening_peak_end=64800,   # 6:00 PM
                base_flow=300,            # vehicles per hour
                peak_flow=1200,           # vehicles per hour during peak
                random_seed=42):
        
        self.simulation_start = simulation_start
        self.simulation_end = simulation_end
        self.morning_peak_start = morning_peak_start
        self.morning_peak_end = morning_peak_end
        self.evening_peak_start = evening_peak_start
        self.evening_peak_end = evening_peak_end
        self.base_flow = base_flow
        self.peak_flow = peak_flow
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Time windows for different periods
        self.time_windows = [
            (simulation_start, morning_peak_start, 'night'),
            (morning_peak_start, morning_peak_end, 'morning_peak'),
            (morning_peak_end, evening_peak_start, 'day'),
            (evening_peak_start, evening_peak_end, 'evening_peak'),
            (evening_peak_end, simulation_end, 'night')
        ]
    
    def _get_flow_rate(self, time, period):
        """Calculate flow rate based on time and period"""
        if period == 'night':
            # Reduced traffic during night (20-30% of base flow)
            return self.base_flow * random.uniform(0.2, 0.3)
        elif period == 'day':
            # Normal traffic during day (80-120% of base flow)
            return self.base_flow * random.uniform(0.8, 1.2)
        elif period in ['morning_peak', 'evening_peak']:
            # Peak traffic with gaussian distribution
            if period == 'morning_peak':
                peak_center = self.morning_peak_start + (self.morning_peak_end - self.morning_peak_start) / 2
                sigma = (self.morning_peak_end - self.morning_peak_start) / 6
            else:
                peak_center = self.evening_peak_start + (self.evening_peak_end - self.evening_peak_start) / 2
                sigma = (self.evening_peak_end - self.evening_peak_start) / 6
            
            # Calculate gaussian factor (0-1)
            gaussian_factor = np.exp(-((time - peak_center) ** 2) / (2 * sigma ** 2))
            
            # Scale between base_flow and peak_flow
            return self.base_flow + (self.peak_flow - self.base_flow) * gaussian_factor
        
        return self.base_flow
    
    def _get_vehicle_type_distribution(self, time):
        """Get vehicle type distribution based on time"""
        # Morning peak: more passenger vehicles
        if self.morning_peak_start <= time <= self.morning_peak_end:
            return {
                'passenger': 0.75,
                'bus': 0.15,
                'truck': 0.10
            }
        # Evening peak: mixed traffic
        elif self.evening_peak_start <= time <= self.evening_peak_end:
            return {
                'passenger': 0.70,
                'bus': 0.20,
                'truck': 0.10
            }
        # Night: more trucks
        elif time < self.morning_peak_start or time > self.evening_peak_end:
            return {
                'passenger': 0.50,
                'bus': 0.10,
                'truck': 0.40
            }
        # Regular day
        else:
            return {
                'passenger': 0.60,
                'bus': 0.25,
                'truck': 0.15
            }
    
    def generate_trips(self, net_file, output_file, edges=None):
        """Generate trip definitions with rush hour patterns"""
        if edges is None:
            # If no edges provided, read from network file
            edges = self._get_network_edges(net_file)
        
        # Create trips XML
        root = ET.Element('routes')
        
        # Add vehicle types
        self._add_vehicle_types(root)
        
        # Generate trips for each time window
        trips = []
        for start_time, end_time, period in self.time_windows:
            current_time = start_time
            while current_time < end_time:
                # Calculate flow rate for current time
                flow_rate = self._get_flow_rate(current_time, period)
                
                # Calculate number of vehicles to generate in this second
                vehicles_per_second = flow_rate / 3600  # Convert from vehicles/hour to vehicles/second
                
                # Use Poisson distribution to determine if we generate a vehicle
                if random.random() < vehicles_per_second:
                    # Get vehicle type based on time distribution
                    vtype = self._get_random_vehicle_type(current_time)
                    
                    # Select random source and destination edges
                    source = random.choice(edges)
                    dest = random.choice([e for e in edges if e != source])
                    
                    # Create trip element
                    trip = ET.SubElement(root, 'trip')
                    trip.set('id', f'veh_{len(trips)}')
                    trip.set('type', vtype)
                    trip.set('depart', str(current_time))
                    trip.set('from', source)
                    trip.set('to', dest)
                    
                    trips.append(trip)
                
                current_time += 1
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        
        print(f"Generated {len(trips)} trips")
        return len(trips)
    
    def _get_network_edges(self, net_file):
        """Extract edge IDs from network file"""
        tree = ET.parse(net_file)
        root = tree.getroot()
        
        # Get all edge elements that aren't internal edges
        edges = [edge.get('id') for edge in root.findall('.//edge')
                if not edge.get('id').startswith(':')]
        
        return edges
    
    def _add_vehicle_types(self, root):
        """Add vehicle type definitions to routes"""
        # Passenger car
        vtype = ET.SubElement(root, 'vType')
        vtype.set('id', 'passenger')
        vtype.set('length', '5.0')
        vtype.set('maxSpeed', '50.0')
        vtype.set('accel', '2.6')
        vtype.set('decel', '4.5')
        vtype.set('sigma', '0.5')
        
        # Bus
        vtype = ET.SubElement(root, 'vType')
        vtype.set('id', 'bus')
        vtype.set('length', '12.0')
        vtype.set('maxSpeed', '30.0')
        vtype.set('accel', '1.2')
        vtype.set('decel', '4.0')
        vtype.set('sigma', '0.3')
        
        # Truck
        vtype = ET.SubElement(root, 'vType')
        vtype.set('id', 'truck')
        vtype.set('length', '7.5')
        vtype.set('maxSpeed', '25.0')
        vtype.set('accel', '1.0')
        vtype.set('decel', '4.0')
        vtype.set('sigma', '0.4')
    
    def _get_random_vehicle_type(self, time):
        """Get random vehicle type based on time distribution"""
        distribution = self._get_vehicle_type_distribution(time)
        r = random.random()
        cumsum = 0
        for vtype, prob in distribution.items():
            cumsum += prob
            if r <= cumsum:
                return vtype
        return 'passenger'  # fallback

def main():
    """Main function to generate rush hour traffic"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate rush hour traffic patterns')
    parser.add_argument('--net-file', required=True, help='Input network file')
    parser.add_argument('--output', required=True, help='Output route file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--base-flow', type=int, default=300,
                      help='Base flow rate (vehicles/hour)')
    parser.add_argument('--peak-flow', type=int, default=1200,
                      help='Peak flow rate (vehicles/hour)')
    
    args = parser.parse_args()
    
    generator = RushHourGenerator(
        base_flow=args.base_flow,
        peak_flow=args.peak_flow,
        random_seed=args.seed
    )
    
    print(f"Generating rush hour traffic patterns...")
    print(f"Base flow: {args.base_flow} vehicles/hour")
    print(f"Peak flow: {args.peak_flow} vehicles/hour")
    
    num_trips = generator.generate_trips(args.net_file, args.output)
    
    print(f"\nGenerated {num_trips} trips")
    print(f"Output written to: {args.output}")

if __name__ == "__main__":
    main()