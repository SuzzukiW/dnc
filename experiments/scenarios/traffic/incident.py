# experiments/scenarios/traffic/incident.py
#!/usr/bin/env python3

import os
import sys
import random
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import sumolib
import traci

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class IncidentGenerator:
    """Generate traffic incidents (accidents, road closures) for SUMO simulation"""
    
    INCIDENT_TYPES = {
        'accident': {
            'duration': (900, 3600),    # 15min - 1h
            'lanes_affected': (1, 2),    # 1-2 lanes
            'speed_reduction': 0.0       # Complete blockage
        },
        'construction': {
            'duration': (7200, 28800),   # 2h - 8h
            'lanes_affected': (1, 1),     # Usually 1 lane
            'speed_reduction': 0.5        # 50% speed reduction
        },
        'broken_vehicle': {
            'duration': (600, 1800),     # 10min - 30min
            'lanes_affected': (1, 1),     # 1 lane
            'speed_reduction': 0.0        # Complete blockage
        },
        'weather_hazard': {
            'duration': (3600, 7200),    # 1h - 2h
            'lanes_affected': (0, 0),     # All lanes
            'speed_reduction': 0.7        # 30% speed reduction
        }
    }
    
    def __init__(self,
                simulation_start=0,
                simulation_end=86400,     # 24 hours in seconds
                num_incidents=5,          # Number of incidents to generate
                min_incident_spacing=1800, # Minimum time between incidents
                random_seed=42):
        
        self.simulation_start = simulation_start
        self.simulation_end = simulation_end
        self.num_incidents = num_incidents
        self.min_incident_spacing = min_incident_spacing
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Store generated incidents
        self.incidents = []
        
    def _get_critical_edges(self, net_file):
        """Identify critical edges in the network based on connectivity and centrality"""
        net = sumolib.net.readNet(net_file)
        
        # Create a simplified graph representation
        graph = {}
        edge_lengths = {}
        
        for edge in net.getEdges():
            from_node = edge.getFromNode().getID()
            to_node = edge.getToNode().getID()
            length = edge.getLength()
            
            if from_node not in graph:
                graph[from_node] = []
            graph[from_node].append(to_node)
            
            edge_lengths[edge.getID()] = length
        
        # Calculate edge betweenness centrality (simplified version)
        betweenness = {edge_id: 0 for edge_id in edge_lengths.keys()}
        nodes = list(graph.keys())
        
        # Sample nodes for approximation if network is large
        if len(nodes) > 100:
            nodes = random.sample(nodes, 100)
        
        # Calculate betweenness
        for start in nodes:
            for end in nodes:
                if start != end:
                    # Simple BFS to find shortest path
                    visited = {node: False for node in graph}
                    parent = {node: None for node in graph}
                    queue = [start]
                    visited[start] = True
                    
                    while queue:
                        node = queue.pop(0)
                        if node == end:
                            break
                        
                        for neighbor in graph.get(node, []):
                            if not visited[neighbor]:
                                visited[neighbor] = True
                                parent[neighbor] = node
                                queue.append(neighbor)
                    
                    # Backtrack to find path
                    if visited[end]:
                        current = end
                        while parent[current] is not None:
                            prev = parent[current]
                            # Find edge ID
                            for edge in net.getEdges():
                                if (edge.getFromNode().getID() == prev and 
                                    edge.getToNode().getID() == current):
                                    betweenness[edge.getID()] += 1
                                    break
                            current = prev
        
        # Normalize betweenness scores
        max_betweenness = max(betweenness.values()) if betweenness else 1
        betweenness = {k: v/max_betweenness for k, v in betweenness.items()}
        
        return betweenness
    
    def _select_incident_locations(self, net_file, num_locations):
        """Select edges where incidents will occur based on centrality"""
        betweenness = self._get_critical_edges(net_file)
        
        # Weight selection by betweenness centrality
        edges = list(betweenness.keys())
        weights = list(betweenness.values())
        
        # Select edges, ensuring they're not too close to each other
        selected_edges = []
        for _ in range(num_locations):
            if not edges:
                break
                
            # Select edge based on weights
            edge = random.choices(edges, weights=weights, k=1)[0]
            selected_edges.append(edge)
            
            # Remove nearby edges (simplified approach)
            edge_idx = edges.index(edge)
            weights.pop(edge_idx)
            edges.pop(edge_idx)
        
        return selected_edges
    
    def generate_incidents(self, net_file):
        """Generate a set of traffic incidents"""
        # Select incident locations
        incident_locations = self._select_incident_locations(net_file, self.num_incidents)
        
        # Generate incident times
        possible_times = list(range(self.simulation_start + 1800,  # Start after 30min
                                  self.simulation_end - 3600))     # End 1h before sim end
        
        incident_times = []
        for _ in range(min(self.num_incidents, len(incident_locations))):
            valid_time = False
            while not valid_time and possible_times:
                time = random.choice(possible_times)
                # Check if time is far enough from other incidents
                if not incident_times or min(abs(time - t) for t in incident_times) >= self.min_incident_spacing:
                    incident_times.append(time)
                    valid_time = True
                possible_times.remove(time)
        
        # Generate incidents
        self.incidents = []
        for location, time in zip(incident_locations, incident_times):
            incident_type = random.choice(list(self.INCIDENT_TYPES.keys()))
            config = self.INCIDENT_TYPES[incident_type]
            
            duration = random.randint(*config['duration'])
            lanes_affected = random.randint(*config['lanes_affected'])
            
            incident = {
                'type': incident_type,
                'edge': location,
                'start_time': time,
                'duration': duration,
                'lanes_affected': lanes_affected,
                'speed_reduction': config['speed_reduction']
            }
            
            self.incidents.append(incident)
        
        return self.incidents
    
    def generate_additional_file(self, output_file):
        """Generate SUMO additional file with incident definitions"""
        root = ET.Element('additional')
        
        for idx, incident in enumerate(self.incidents):
            # Create closing object
            closing = ET.SubElement(root, 'closing')
            closing.set('id', f'incident_{idx}')
            closing.set('edge', incident['edge'])
            closing.set('lanes', f"0:{incident['lanes_affected']}")  # Affect lanes from 0 to n
            closing.set('startTime', str(incident['start_time']))
            closing.set('endTime', str(incident['start_time'] + incident['duration']))
            
            if incident['speed_reduction'] > 0:
                # If not complete closure, add speed reduction
                speed = ET.SubElement(root, 'variableSpeedSign')
                speed.set('id', f'speed_{idx}')
                speed.set('lanes', incident['edge'] + '_0')  # Affect first lane
                
                step = ET.SubElement(speed, 'step')
                step.set('time', str(incident['start_time']))
                step.set('speed', str(incident['speed_reduction'] * 13.89))  # Convert to m/s
                
                # Reset speed after incident
                step = ET.SubElement(speed, 'step')
                step.set('time', str(incident['start_time'] + incident['duration']))
                step.set('speed', str(13.89))  # Reset to 50 km/h
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    def generate_rerouter_file(self, net_file, output_file):
        """Generate rerouters for affected edges"""
        net = sumolib.net.readNet(net_file)
        root = ET.Element('additional')
        
        for idx, incident in enumerate(self.incidents):
            # Get affected edge
            edge = net.getEdge(incident['edge'])
            
            # Create rerouter
            rerouter = ET.SubElement(root, 'rerouter')
            rerouter.set('id', f'rerouter_{idx}')
            rerouter.set('edges', incident['edge'])
            
            # Add rerouting interval
            interval = ET.SubElement(rerouter, 'interval')
            interval.set('begin', str(incident['start_time']))
            interval.set('end', str(incident['start_time'] + incident['duration']))
            
            # Add closing reroute
            closing = ET.SubElement(interval, 'closingReroute')
            closing.set('id', incident['edge'])
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    def print_incident_summary(self):
        """Print summary of generated incidents"""
        print("\nGenerated Traffic Incidents:")
        print("-" * 50)
        
        for idx, incident in enumerate(self.incidents, 1):
            start_time = str(timedelta(seconds=incident['start_time']))
            end_time = str(timedelta(seconds=incident['start_time'] + incident['duration']))
            
            print(f"\nIncident {idx}:")
            print(f"Type: {incident['type']}")
            print(f"Location: Edge {incident['edge']}")
            print(f"Start Time: {start_time}")
            print(f"End Time: {end_time}")
            print(f"Duration: {incident['duration']} seconds")
            print(f"Lanes Affected: {incident['lanes_affected']}")
            if incident['speed_reduction'] > 0:
                print(f"Speed Reduction: {(1-incident['speed_reduction'])*100}%")
            else:
                print("Complete lane closure")

def main():
    """Main function to generate incidents"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate traffic incidents')
    parser.add_argument('--net-file', required=True, help='Input network file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--num-incidents', type=int, default=5,
                      help='Number of incidents to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = IncidentGenerator(
        num_incidents=args.num_incidents,
        random_seed=args.seed
    )
    
    print(f"Generating {args.num_incidents} traffic incidents...")
    
    # Generate incidents
    incidents = generator.generate_incidents(args.net_file)
    
    # Generate additional files
    generator.generate_additional_file(output_dir / "incidents.add.xml")
    generator.generate_rerouter_file(args.net_file, output_dir / "rerouters.add.xml")
    
    # Print summary
    generator.print_incident_summary()
    
    print(f"\nOutput files written to: {output_dir}")

if __name__ == "__main__":
    main()