# experiments/scenarios/traffic/uniform_traffic.py

import os
import sys
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
import yaml
from datetime import datetime, timedelta
import random
import traci
import sumolib

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class UniformTrafficGenerator:
    """
    Generates uniform traffic patterns for SUMO simulation
    Distributes vehicles evenly across time and space
    """
    def __init__(self, net_file, route_file, config):
        """
        Args:
            net_file: Path to SUMO network file
            route_file: Path to output route file
            config: Traffic generation configuration
        """
        self.net_file = net_file
        self.route_file = route_file
        
        # Load configuration
        self.vehicle_count = config.get('vehicle_count', 1000)
        self.start_time = config.get('start_time', 0)
        self.end_time = config.get('end_time', 3600)
        self.vehicle_types = config.get('vehicle_types', {'passenger': 0.8, 'truck': 0.2})
        self.seed = config.get('seed', 42)
        
        # Load SUMO network
        self.net = sumolib.net.readNet(net_file)
        
        # Set random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def get_edge_weights(self):
        """Calculate edge weights based on length and speed"""
        weights = {}
        for edge in self.net.getEdges():
            # Weight based on length and speed limit
            weights[edge.getID()] = edge.getLength() / edge.getSpeed()
        return weights
    
    def generate_routes(self):
        """Generate routes with uniform distribution"""
        # Get all edges
        edges = [edge for edge in self.net.getEdges() if edge.allows("passenger")]
        edge_weights = self.get_edge_weights()
        
        # Calculate time distribution
        time_between_vehicles = (self.end_time - self.start_time) / self.vehicle_count
        
        # Generate routes
        routes = []
        for i in range(self.vehicle_count):
            # Select random start and end edges
            start_edge = random.choice(edges)
            end_edge = random.choice(edges)
            
            while end_edge == start_edge:
                end_edge = random.choice(edges)
            
            # Find route between edges
            try:
                route = self.net.getShortestPath(start_edge, end_edge, weightFunction=lambda e: edge_weights[e.getID()])[0]
                route_edges = [edge.getID() for edge in route]
                
                if route_edges:  # Only add if route exists
                    # Select vehicle type based on distribution
                    vehicle_type = random.choices(
                        list(self.vehicle_types.keys()),
                        weights=list(self.vehicle_types.values())
                    )[0]
                    
                    # Calculate departure time
                    depart_time = self.start_time + (i * time_between_vehicles)
                    
                    routes.append({
                        'id': f'vehicle_{i}',
                        'type': vehicle_type,
                        'depart': depart_time,
                        'edges': route_edges
                    })
            except:
                continue
        
        return routes
    
    def create_route_file(self, additional_params=None):
        """Create SUMO route file with generated routes"""
        routes = self.generate_routes()
        
        # Create XML tree
        root = ET.Element('routes')
        
        # Add vehicle types
        vtype_passenger = ET.SubElement(root, 'vType',
            id='passenger',
            accel='2.6',
            decel='4.5',
            sigma='0.5',
            length='5',
            minGap='2.5',
            maxSpeed='70',
            guiShape='passenger'
        )
        
        vtype_truck = ET.SubElement(root, 'vType',
            id='truck',
            accel='1.3',
            decel='4.0',
            sigma='0.5',
            length='12',
            minGap='3.0',
            maxSpeed='55',
            guiShape='truck'
        )
        
        # Add routes and vehicles
        for route in routes:
            vehicle = ET.SubElement(root, 'vehicle',
                id=route['id'],
                type=route['type'],
                depart=str(route['depart'])
            )
            
            route_element = ET.SubElement(vehicle, 'route',
                edges=' '.join(route['edges'])
            )
        
        # Create XML tree and save
        tree = ET.ElementTree(root)
        tree.write(self.route_file, encoding='utf-8', xml_declaration=True)
    
    def get_scenario_info(self):
        """Return information about the generated scenario"""
        return {
            'vehicle_count': self.vehicle_count,
            'time_period': f"{self.start_time} - {self.end_time}",
            'vehicle_types': self.vehicle_types,
            'seed': self.seed
        }

def create_uniform_scenario(env_config, traffic_config):
    """
    Create uniform traffic scenario
    
    Args:
        env_config: Environment configuration
        traffic_config: Traffic generation configuration
    
    Returns:
        scenario_path: Path to generated scenario files
    """
    # Create scenario directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    scenario_dir = Path("experiments/scenarios/traffic/generated/uniform") / timestamp
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate route file
    route_file = scenario_dir / "routes.xml"
    generator = UniformTrafficGenerator(
        net_file=env_config['net_file'],
        route_file=route_file,
        config=traffic_config
    )
    
    generator.create_route_file()
    
    # Save configurations
    with open(scenario_dir / "env_config.yaml", 'w') as f:
        yaml.dump(env_config, f)
    with open(scenario_dir / "traffic_config.yaml", 'w') as f:
        yaml.dump(traffic_config, f)
    
    # Save scenario info
    scenario_info = generator.get_scenario_info()
    with open(scenario_dir / "scenario_info.yaml", 'w') as f:
        yaml.dump(scenario_info, f)
    
    return scenario_dir

def verify_scenario(scenario_dir):
    """Verify generated scenario using SUMO"""
    try:
        # Load configurations
        with open(scenario_dir / "env_config.yaml", 'r') as f:
            env_config = yaml.safe_load(f)
        
        # Setup SUMO command
        sumo_binary = 'sumo'
        sumo_cmd = [
            sumo_binary,
            '-n', env_config['net_file'],
            '-r', str(scenario_dir / "routes.xml"),
            '--no-step-log',
            '--duration-log.statistics',
            '--tripinfo-output', str(scenario_dir / "tripinfo.xml")
        ]
        
        # Run short simulation
        traci.start(sumo_cmd)
        for step in range(100):  # Run 100 steps for verification
            traci.simulationStep()
        traci.close()
        
        return True
    except Exception as e:
        print(f"Scenario verification failed: {e}")
        return False

if __name__ == "__main__":
    # Load environment configuration
    with open("config/env_config.yaml", 'r') as f:
        env_config = yaml.safe_load(f)
    
    # Define traffic configuration for uniform scenario
    traffic_config = {
        'vehicle_count': 1000,
        'start_time': 0,
        'end_time': 3600,  # 1 hour
        'vehicle_types': {
            'passenger': 0.8,  # 80% passenger cars
            'truck': 0.2      # 20% trucks
        },
        'seed': 42
    }
    
    # Generate scenario
    print("Generating uniform traffic scenario...")
    scenario_dir = create_uniform_scenario(env_config, traffic_config)
    
    # Verify scenario
    print("Verifying generated scenario...")
    if verify_scenario(scenario_dir):
        print(f"Scenario generated successfully at: {scenario_dir}")
        
        # Load and display scenario info
        with open(scenario_dir / "scenario_info.yaml", 'r') as f:
            scenario_info = yaml.safe_load(f)
        
        print("\nScenario Information:")
        print(f"Total vehicles: {scenario_info['vehicle_count']}")
        print(f"Time period: {scenario_info['time_period']}")
        print("Vehicle distribution:")
        for vtype, ratio in scenario_info['vehicle_types'].items():
            print(f"  - {vtype}: {ratio*100}%")
    else:
        print("Failed to generate scenario. Check the error messages above.")