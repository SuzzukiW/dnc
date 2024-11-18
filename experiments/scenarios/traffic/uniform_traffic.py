# experiments/scenarios/traffic/uniform_traffic.py
#!/usr/bin/env python3

import os
import sys
import random
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import sumolib
from typing import Dict, List, Tuple, Optional
import logging

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class NetworkAnalyzer:
    """Analyze network structure for uniform traffic distribution"""
    
    def __init__(self, net_file: str):
        """
        Initialize network analyzer
        
        Args:
            net_file: Path to SUMO network file
        """
        self.net = sumolib.net.readNet(net_file)
        
        # Network properties
        self.edges = self._get_valid_edges()
        self.edge_lengths = self._calculate_edge_lengths()
        self.connectivity = self._analyze_connectivity()
        self.districts = self._identify_districts()
        
        # Network statistics
        self.stats = self._calculate_network_stats()
        
        # Set up logging
        self.logger = logging.getLogger('NetworkAnalyzer')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _get_valid_edges(self) -> List[str]:
        """Get list of valid edges for traffic"""
        valid_edges = []
        
        for edge in self.net.getEdges():
            edge_id = edge.getID()
            
            # Skip internal edges and pedestrian paths
            if (not edge_id.startswith(':') and  # Not internal
                edge.allows('passenger') and      # Allows cars
                edge.getLength() > 10):           # Reasonable length
                valid_edges.append(edge_id)
        
        return valid_edges
    
    def _calculate_edge_lengths(self) -> Dict[str, float]:
        """Calculate lengths of all edges"""
        return {edge.getID(): edge.getLength() 
                for edge in self.net.getEdges()}
    
    def _analyze_connectivity(self) -> Dict[str, List[str]]:
        """Analyze network connectivity"""
        connectivity = {}
        
        for edge in self.net.getEdges():
            edge_id = edge.getID()
            if edge_id in self.edges:
                # Get outgoing connections
                outgoing = []
                for conn in edge.getOutgoing():
                    to_edge = conn.getTo()
                    if to_edge.getID() in self.edges:
                        outgoing.append(to_edge.getID())
                
                connectivity[edge_id] = outgoing
        
        return connectivity
    
    def _identify_districts(self, grid_size: float = 500.0) -> Dict[str, List[str]]:
        """
        Identify network districts using grid-based approach
        
        Args:
            grid_size: Size of grid cells in meters
            
        Returns:
            Dictionary mapping district IDs to lists of edge IDs
        """
        districts = {}
        
        # Get network boundaries
        bounds = self.net.getBoundary()
        min_x, min_y = bounds[0], bounds[1]
        max_x, max_y = bounds[2], bounds[3]
        
        # Create grid
        num_cols = int((max_x - min_x) / grid_size) + 1
        num_rows = int((max_y - min_y) / grid_size) + 1
        
        # Assign edges to districts
        for edge in self.net.getEdges():
            if edge.getID() in self.edges:
                # Get edge center coordinates
                x, y = edge.getFromNode().getCoord()
                
                # Calculate grid cell
                col = int((x - min_x) / grid_size)
                row = int((y - min_y) / grid_size)
                district_id = f"d_{row}_{col}"
                
                if district_id not in districts:
                    districts[district_id] = []
                districts[district_id].append(edge.getID())
        
        return districts
    
    def _calculate_network_stats(self) -> dict:
        """Calculate network statistics"""
        stats = {
            'num_edges': len(self.edges),
            'total_length': sum(self.edge_lengths.values()),
            'avg_edge_length': np.mean(list(self.edge_lengths.values())),
            'num_districts': len(self.districts),
            'connectivity': {}
        }
        
        # Calculate connectivity statistics
        out_degrees = []
        for edge_id in self.edges:
            out_degree = len(self.connectivity.get(edge_id, []))
            out_degrees.append(out_degree)
        
        stats['connectivity'].update({
            'avg_out_degree': np.mean(out_degrees),
            'max_out_degree': max(out_degrees),
            'min_out_degree': min(out_degrees)
        })
        
        return stats
    
    def get_edge_distribution(self) -> Dict[str, float]:
        """
        Calculate edge weights for uniform distribution
        
        Returns:
            Dictionary mapping edge IDs to their weights
        """
        weights = {}
        
        # Calculate base weights using edge lengths and connectivity
        for edge_id in self.edges:
            length = self.edge_lengths[edge_id]
            out_degree = len(self.connectivity.get(edge_id, []))
            
            # Weight based on length and connectivity
            weight = length * (1 + 0.1 * out_degree)  # Small bonus for connectivity
            weights[edge_id] = weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {edge_id: w/total_weight 
                      for edge_id, w in weights.items()}
        
        return weights
    
    def get_district_pairs(self, num_pairs: int) -> List[Tuple[str, str]]:
        """
        Generate pairs of districts for traffic flow
        
        Args:
            num_pairs: Number of district pairs to generate
            
        Returns:
            List of (source_district, target_district) pairs
        """
        district_ids = list(self.districts.keys())
        pairs = []
        
        for _ in range(num_pairs):
            source = random.choice(district_ids)
            target = random.choice([d for d in district_ids if d != source])
            pairs.append((source, target))
        
        return pairs
    
    def find_route(self, 
                  from_edge: str, 
                  to_edge: str, 
                  max_alternatives: int = 3) -> List[List[str]]:
        """
        Find possible routes between edges
        
        Args:
            from_edge: Starting edge ID
            to_edge: Target edge ID
            max_alternatives: Maximum number of alternative routes
            
        Returns:
            List of routes (each route is a list of edge IDs)
        """
        try:
            # Get edge objects
            from_edge_obj = self.net.getEdge(from_edge)
            to_edge_obj = self.net.getEdge(to_edge)
            
            # Find routes
            routes = []
            for _ in range(max_alternatives):
                try:
                    # Use SUMO's routing algorithm
                    route = self.net.getOptimalPath(
                        from_edge_obj,
                        to_edge_obj,
                        randomize=True
                    )
                    
                    if route:
                        # Extract edge IDs
                        edge_ids = [edge.getID() for edge in route]
                        if edge_ids not in routes:
                            routes.append(edge_ids)
                except:
                    continue
            
            return routes if routes else [[from_edge, to_edge]]
            
        except Exception as e:
            self.logger.error(f"Error finding route: {e}")
            return [[from_edge, to_edge]]
    
    def get_network_summary(self) -> str:
        """Get human-readable network summary"""
        summary = [
            "Network Summary:",
            f"Total edges: {self.stats['num_edges']}",
            f"Total length: {self.stats['total_length']:.2f} meters",
            f"Average edge length: {self.stats['avg_edge_length']:.2f} meters",
            f"Number of districts: {self.stats['num_districts']}",
            "\nConnectivity Statistics:",
            f"Average out-degree: {self.stats['connectivity']['avg_out_degree']:.2f}",
            f"Maximum out-degree: {self.stats['connectivity']['max_out_degree']}",
            f"Minimum out-degree: {self.stats['connectivity']['min_out_degree']}"
        ]
        
        return "\n".join(summary)

class TrafficDemandCalculator:
    """Calculate uniform traffic demand across the network"""
    
    def __init__(self,
                network: NetworkAnalyzer,
                base_flow: float = 300.0,      # vehicles per hour
                time_interval: int = 3600,      # 1 hour
                flow_variation: float = 0.1):   # 10% variation
        """
        Initialize traffic demand calculator
        
        Args:
            network: NetworkAnalyzer instance
            base_flow: Base flow rate (vehicles/hour)
            time_interval: Time interval for flow calculation
            flow_variation: Allowed variation in flow rate
        """
        self.network = network
        self.base_flow = base_flow
        self.time_interval = time_interval
        self.flow_variation = flow_variation
        
        # Calculate edge weights
        self.edge_weights = network.get_edge_distribution()
        
        # Initialize demand matrices
        self.od_matrix = self._initialize_od_matrix()
        self.flow_matrix = self._calculate_flow_matrix()
        
        # Set up logging
        self.logger = logging.getLogger('TrafficDemandCalculator')
        self.logger.setLevel(logging.INFO)
    
    def _initialize_od_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize origin-destination matrix"""
        od_matrix = {}
        
        for from_edge in self.network.edges:
            od_matrix[from_edge] = {}
            for to_edge in self.network.edges:
                if from_edge != to_edge:
                    # Calculate base probability
                    prob = (self.edge_weights[from_edge] * 
                           self.edge_weights[to_edge])
                    od_matrix[from_edge][to_edge] = prob
        
        return od_matrix
    
    def _calculate_flow_matrix(self) -> Dict[str, Dict[str, float]]:
        """Calculate flow rates between edges"""
        flow_matrix = {}
        total_prob = sum(sum(probs.values()) 
                        for probs in self.od_matrix.values())
        
        if total_prob > 0:
            # Scale probabilities to flow rates
            total_flow = self.base_flow * self.time_interval / 3600
            
            for from_edge in self.od_matrix:
                flow_matrix[from_edge] = {}
                for to_edge, prob in self.od_matrix[from_edge].items():
                    flow = (prob / total_prob) * total_flow
                    flow_matrix[from_edge][to_edge] = flow
        
        return flow_matrix
    
    def get_flow_rate(self,
                     from_edge: str,
                     to_edge: str,
                     time: float) -> float:
        """
        Get flow rate between edges at given time
        
        Args:
            from_edge: Origin edge ID
            to_edge: Destination edge ID
            time: Current simulation time
            
        Returns:
            Flow rate (vehicles/hour)
        """
        if from_edge in self.flow_matrix and to_edge in self.flow_matrix[from_edge]:
            base_flow = self.flow_matrix[from_edge][to_edge]
            
            # Add small random variation
            variation = random.uniform(
                -self.flow_variation,
                self.flow_variation
            )
            
            return base_flow * (1 + variation)
        
        return 0.0

# Part B

class UniformTrafficGenerator:
    """Generate uniform traffic patterns"""
    
    # Vehicle type distributions
    VEHICLE_TYPES = {
        'passenger': {
            'proportion': 0.75,
            'length': '5.0',
            'maxSpeed': '50.0',
            'accel': '2.6',
            'decel': '4.5',
            'sigma': '0.5',
            'variants': [
                ('standard', 0.6),
                ('fast', 0.2),
                ('eco', 0.2)
            ]
        },
        'truck': {
            'proportion': 0.15,
            'length': '7.5',
            'maxSpeed': '35.0',
            'accel': '1.0',
            'decel': '4.0',
            'sigma': '0.5',
            'variants': [
                ('light', 0.4),
                ('heavy', 0.6)
            ]
        },
        'bus': {
            'proportion': 0.10,
            'length': '12.0',
            'maxSpeed': '30.0',
            'accel': '1.2',
            'decel': '4.0',
            'sigma': '0.3',
            'variants': [
                ('city', 0.7),
                ('coach', 0.3)
            ]
        }
    }
    
    def __init__(self,
                net_file: str,
                output_dir: str,
                demand_calculator: TrafficDemandCalculator,
                random_seed: Optional[int] = None):
        """
        Initialize uniform traffic generator
        
        Args:
            net_file: Path to SUMO network file
            output_dir: Directory for output files
            demand_calculator: TrafficDemandCalculator instance
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.demand_calc = demand_calculator
        
        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Initialize counters
        self.vehicle_counts = defaultdict(int)
        self.type_counts = defaultdict(int)
        
        # Track generated data
        self.trips = []
        self.routes = defaultdict(list)
        self.statistics = defaultdict(list)
        
        # Set up logging
        self.logger = logging.getLogger('UniformTrafficGenerator')
        self.logger.setLevel(logging.INFO)
    
    def _create_vehicle_types(self) -> ET.Element:
        """Create vehicle type definitions with variants"""
        vtypes = ET.Element('vtypes')
        
        for vtype, config in self.VEHICLE_TYPES.items():
            for variant, prob in config['variants']:
                type_id = f"{vtype}_{variant}"
                vtype_el = ET.SubElement(vtypes, 'vType')
                vtype_el.set('id', type_id)
                
                # Set base attributes
                for attr, value in config.items():
                    if attr not in ['proportion', 'variants']:
                        vtype_el.set(attr, value)
                
                # Modify attributes based on variant
                if variant == 'fast':
                    vtype_el.set('maxSpeed', str(float(config['maxSpeed']) * 1.2))
                    vtype_el.set('accel', str(float(config['accel']) * 1.2))
                elif variant == 'eco':
                    vtype_el.set('maxSpeed', str(float(config['maxSpeed']) * 0.9))
                    vtype_el.set('accel', str(float(config['accel']) * 0.8))
                elif variant == 'heavy':
                    vtype_el.set('length', str(float(config['length']) * 1.2))
                    vtype_el.set('maxSpeed', str(float(config['maxSpeed']) * 0.9))
                elif variant == 'coach':
                    vtype_el.set('maxSpeed', str(float(config['maxSpeed']) * 1.1))
        
        return vtypes
    
    def _select_vehicle_type(self) -> str:
        """Select random vehicle type based on proportions"""
        rand = random.random()
        cumsum = 0
        
        for vtype, config in self.VEHICLE_TYPES.items():
            cumsum += config['proportion']
            if rand <= cumsum:
                # Select variant
                variant = random.choices(
                    [v[0] for v in config['variants']],
                    weights=[v[1] for v in config['variants']]
                )[0]
                return f"{vtype}_{variant}"
        
        return "passenger_standard"  # Default
    
    def _generate_trips(self,
                      start_time: int,
                      end_time: int,
                      step_length: int = 1) -> List[dict]:
        """Generate trips for time period"""
        trips = []
        current_time = start_time
        
        while current_time < end_time:
            # Get active district pairs
            district_pairs = self.demand_calc.network.get_district_pairs(5)
            
            for source_district, target_district in district_pairs:
                # Get edges from districts
                source_edges = self.demand_calc.network.districts[source_district]
                target_edges = self.demand_calc.network.districts[target_district]
                
                if source_edges and target_edges:
                    # Select random edges
                    from_edge = random.choice(source_edges)
                    to_edge = random.choice(target_edges)
                    
                    # Calculate flow rate
                    flow = self.demand_calc.get_flow_rate(
                        from_edge, to_edge, current_time
                    )
                    
                    # Generate vehicles based on flow rate
                    num_vehicles = np.random.poisson(flow * step_length / 3600)
                    
                    for _ in range(num_vehicles):
                        # Select vehicle type
                        vtype = self._select_vehicle_type()
                        vid = f"{vtype}_{self.vehicle_counts[vtype]}"
                        self.vehicle_counts[vtype] += 1
                        
                        # Create trip
                        trip = {
                            'id': vid,
                            'type': vtype,
                            'from': from_edge,
                            'to': to_edge,
                            'depart': current_time,
                            'departSpeed': 'max',  # Start at max speed
                            'departLane': 'best'   # Best lane for route
                        }
                        
                        trips.append(trip)
                        
                        # Update statistics
                        base_type = vtype.split('_')[0]
                        self.type_counts[base_type] += 1
            
            current_time += step_length
        
        return trips
    
    def generate(self,
                simulation_duration: int,
                step_length: int = 1) -> Tuple[str, dict]:
        """
        Generate uniform traffic pattern
        
        Args:
            simulation_duration: Total simulation time (seconds)
            step_length: Simulation step length (seconds)
            
        Returns:
            Tuple of (output file path, statistics)
        """
        # Create output file
        routes_file = self.output_dir / 'uniform_routes.rou.xml'
        root = ET.Element('routes')
        
        # Add vehicle types
        root.append(self._create_vehicle_types())
        
        # Generate trips
        self.trips = self._generate_trips(
            0, simulation_duration, step_length
        )
        
        # Add trips to XML
        for trip in self.trips:
            trip_element = ET.SubElement(root, 'trip')
            for key, value in trip.items():
                trip_element.set(key, str(value))
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(routes_file, encoding='utf-8', xml_declaration=True)
        
        # Generate statistics
        stats = {
            'total_vehicles': sum(self.type_counts.values()),
            'vehicle_distribution': {
                vtype: count / max(sum(self.type_counts.values()), 1)
                for vtype, count in self.type_counts.items()
            },
            'time_distribution': self._calculate_time_distribution(step_length),
            'network_statistics': self.demand_calc.network.stats
        }
        
        # Save statistics
        stats_file = self.output_dir / 'uniform_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)
        
        return str(routes_file), stats
    
    def _calculate_time_distribution(self, step_length: int) -> Dict[str, List[int]]:
        """Calculate time-based trip distribution"""
        time_slots = defaultdict(int)
        slot_size = 300  # 5-minute slots
        
        for trip in self.trips:
            slot = (trip['depart'] // slot_size) * slot_size
            time_slots[str(slot)] += 1
        
        return dict(time_slots)
    
    def get_generation_summary(self) -> str:
        """Get human-readable generation summary"""
        summary = [
            "Traffic Generation Summary:",
            f"Total vehicles: {sum(self.type_counts.values())}",
            "\nVehicle Type Distribution:"
        ]
        
        total = sum(self.type_counts.values()) or 1
        for vtype, count in self.type_counts.items():
            percentage = (count / total) * 100
            summary.append(f"- {vtype}: {count} ({percentage:.1f}%)")
        
        time_dist = self._calculate_time_distribution(300)
        if time_dist:
            summary.extend([
                "\nTime Distribution (5-minute intervals):",
                f"Peak period: {max(time_dist.items(), key=lambda x: x[1])[0]} seconds",
                f"Average vehicles per interval: {np.mean(list(time_dist.values())):.1f}"
            ])
        
        return "\n".join(summary)

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate uniform traffic')
    parser.add_argument('--net-file', required=True, help='Input network file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--duration', type=int, default=3600,
                      help='Simulation duration (seconds)')
    parser.add_argument('--flow-rate', type=float, default=300,
                      help='Base flow rate (vehicles/hour)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create network analyzer
    network = NetworkAnalyzer(args.net_file)
    print("\nNetwork Analysis:")
    print(network.get_network_summary())
    
    # Create demand calculator
    demand_calc = TrafficDemandCalculator(
        network,
        base_flow=args.flow_rate
    )
    
    # Create traffic generator
    generator = UniformTrafficGenerator(
        args.net_file,
        args.output_dir,
        demand_calc,
        args.seed
    )
    
    # Generate traffic
    routes_file, stats = generator.generate(
        simulation_duration=args.duration
    )
    
    print("\nTraffic Generation Results:")
    print(generator.get_generation_summary())
    
    print(f"\nOutput files:")
    print(f"- Routes: {routes_file}")
    print(f"- Statistics: {args.output_dir}/uniform_statistics.json")

if __name__ == "__main__":
    main()

