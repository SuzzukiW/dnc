import os
import sys
import random
import numpy as np
from typing import Dict, List, Tuple
import yaml

# SUMO imports
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")
    
import sumolib

class HierarchicalScenarios:
    """
    Defines and manages different traffic scenarios for hierarchical control testing
    """
    
    def __init__(self, config_path: str):
        """
        Initialize scenario generator
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.net_file = self.config['environment']['sumo_net_file']
        self.route_file = self.config['environment']['sumo_route_file']
        self.net = sumolib.net.readNet(self.net_file)
        
        # Initialize scenario parameters
        self.num_regions = self.config['hierarchy']['num_regions']
        self.intersections_per_region = self.config['hierarchy']['intersections_per_region']
        
        # Load network topology
        self.intersections = self._load_intersections()
        self.regions = self._create_regions()
        
    def _load_intersections(self) -> List[dict]:
        """
        Load intersection information from SUMO network
        
        Returns:
            List of dictionaries containing intersection details
        """
        intersections = []
        for node in self.net.getNodes():
            if node.getType() == 'traffic_light':
                incoming_edges = node.getIncoming()
                outgoing_edges = node.getOutgoing()
                
                intersection = {
                    'id': node.getID(),
                    'position': node.getCoord(),
                    'incoming_lanes': [lane.getID() for edge in incoming_edges 
                                     for lane in edge.getLanes()],
                    'outgoing_lanes': [lane.getID() for edge in outgoing_edges 
                                     for lane in edge.getLanes()],
                    'num_lanes': len([lane for edge in incoming_edges 
                                    for lane in edge.getLanes()])
                }
                intersections.append(intersection)
        
        return intersections
    
    def _create_regions(self) -> List[dict]:
        """
        Create regional groupings of intersections
        
        Returns:
            List of dictionaries containing region information
        """
        from sklearn.cluster import KMeans
        
        # Extract coordinates for clustering
        coords = np.array([intersection['position'] for intersection in self.intersections])
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=self.num_regions, random_state=42)
        labels = kmeans.fit_predict(coords)
        
        # Create region dictionaries
        regions = []
        for i in range(self.num_regions):
            region_intersections = [
                self.intersections[j] for j in range(len(self.intersections))
                if labels[j] == i
            ]
            
            region = {
                'id': i,
                'center': kmeans.cluster_centers_[i],
                'intersections': region_intersections,
                'boundaries': self._calculate_region_boundaries(region_intersections)
            }
            regions.append(region)
        
        return regions
    
    def _calculate_region_boundaries(self, intersections: List[dict]) -> dict:
        """
        Calculate the geographical boundaries of a region
        
        Args:
            intersections: List of intersections in the region
            
        Returns:
            Dictionary containing region boundaries
        """
        if not intersections:
            return {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}
            
        coords = np.array([intersection['position'] for intersection in intersections])
        return {
            'min_x': np.min(coords[:, 0]),
            'max_x': np.max(coords[:, 0]),
            'min_y': np.min(coords[:, 1]),
            'max_y': np.max(coords[:, 1])
        }
    
    def generate_scenarios(self) -> Dict[str, dict]:
        """
        Generate different traffic scenarios for testing
        
        Returns:
            Dictionary of scenario configurations
        """
        scenarios = {
            'uniform': self._generate_uniform_scenario(),
            'rush_hour': self._generate_rush_hour_scenario(),
            'incident': self._generate_incident_scenario(),
            'regional_event': self._generate_regional_event_scenario()
        }
        return scenarios
    
    def _generate_uniform_scenario(self) -> dict:
        """
        Generate uniform traffic distribution scenario
        
        Returns:
            Scenario configuration dictionary
        """
        return {
            'name': 'uniform',
            'description': 'Uniform traffic distribution across all regions',
            'duration': 3600,  # 1 hour
            'vehicle_rate': 0.1,  # Vehicles per second per entry point
            'regional_variations': {
                region['id']: {
                    'flow_multiplier': 1.0,
                    'vehicle_composition': {
                        'passenger': 0.7,
                        'bus': 0.1,
                        'truck': 0.2
                    }
                } for region in self.regions
            },
            'traffic_patterns': {'uniform': 1.0},
            'special_events': []
        }
    
    def _generate_rush_hour_scenario(self) -> dict:
        """
        Generate rush hour scenario with varying traffic intensities
        
        Returns:
            Scenario configuration dictionary
        """
        # Define time periods in seconds
        morning_rush = (7 * 3600, 9 * 3600)
        evening_rush = (16 * 3600, 18 * 3600)
        
        # Create flow variations for each region
        regional_variations = {}
        for region in self.regions:
            # Calculate flow multiplier based on region characteristics
            num_lanes = sum(len(intersection['incoming_lanes']) 
                          for intersection in region['intersections'])
            capacity_factor = min(1.5, num_lanes / 20)  # Normalize by typical size
            
            regional_variations[region['id']] = {
                'flow_multiplier': capacity_factor,
                'vehicle_composition': {
                    'passenger': 0.8,
                    'bus': 0.15,
                    'truck': 0.05
                },
                'peak_directions': self._calculate_peak_directions(region)
            }
        
        return {
            'name': 'rush_hour',
            'description': 'Rush hour traffic patterns with directional flow',
            'duration': 24 * 3600,  # 24 hours
            'vehicle_rate': {
                'base': 0.05,  # Base rate during off-peak
                'peak': 0.3    # Peak rate during rush hours
            },
            'regional_variations': regional_variations,
            'traffic_patterns': {
                'morning_rush': {
                    'time_window': morning_rush,
                    'flow_multiplier': 2.5
                },
                'evening_rush': {
                    'time_window': evening_rush,
                    'flow_multiplier': 2.0
                }
            },
            'special_events': []
        }
    
    def _generate_incident_scenario(self) -> dict:
        """
        Generate scenario with traffic incidents
        
        Returns:
            Scenario configuration dictionary
        """
        # Randomly select regions and intersections for incidents
        incidents = []
        for _ in range(3):  # Generate 3 random incidents
            region = random.choice(self.regions)
            intersection = random.choice(region['intersections'])
            
            incident = {
                'location': intersection['id'],
                'start_time': random.randint(0, 3000),
                'duration': random.randint(300, 1200),  # 5-20 minutes
                'severity': random.uniform(0.3, 0.8),  # Capacity reduction
                'affected_lanes': random.sample(
                    intersection['incoming_lanes'], 
                    k=random.randint(1, len(intersection['incoming_lanes']))
                )
            }
            incidents.append(incident)
        
        return {
            'name': 'incident',
            'description': 'Scenarios with random traffic incidents',
            'duration': 3600,
            'vehicle_rate': 0.15,
            'regional_variations': {
                region['id']: {
                    'flow_multiplier': 1.0,
                    'vehicle_composition': {
                        'passenger': 0.75,
                        'bus': 0.15,
                        'truck': 0.1
                    }
                } for region in self.regions
            },
            'traffic_patterns': {'uniform': 1.0},
            'special_events': incidents
        }
    
    def _generate_regional_event_scenario(self) -> dict:
        """
        Generate scenario with a major event in one region
        
        Returns:
            Scenario configuration dictionary
        """
        # Select a random region for the event
        event_region = random.choice(self.regions)
        event_center = event_region['center']
        
        # Calculate traffic flow changes based on distance from event
        regional_variations = {}
        for region in self.regions:
            distance = np.linalg.norm(
                np.array(region['center']) - np.array(event_center)
            )
            # Flow multiplier decreases with distance from event
            flow_multiplier = 2.0 * np.exp(-distance / 1000) + 1.0
            
            regional_variations[region['id']] = {
                'flow_multiplier': flow_multiplier,
                'vehicle_composition': {
                    'passenger': 0.9,
                    'bus': 0.08,
                    'truck': 0.02
                }
            }
        
        return {
            'name': 'regional_event',
            'description': 'Major event causing increased traffic in one region',
            'duration': 4 * 3600,  # 4 hours
            'vehicle_rate': {
                'base': 0.1,
                'event_area': 0.3
            },
            'regional_variations': regional_variations,
            'traffic_patterns': {
                'event_buildup': {
                    'time_window': (0, 3600),
                    'flow_multiplier': 1.5
                },
                'event_peak': {
                    'time_window': (3600, 2 * 3600),
                    'flow_multiplier': 2.0
                },
                'event_dispersal': {
                    'time_window': (2 * 3600, 4 * 3600),
                    'flow_multiplier': 1.2
                }
            },
            'special_events': [{
                'type': 'major_event',
                'location': event_region['id'],
                'start_time': 0,
                'duration': 4 * 3600,
                'affected_area_radius': 1000  # meters
            }]
        }
    
    def _calculate_peak_directions(self, region: dict) -> List[Tuple[float, float]]:
        """
        Calculate main traffic flow directions for a region
        
        Args:
            region: Region dictionary containing intersection information
            
        Returns:
            List of (angle, weight) tuples representing main flow directions
        """
        # Calculate traffic flow directions based on road network
        directions = []
        for intersection in region['intersections']:
            # Get incoming and outgoing lanes
            incoming = np.array([self.net.getLane(lane).getShape()[0] 
                               for lane in intersection['incoming_lanes']])
            outgoing = np.array([self.net.getLane(lane).getShape()[-1] 
                               for lane in intersection['outgoing_lanes']])
            
            # Calculate angles between lanes
            for in_pos in incoming:
                for out_pos in outgoing:
                    angle = np.arctan2(out_pos[1] - in_pos[1], 
                                     out_pos[0] - in_pos[0])
                    directions.append(angle)
        
        # Find major flow directions using histogram
        hist, bins = np.histogram(directions, bins=8)
        peak_indices = np.argsort(hist)[-2:]  # Get top 2 directions
        
        return [(bins[i], hist[i] / np.sum(hist)) for i in peak_indices]
    
    def save_scenario(self, scenario: dict, output_path: str):
        """
        Save scenario configuration to file
        
        Args:
            scenario: Scenario configuration dictionary
            output_path: Path to save the scenario
        """
        with open(output_path, 'w') as f:
            yaml.dump(scenario, f, default_flow_style=False)
    
    def load_scenario(self, scenario_path: str) -> dict:
        """
        Load scenario configuration from file
        
        Args:
            scenario_path: Path to scenario file
            
        Returns:
            Scenario configuration dictionary
        """
        with open(scenario_path, 'r') as f:
            return yaml.safe_load(f)
    
if __name__ == '__main__':
    # Example usage
    config_path = 'config/hierarchical_config.yaml'
    scenario_generator = HierarchicalScenarios(config_path)
    
    # Generate all scenarios
    scenarios = scenario_generator.generate_scenarios()
    
    # Save scenarios to files
    os.makedirs('scenarios/hierarchical', exist_ok=True)
    for name, scenario in scenarios.items():
        output_path = f'scenarios/hierarchical/{name}.yaml'
        scenario_generator.save_scenario(scenario, output_path)