# src/environment/boston_network_setup.py

import os
import subprocess
from typing import Dict

class BostonNetworkSetup:
    """Sets up the Downtown Boston network in SUMO"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.network_dir = os.path.join('data', 'sumo_nets', 'boston')
        self.osm_file = os.path.join(self.network_dir, 'downtown_boston.osm')
        self.net_file = os.path.join(self.network_dir, 'downtown_boston.net.xml')
        self.poly_file = os.path.join(self.network_dir, 'downtown_boston.poly.xml')
        
        # Create directory if it doesn't exist
        os.makedirs(self.network_dir, exist_ok=True)

    def download_osm_data(self):
        """Download OpenStreetMap data for Downtown Boston"""
        bbox = "42.348,-71.071,42.366,-71.051"  # Downtown Boston bounding box
        osm_api_url = f"https://api.openstreetmap.org/api/0.6/map?bbox={bbox}"
        
        # Use wget to download the data
        subprocess.run(['wget', '-O', self.osm_file, osm_api_url])

    def generate_sumo_network(self):
        """Generate SUMO network from OSM data"""
        # Convert OSM to SUMO network
        netconvert_cmd = [
            'netconvert',
            '--osm-files', self.osm_file,
            '--output-file', self.net_file,
            '--geometry.remove', 'true',
            '--roundabouts.guess', 'true',
            '--ramps.guess', 'true',
            '--junctions.join', 'true',
            '--tls.guess-signals', 'true',
            '--tls.discard-simple', 'true',
            '--tls.join', 'true',
            '--crossings.guess', 'true',
            '--osm.stop-output-file', os.path.join(self.network_dir, 'stops.add.xml')
        ]
        subprocess.run(netconvert_cmd)
        
        # Generate polygons for visualization
        polyconvert_cmd = [
            'polyconvert',
            '--net-file', self.net_file,
            '--osm-files', self.osm_file,
            '--output-file', self.poly_file
        ]
        subprocess.run(polyconvert_cmd)

    def generate_traffic_demand(self):
        """Generate traffic demand for different scenarios"""
        # Morning rush hour
        morning_routes = os.path.join(self.network_dir, 'morning_rush.rou.xml')
        morning_cmd = [
            'python', '$SUMO_HOME/tools/randomTrips.py',
            '-n', self.net_file,
            '-o', morning_routes,
            '-p', '2',  # Higher frequency for rush hour
            '-b', '25200',  # 7:00 AM
            '-e', '32400',  # 9:00 AM
            '--fringe-factor', '5',  # More trips from edges
            '--trip-attributes', 'departLane="best" departSpeed="max"'
        ]
        subprocess.run(morning_cmd)
        
        # Evening rush hour
        evening_routes = os.path.join(self.network_dir, 'evening_rush.rou.xml')
        evening_cmd = [
            'python', '$SUMO_HOME/tools/randomTrips.py',
            '-n', self.net_file,
            '-o', evening_routes,
            '-p', '2',
            '-b', '57600',  # 4:00 PM
            '-e', '64800',  # 6:00 PM
            '--fringe-factor', '5',
            '--trip-attributes', 'departLane="best" departSpeed="max"'
        ]
        subprocess.run(evening_cmd)
        
        # Normal daytime
        normal_routes = os.path.join(self.network_dir, 'normal_day.rou.xml')
        normal_cmd = [
            'python', '$SUMO_HOME/tools/randomTrips.py',
            '-n', self.net_file,
            '-o', normal_routes,
            '-p', '4',  # Lower frequency
            '-b', '32400',  # 9:00 AM
            '-e', '57600',  # 4:00 PM
            '--fringe-factor', '3',
            '--trip-attributes', 'departLane="best" departSpeed="max"'
        ]
        subprocess.run(normal_cmd)

    def create_sumocfg_files(self):
        """Create SUMO configuration files for different scenarios"""
        scenarios = ['morning_rush', 'evening_rush', 'normal_day']
        
        for scenario in scenarios:
            cfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="downtown_boston.net.xml"/>
        <route-files value="{scenario}.rou.xml"/>
        <additional-files value="downtown_boston.poly.xml,stops.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="86400"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
        <time-to-teleport value="300"/>
        <max-depart-delay value="300"/>
    </processing>
    <report>
        <verbose value="true"/>
        <duration-log.statistics value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>"""
            
            with open(os.path.join(self.network_dir, f'{scenario}.sumocfg'), 'w') as f:
                f.write(cfg_content)

    def setup_network(self):
        """Run complete network setup"""
        print("Downloading OpenStreetMap data...")
        self.download_osm_data()
        
        print("Generating SUMO network...")
        self.generate_sumo_network()
        
        print("Generating traffic demand...")
        self.generate_traffic_demand()
        
        print("Creating configuration files...")
        self.create_sumocfg_files()
        
        print("Network setup complete!")