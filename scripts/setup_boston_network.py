# scripts/setup_boston_network.py

import os
import sys
import subprocess
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from datetime import datetime, time

class BostonNetworkGenerator:
    def __init__(self, output_dir="data/sumo_nets/boston"):
        # Check and set SUMO_HOME
        if 'SUMO_HOME' not in os.environ:
            sumo_home = "/opt/homebrew/share/sumo"  # Default MacOS Homebrew location
            if os.path.exists(sumo_home):
                os.environ['SUMO_HOME'] = sumo_home
            else:
                raise EnvironmentError("SUMO_HOME not found. Please install SUMO using Homebrew.")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # City of Boston boundaries
        self.bounds = {
            'north': 42.395900,  # Including Charlestown and parts of Somerville
            'south': 42.227900,  # Including Hyde Park and Mattapan
            'east': -70.968500,  # Including East Boston and Logan Airport
            'west': -71.191100   # Including West Roxbury and parts of Brookline
        }
        
        # SUMO file paths
        self.osm_file = self.output_dir / "boston.osm"
        self.net_file = self.output_dir / "boston.net.xml"
        self.poly_file = self.output_dir / "boston.poly.xml"
        self.tll_file = self.output_dir / "boston.tll.xml"
        self.typ_file = self.output_dir / "boston.typ.xml"

    def generate_network(self):
        """Generate SUMO network from OSM data"""
        print("Generating SUMO network...")
        
        netconvert_cmd = [
            "netconvert",
            "--osm-files", str(self.osm_file),
            "--output-file", str(self.net_file),
            "--type-files", str(self.typ_file),
            "--geometry.remove", "true",
            "--roundabouts.guess", "true",
            "--ramps.guess", "true",
            "--junctions.join", "true",
            "--tls.guess-signals", "true",
            "--tls.discard-simple", "true",
            "--tls.join", "true",
            "--crossings.guess", "true",
            "--edges.join", "true",
            "--keep-edges.by-vclass", "passenger,bus,taxi",
            "--remove-edges.isolated", "true",
            "--no-turnarounds", "true",
            "--junctions.corner-detail", "5",
            "--junctions.limit-turn-speed", "5.5",
            "--tls.default-type", "actuated",
            "--tls.yellow.time", "3",
            "--output.street-names", "true",
            "--geometry.max-segment-length", "300",  # Handle larger network
            "--process-output-options", "verbose",
            "--ignore-errors", "true",  # Handle potential OSM data issues
            "--osm.all-attributes", "true",  # Include all OSM attributes
            "--osm.extra-attributes", "streetlights,surface,lit"  # Additional road info
        ]
        
        try:
            result = subprocess.run(netconvert_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("Error generating network:")
                print(result.stderr)
                return False
            print(f"Generated network file at {self.net_file}")
            return True
        except Exception as e:
            print(f"Error running netconvert: {e}")
            return False

    def generate_traffic_demand(self):
        """Generate traffic demand for different scenarios"""
        print("Generating traffic demand...")
        
        # Define areas of high traffic generation/attraction
        major_areas = {
            'downtown': {'weight': 2.0, 'bounds': (42.348, -71.071, 42.366, -71.051)},
            'logan_airport': {'weight': 1.8, 'bounds': (42.361, -71.020, 42.374, -71.001)},
            'harvard_uni': {'weight': 1.5, 'bounds': (42.370, -71.118, 42.377, -71.113)},
            'fenway': {'weight': 1.3, 'bounds': (42.342, -71.102, 42.348, -71.094)},
        }
        
        # Define time periods with specific characteristics
        periods = {
            'morning_rush': {
                'time': (time(6, 0), time(10, 0)),
                'factor': 2.0,
                'flows': {
                    'to_downtown': 0.6,
                    'to_airport': 0.2,
                    'to_universities': 0.2
                }
            },
            'midday': {
                'time': (time(10, 0), time(15, 0)),
                'factor': 1.0,
                'flows': {
                    'shopping_areas': 0.4,
                    'business_districts': 0.3,
                    'tourist_spots': 0.3
                }
            },
            'evening_rush': {
                'time': (time(15, 0), time(19, 0)),
                'factor': 2.0,
                'flows': {
                    'from_downtown': 0.5,
                    'from_universities': 0.2,
                    'shopping_return': 0.3
                }
            },
            'night': {
                'time': (time(19, 0), time(23, 0)),
                'factor': 0.7,
                'flows': {
                    'entertainment_areas': 0.4,
                    'residential_return': 0.4,
                    'late_work': 0.2
                }
            }
        }

        def get_edges_in_area(bounds):
            """Get all edge IDs within given bounds"""
            min_lat, min_lon, max_lat, max_lon = bounds
            edges = []
            
            # Parse the network file to find edges within bounds
            tree = ET.parse(self.net_file)
            root = tree.getroot()
            
            for edge in root.findall('edge'):
                # Skip internal edges
                if edge.get('function') == 'internal':
                    continue
                    
                lane = edge.find('lane')
                if lane is not None:
                    shape = lane.get('shape')
                    if shape:
                        # Convert shape string to coordinates
                        coords = [tuple(map(float, point.split(','))) for point in shape.split(' ')]
                        
                        # Check if any point is within bounds
                        for x, y in coords:
                            if min_lon <= x <= max_lon and min_lat <= y <= max_lat:
                                edges.append(edge.get('id'))
                                break
            
            return edges

        def create_flow_xml(period_name, period_data):
            """Create flow definition XML for a specific period"""
            period_start, period_end = period_data['time']
            start_seconds = period_start.hour * 3600 + period_start.minute * 60
            end_seconds = period_end.hour * 3600 + period_end.minute * 60
            
            flow_file = self.output_dir / f"{period_name}_flows.xml"
            
            with open(flow_file, 'w') as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
                
                # Define vehicle types
                f.write('''    <vType id="car" vClass="passenger" length="5" maxSpeed="50" speedFactor="1.0" speedDev="0.1" sigma="0.5"/>
        <vType id="bus" vClass="bus" length="12" maxSpeed="25" speedFactor="1.0" speedDev="0.1" sigma="0.3"/>
        <vType id="taxi" vClass="taxi" length="5" maxSpeed="50" speedFactor="1.1" speedDev="0.1" sigma="0.5"/>\n''')
                
                flow_id = 0
                
                # Generate flows for each major area based on the period's characteristics
                for source_area_name, source_area in major_areas.items():
                    source_edges = get_edges_in_area(source_area['bounds'])
                    
                    for target_area_name, target_area in major_areas.items():
                        if source_area_name == target_area_name:
                            continue
                            
                        target_edges = get_edges_in_area(target_area['bounds'])
                        
                        # Calculate flow probability based on area weights and period flows
                        base_prob = source_area['weight'] * target_area['weight']
                        
                        # Adjust probability based on period-specific flows
                        flow_adjustment = 1.0
                        for flow_type, flow_weight in period_data['flows'].items():
                            if flow_type in [f"to_{source_area_name}", f"from_{source_area_name}"]:
                                flow_adjustment *= flow_weight
                        
                        # Calculate vehicles per hour
                        veh_per_hour = int(300 * base_prob * flow_adjustment * period_data['factor'])
                        
                        # Create flows for different vehicle types
                        vehicle_distributions = [
                            ('car', 0.8),   # 80% cars
                            ('taxi', 0.15),  # 15% taxis
                            ('bus', 0.05)    # 5% buses
                        ]
                        
                        for veh_type, type_ratio in vehicle_distributions:
                            if source_edges and target_edges:  # Only create flow if we have valid edges
                                f.write(f'''    <flow id="flow_{flow_id}" type="{veh_type}" begin="{start_seconds}" end="{end_seconds}"
                          vehsPerHour="{int(veh_per_hour * type_ratio)}" from="{random.choice(source_edges)}" to="{random.choice(target_edges)}"/>\n''')
                                flow_id += 1
                
                f.write('</routes>\n')
            
            return flow_file

        def create_scenario_config(period_name, flow_file):
            """Create SUMO configuration file for a specific scenario"""
            config_file = self.output_dir / f"{period_name}.sumocfg"
            
            with open(config_file, 'w') as f:
                f.write('''<?xml version="1.0" encoding="UTF-8"?>
    <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <input>
            <net-file value="boston.net.xml"/>
            <route-files value="{}"/>
            <additional-files value="boston.poly.xml"/>
        </input>
        <time>
            <begin value="0"/>
            <end value="86400"/>
            <step-length value="1"/>
        </time>
        <processing>
            <ignore-route-errors value="true"/>
            <time-to-teleport value="300"/>
            <time-to-teleport.highways value="0"/>
            <max-depart-delay value="300"/>
        </processing>
        <routing>
            <device.rerouting.probability value="0.8"/>
            <device.rerouting.period value="300"/>
            <device.rerouting.pre-period value="300"/>
        </routing>
        <report>
            <verbose value="true"/>
            <duration-log.statistics value="true"/>
            <log value="simulation_{}.log"/>
        </report>
    </configuration>'''.format(flow_file.name, period_name))
            
            return config_file

        try:
            print("Generating traffic demand for different periods...")
            
            # Generate flows and configs for each period
            for period_name, period_data in periods.items():
                print(f"\nProcessing {period_name}...")
                
                # Create flow definitions
                flow_file = create_flow_xml(period_name, period_data)
                print(f"Created flow file: {flow_file}")
                
                # Create scenario configuration
                config_file = create_scenario_config(period_name, flow_file)
                print(f"Created config file: {config_file}")
                
                # Generate routes using DUAROUTER
                routes_file = self.output_dir / f"{period_name}.rou.xml"
                duarouter_cmd = [
                    "duarouter",
                    "-n", str(self.net_file),
                    "-r", str(flow_file),
                    "-o", str(routes_file),
                    "--ignore-errors",
                    "--no-warnings",
                    "--begin", str(period_data['time'][0].hour * 3600),
                    "--end", str(period_data['time'][1].hour * 3600),
                    "--max-alternatives", "3",
                    "--weights.priority-factor", "0.5"
                ]
                
                print(f"Generating routes for {period_name}...")
                result = subprocess.run(duarouter_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"Successfully generated routes: {routes_file}")
                else:
                    print(f"Warning: Error generating routes for {period_name}")
                    print(result.stderr)

        except Exception as e:
            print(f"Error generating traffic demand: {e}")
            raise

        print("\nTraffic demand generation completed!")

    def create_district_file(self):
        """Create district definitions for major areas"""
        districts = {
            'downtown': (42.348, -71.071, 42.366, -71.051),
            'back_bay': (42.345, -71.085, 42.355, -71.070),
            'south_boston': (42.330, -71.060, 42.340, -71.020),
            'dorchester': (42.280, -71.080, 42.320, -71.040),
            'roxbury': (42.315, -71.100, 42.335, -71.075),
            'jamaica_plain': (42.300, -71.120, 42.325, -71.100),
            'brighton': (42.345, -71.170, 42.365, -71.140),
            'east_boston': (42.365, -71.040, 42.385, -71.000),
            'charlestown': (42.370, -71.070, 42.385, -71.045),
            'hyde_park': (42.235, -71.140, 42.265, -71.110),
            'west_roxbury': (42.270, -71.190, 42.300, -71.150),
            'logan_airport': (42.361, -71.020, 42.374, -71.001)
        }
        
        taz_file = self.output_dir / "boston_districts.taz.xml"
        with open(taz_file, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<tazs>\n')
            for name, bounds in districts.items():
                f.write(f'    <taz id="{name}" shape="{bounds[1]},{bounds[0]} {bounds[3]},{bounds[0]} '
                       f'{bounds[3]},{bounds[2]} {bounds[1]},{bounds[2]} {bounds[1]},{bounds[0]}"/>\n')
            f.write('</tazs>\n')

    def setup_network(self):
        """Run complete network setup"""
        try:
            print("Starting Boston network setup...")
            
            # Download OSM data if not exists
            if not self.osm_file.exists():
                self.download_osm_data()
            else:
                print("Using existing OSM data file")

            # Create type file if not exists
            if not self.typ_file.exists():
                self.create_type_file()
            else:
                print("Using existing type file")

            # Generate network
            if not self.net_file.exists() or not self.generate_network():
                print("Failed to generate network")
                return

            # Generate traffic demand
            self.generate_traffic_demand()
            
            # Create config files
            self.create_config_files()
            self.create_gui_settings()
            
            print("\nBoston network setup completed!")
            print(f"\nNetwork files are located in: {self.output_dir}")
            print("\nVerifying files...")
            self.verify_files()
            
        except Exception as e:
            print(f"\nError during setup: {e}")
            raise

    def verify_files(self):
        """Verify that all necessary files were created"""
        required_files = [
            self.osm_file,
            self.net_file,
            self.typ_file,
            self.output_dir / "morning_rush.sumocfg",
            self.output_dir / "midday.sumocfg",
            self.output_dir / "evening_rush.sumocfg"
        ]
        
        all_exist = True
        for file in required_files:
            if file.exists():
                print(f"✓ Found {file.name}")
            else:
                print(f"✗ Missing {file.name}")
                all_exist = False
        
        if all_exist:
            print("\nAll required files are present. You can now run:")
            print(f"sumo-gui -c {self.output_dir}/morning_rush.sumocfg")
        else:
            print("\nSome files are missing. Please check the errors above.")

def main():
    try:
        generator = BostonNetworkGenerator()
        generator.setup_network()
    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()