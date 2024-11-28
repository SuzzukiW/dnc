# experiments/scenarios/communication/shared_experience.py

import os
import sys
import yaml
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

class SharedExperienceScenario:
    def __init__(self, config_path: str):
        """Initialize the shared experience scenario."""
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.net_file = self.config['simulation']['net_file']
        self.route_file = self.config['simulation']['route_file']
        self.observation_radius = self.config['environment']['observation_radius']
        
        # Initialize intersection topology
        self.intersection_distances = defaultdict(dict)
        self._build_intersection_topology()

    def _build_intersection_topology(self):
        """Build topology of intersections including neighbors and distances."""
        try:
            import sumolib
            net = sumolib.net.readNet(self.net_file)
            
            # Get all traffic light nodes
            all_nodes = net.getNodes()
            tl_nodes = []
            
            # Collect all traffic light nodes
            for node in all_nodes:
                node_id = node.getID()
                # Include all traffic lights and special nodes
                if (node.getType() in ['traffic_light', 'traffic_light_right_on_red', 'traffic_light_unregulated'] or
                    'cluster' in node_id.lower() or 
                    'joinedS' in node_id or 
                    'GS_' in node_id):
                    if node.getIncoming() or node.getOutgoing():
                        tl_nodes.append(node)
                        # Initialize empty lists for each node (for compatibility)
                        self.intersection_distances[node_id] = {}
            
            print(f"\nFound {len(tl_nodes)} traffic light nodes")
            # No need to calculate neighbors since all agents share experiences
        
        except Exception as e:
            print(f"Error building intersection topology: {e}")
            import traceback
            print(traceback.format_exc())

    def initialize_intersection(self, intersection_id):
        """Initialize an intersection with the given ID."""
        print(f"\nSetting up traffic light: {intersection_id}")
        
        # Get program logic
        print(f"Getting program logic for {intersection_id}")
        program_logic = self.net.getTLS(intersection_id).getPrograms()['0']
        phases = program_logic.getPhases()
        print(f"Found {len(phases)} phases")
        
        # Get controlled links
        print(f"Getting controlled links for {intersection_id}")
        controlled_links = self.net.getTLS(intersection_id).getLinks()
        print(f"Found {len(controlled_links)} links")
        
        print(f"Successfully initialized {intersection_id}")

    def get_neighbor_weights(self, intersection_id: str) -> Dict[str, float]:
        """Return empty dict since we're sharing experiences globally."""
        return {}

    def get_sharing_config(self) -> dict:
        """Get configuration parameters for experience sharing."""
        return self.config['shared_experience']

    def get_intersection_info(self) -> Dict[str, Dict[str, float]]:
        """Get intersection topology information."""
        return self.intersection_distances

    def should_share_experience(self, step: int) -> bool:
        """Determine if experience should be shared at current step."""
        return step % self.config['shared_experience']['sharing_interval'] == 0

    def get_reward_weights(self) -> dict:
        """Get weights for different components of the reward function."""
        return self.config['reward']

    def get_training_params(self) -> dict:
        """Get training-related parameters."""
        return self.config['training']