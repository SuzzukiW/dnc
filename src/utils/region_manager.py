"""
Region Manager for Traffic Light Coordination
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import networkx as nx
import traci

class RegionManager:
    """Manages traffic light regions for hierarchical coordination"""
    
    def __init__(self, neighbor_map: Dict[str, List[str]], num_regions: int = 4):
        """
        Initialize region manager
        
        Args:
            neighbor_map: Dictionary mapping traffic light IDs to their neighbor IDs
            num_regions: Number of regions to create
        """
        self.neighbor_map = neighbor_map
        self.num_regions = num_regions
        
        # Create graph from neighbor map
        self.graph = nx.Graph()
        for tl_id, neighbors in neighbor_map.items():
            self.graph.add_node(tl_id)
            for neighbor in neighbors:
                self.graph.add_edge(tl_id, neighbor)
        
        # Partition graph into regions using spectral clustering
        self.regions = self._create_regions()
        
        # Create reverse mapping from traffic light to region
        self.tl_to_region = {}
        for region_id, tls in enumerate(self.regions):
            for tl in tls:
                self.tl_to_region[tl] = region_id
    
    def _create_regions(self) -> List[Set[str]]:
        """
        Create regions using spectral clustering
        
        Returns:
            List of sets containing traffic light IDs for each region
        """
        if len(self.graph) < self.num_regions:
            # If fewer nodes than regions, each node gets its own region
            return [{node} for node in self.graph.nodes()]
        
        try:
            # Try spectral clustering first
            clustering = nx.spectral_clustering(
                self.graph,
                n_clusters=self.num_regions,
                random_state=42
            )
        except:
            # Fall back to random partition if spectral clustering fails
            clustering = np.random.randint(0, self.num_regions, len(self.graph))
        
        # Convert clustering to list of sets
        regions = [set() for _ in range(self.num_regions)]
        for node_idx, region_idx in enumerate(clustering):
            node = list(self.graph.nodes())[node_idx]
            regions[region_idx].add(node)
        
        return regions
    
    def get_region(self, tl_id: str) -> int:
        """
        Get region ID for a traffic light
        
        Args:
            tl_id: Traffic light ID
            
        Returns:
            Region ID
        """
        return self.tl_to_region.get(tl_id, -1)
    
    def get_region_members(self, region_id: int) -> Set[str]:
        """
        Get all traffic lights in a region
        
        Args:
            region_id: Region ID
            
        Returns:
            Set of traffic light IDs in the region
        """
        if 0 <= region_id < len(self.regions):
            return self.regions[region_id]
        return set()
    
    def get_region_neighbors(self, region_id: int) -> Set[str]:
        """
        Get traffic lights that border a region
        
        Args:
            region_id: Region ID
            
        Returns:
            Set of traffic light IDs that border the region
        """
        if not (0 <= region_id < len(self.regions)):
            return set()
            
        border_tls = set()
        region_tls = self.regions[region_id]
        
        # Check each traffic light in the region
        for tl in region_tls:
            # Add neighbors that are not in the same region
            for neighbor in self.neighbor_map.get(tl, []):
                if neighbor not in region_tls:
                    border_tls.add(neighbor)
        
        return border_tls
    
    def calculate_regional_reward(self, rewards: Dict[str, float], region_id: int) -> float:
        """
        Calculate aggregate reward for a region
        
        Args:
            rewards: Dictionary mapping traffic light IDs to their rewards
            region_id: Region ID
            
        Returns:
            Aggregate reward for the region
        """
        if not (0 <= region_id < len(self.regions)):
            return 0.0
            
        region_tls = self.regions[region_id]
        region_rewards = [rewards.get(tl, 0.0) for tl in region_tls]
        
        if not region_rewards:
            return 0.0
            
        # Use mean reward as regional reward
        return np.mean(region_rewards)
    
    def calculate_hierarchical_rewards(
        self,
        local_rewards: Dict[str, float],
        regional_weight: float = 0.3
    ) -> Dict[str, float]:
        """
        Calculate hierarchical rewards combining local and regional components
        
        Args:
            local_rewards: Dictionary mapping traffic light IDs to their local rewards
            regional_weight: Weight for regional reward component (0-1)
            
        Returns:
            Dictionary mapping traffic light IDs to their hierarchical rewards
        """
        hierarchical_rewards = {}
        
        # Calculate regional rewards
        regional_rewards = {
            i: self.calculate_regional_reward(local_rewards, i)
            for i in range(self.num_regions)
        }
        
        # Combine local and regional rewards for each traffic light
        for tl_id in local_rewards:
            region_id = self.get_region(tl_id)
            if region_id >= 0:
                local_reward = local_rewards[tl_id]
                regional_reward = regional_rewards[region_id]
                
                # Weighted combination
                hierarchical_rewards[tl_id] = (
                    (1 - regional_weight) * local_reward +
                    regional_weight * regional_reward
                )
            else:
                # If traffic light is not in a region, use local reward only
                hierarchical_rewards[tl_id] = local_rewards[tl_id]
        
        return hierarchical_rewards
