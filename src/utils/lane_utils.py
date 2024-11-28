"""Utility functions for handling lane IDs in traffic simulation."""

import re
import traci
from typing import List, Optional, Set, Dict, Any

class LaneProcessor:
    def __init__(self, connection=None):
        """
        Initialize LaneProcessor with optional TraCI connection.
        
        Args:
            connection: Optional TraCI connection for advanced lane processing
        """
        # Set of lane IDs to ignore
        self.ignored_lanes: Set[str] = set()
        
        # Compiled regex patterns
        self.cluster_pattern = re.compile(r':cluster_.*?_[cw]\d+_\d+$')
        
        # TraCI connection for advanced lane retrieval
        self.connection = connection
        
    def get_lane_details(self, lane_id: str) -> Dict[str, Any]:
        """
        Retrieve detailed information about a lane using TraCI.
        
        Args:
            lane_id: The ID of the lane to retrieve details for
        
        Returns:
            Dict of lane details with various attributes
        """
        if not self.connection:
            return {}
        
        try:
            return {
                'edge_id': self.connection.lane.getEdgeID(lane_id),
                'length': self.connection.lane.getLength(lane_id),
                'max_speed': self.connection.lane.getMaxSpeed(lane_id),
                'width': self.connection.lane.getWidth(lane_id),
                'vehicle_count': self.connection.lane.getLastStepVehicleNumber(lane_id),
                'mean_speed': self.connection.lane.getLastStepMeanSpeed(lane_id),
                'occupancy': self.connection.lane.getLastStepOccupancy(lane_id),
                'halting_vehicles': self.connection.lane.getLastStepHaltingNumber(lane_id)
            }
        except traci.exceptions.TraCIException as e:
            print(f"Error retrieving lane details for {lane_id}: {e}")
            return {}
        
    def should_process_lane(self, lane_id: str) -> bool:
        """
        Determine if a lane should be processed with enhanced logic.
        
        Args:
            lane_id: The ID of the lane to check
            
        Returns:
            bool: True if the lane should be processed, False otherwise
        """
        # Skip if lane is in ignored set
        if lane_id in self.ignored_lanes:
            return False
        
        # Special handling for cluster lanes
        if self.cluster_pattern.match(lane_id):
            # Additional checks for cluster lanes
            if self.connection:
                try:
                    # Check if the cluster lane has any vehicles or is significant
                    vehicle_count = self.connection.lane.getLastStepVehicleNumber(lane_id)
                    if vehicle_count > 0:
                        print(f"Processing significant cluster lane: {lane_id} (vehicles: {vehicle_count})")
                        return True
                except traci.exceptions.TraCIException:
                    # If retrieval fails, default to processing
                    print(f"Attempting to process cluster lane: {lane_id}")
                    return True
            
            # Without connection, process all cluster lanes
            return True
        
        return True
    
    def clean_lane_id(self, lane_id: str) -> Optional[str]:
        """
        Clean and validate a lane ID with enhanced processing.
        
        Args:
            lane_id: The ID of the lane to clean
            
        Returns:
            Optional[str]: Cleaned lane ID if valid, None otherwise
        """
        # Remove any leading/trailing colons or whitespace
        cleaned_id = lane_id.strip(':').strip()
        
        # Preserve cluster lanes and internal lanes
        if cleaned_id.startswith(('cluster_', ':')) or cleaned_id:
            return cleaned_id
        
        return None
    
    def get_valid_lanes(self, lane_ids: List[str]) -> List[str]:
        """
        Filter and return only valid lane IDs with detailed processing.
        
        Args:
            lane_ids: List of lane IDs to filter
            
        Returns:
            List[str]: List of valid lane IDs
        """
        valid_lanes = []
        
        for lane_id in lane_ids:
            # Enhanced lane processing
            if self.should_process_lane(lane_id):
                cleaned_id = self.clean_lane_id(lane_id)
                if cleaned_id:
                    valid_lanes.append(cleaned_id)
        
        return valid_lanes
    
    def reset(self):
        """Reset the processor state."""
        self.ignored_lanes.clear()
    
    def log_lane_details(self, lane_id: str):
        """
        Log detailed information about a lane.
        
        Args:
            lane_id: The ID of the lane to log details for
        """
        details = self.get_lane_details(lane_id)
        if details:
            print(f"Lane Details for {lane_id}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        else:
            print(f"No details available for lane: {lane_id}")
