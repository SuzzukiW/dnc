# experiments/scenarios/rewards/local_rewards.py

import os
import sys
import numpy as np
import traci
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class LocalRewardCalculator:
    """Calculate local rewards for individual traffic light agents"""
    
    def __init__(self,
                max_waiting_time: float = 180.0,    # Maximum waiting time (seconds)
                max_queue_length: int = 20,         # Maximum queue length (vehicles)
                min_green_util: float = 0.3,        # Minimum green time utilization
                reward_weights: Optional[Dict[str, float]] = None):
        """
        Initialize reward calculator with configurable parameters
        
        Args:
            max_waiting_time: Cap for waiting time consideration
            max_queue_length: Cap for queue length consideration
            min_green_util: Minimum expected green time utilization
            reward_weights: Custom weights for different reward components
        """
        self.max_waiting_time = max_waiting_time
        self.max_queue_length = max_queue_length
        self.min_green_util = min_green_util
        
        # Default reward weights
        self.weights = reward_weights or {
            'waiting_time': 0.35,      # Penalty for vehicle waiting time
            'queue_length': 0.25,      # Penalty for queue formation
            'green_utilization': 0.15, # Reward for efficient green time use
            'speed': 0.15,            # Reward for maintaining traffic flow
            'emergency': 0.10         # Priority for emergency vehicles
        }
        
        # Store previous metrics for improvement calculation
        self.previous_metrics = defaultdict(dict)
        
        # Initialize emergency vehicle tracking
        self.emergency_vehicles = set()
        
    def _get_lane_waiting_time(self, lane_id: str) -> float:
        """Get total waiting time for vehicles on a lane"""
        try:
            return min(traci.lane.getWaitingTime(lane_id), self.max_waiting_time)
        except traci.exceptions.TraCIException:
            return 0.0
    
    def _get_lane_queue_length(self, lane_id: str) -> int:
        """Get number of queued vehicles on a lane"""
        try:
            return min(traci.lane.getLastStepHaltingNumber(lane_id), 
                      self.max_queue_length)
        except traci.exceptions.TraCIException:
            return 0
    
    def _get_lane_speed_score(self, lane_id: str) -> float:
        """Calculate speed score for a lane"""
        try:
            current_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            max_speed = traci.lane.getMaxSpeed(lane_id)
            
            if max_speed <= 0:
                return 0.0
            
            return current_speed / max_speed
        except traci.exceptions.TraCIException:
            return 0.0
    
    def _get_green_utilization(self, 
                             tl_id: str, 
                             controlled_lanes: List[str]) -> float:
        """Calculate green time utilization"""
        try:
            # Get current phase info
            phase_state = traci.trafficlight.getRedYellowGreenState(tl_id)
            green_lanes = [lane for i, lane in enumerate(controlled_lanes)
                         if phase_state[i].lower() == 'g']
            
            if not green_lanes:
                return 0.0
            
            # Calculate utilization
            utilized_lanes = sum(1 for lane in green_lanes
                              if traci.lane.getLastStepVehicleNumber(lane) > 0)
            
            return utilized_lanes / len(green_lanes)
        except traci.exceptions.TraCIException:
            return 0.0
    
    def _check_emergency_vehicles(self, 
                                tl_id: str, 
                                controlled_lanes: List[str]) -> Tuple[bool, float]:
        """Check for emergency vehicles and calculate priority score"""
        try:
            emergency_present = False
            total_waiting_time = 0.0
            
            for lane in controlled_lanes:
                vehicle_list = traci.lane.getLastStepVehicleIDs(lane)
                
                for vehicle in vehicle_list:
                    try:
                        veh_type = traci.vehicle.getVehicleClass(vehicle)
                        if veh_type == "emergency":
                            emergency_present = True
                            # Track emergency vehicle
                            self.emergency_vehicles.add(vehicle)
                            # Add waiting time for priority
                            total_waiting_time += traci.vehicle.getWaitingTime(vehicle)
                    except:
                        continue
            
            # Remove emergency vehicles that have left the simulation
            self.emergency_vehicles = {v for v in self.emergency_vehicles 
                                    if v in traci.vehicle.getIDList()}
            
            priority_score = min(total_waiting_time / self.max_waiting_time, 1.0)
            
            return emergency_present, priority_score
        
        except traci.exceptions.TraCIException:
            return False, 0.0
    
    def calculate_reward(self, tl_id: str, controlled_lanes: List[str]) -> float:
        """
        Calculate comprehensive local reward for a traffic light
        
        Args:
            tl_id: Traffic light ID
            controlled_lanes: List of lanes controlled by this traffic light
            
        Returns:
            float: Calculated reward value
        """
        if not controlled_lanes:
            return 0.0
        
        # Calculate basic metrics
        lane_metrics = {lane: {
            'waiting_time': self._get_lane_waiting_time(lane),
            'queue_length': self._get_lane_queue_length(lane),
            'speed_score': self._get_lane_speed_score(lane)
        } for lane in controlled_lanes}
        
        # Calculate average metrics
        avg_metrics = {
            'waiting_time': np.mean([m['waiting_time'] 
                                   for m in lane_metrics.values()]),
            'queue_length': np.mean([m['queue_length'] 
                                   for m in lane_metrics.values()]),
            'speed_score': np.mean([m['speed_score'] 
                                  for m in lane_metrics.values()])
        }
        
        # Get green time utilization
        green_util = self._get_green_utilization(tl_id, controlled_lanes)
        
        # Check for emergency vehicles
        has_emergency, emergency_score = self._check_emergency_vehicles(
            tl_id, controlled_lanes)
        
        # Calculate reward components
        reward_components = {
            # Waiting time component (negative reward)
            'waiting_time': -avg_metrics['waiting_time'] / self.max_waiting_time,
            
            # Queue length component (negative reward)
            'queue_length': -avg_metrics['queue_length'] / self.max_queue_length,
            
            # Green utilization component (positive reward)
            'green_utilization': max(0, green_util - self.min_green_util),
            
            # Speed component (positive reward)
            'speed': avg_metrics['speed_score'],
            
            # Emergency vehicle component (negative reward for waiting)
            'emergency': -emergency_score if has_emergency else 0.0
        }
        
        # Calculate weighted sum of components
        reward = sum(self.weights[component] * value 
                    for component, value in reward_components.items())
        
        # Add improvement bonuses
        if tl_id in self.previous_metrics:
            prev = self.previous_metrics[tl_id]
            
            # Check for improvements
            waiting_improved = (avg_metrics['waiting_time'] < 
                              prev.get('waiting_time', float('inf')))
            queue_improved = (avg_metrics['queue_length'] < 
                            prev.get('queue_length', float('inf')))
            speed_improved = (avg_metrics['speed_score'] > 
                            prev.get('speed_score', 0))
            
            # Add bonus for multiple improvements
            if sum([waiting_improved, queue_improved, speed_improved]) >= 2:
                reward += 0.2
            
            # Add bonus for emergency vehicle clearance
            if has_emergency and emergency_score < prev.get('emergency_score', float('inf')):
                reward += 0.3
        
        # Store current metrics for next comparison
        self.previous_metrics[tl_id] = {
            'waiting_time': avg_metrics['waiting_time'],
            'queue_length': avg_metrics['queue_length'],
            'speed_score': avg_metrics['speed_score'],
            'emergency_score': emergency_score if has_emergency else 0.0
        }
        
        # Scale reward to [-1, 1] range
        return np.clip(reward, -1.0, 1.0)
    
    def get_reward_components(self, 
                            tl_id: str, 
                            controlled_lanes: List[str]) -> Dict[str, float]:
        """
        Get detailed breakdown of reward components
        
        Args:
            tl_id: Traffic light ID
            controlled_lanes: List of controlled lanes
            
        Returns:
            Dict containing individual reward components
        """
        lane_metrics = {lane: {
            'waiting_time': self._get_lane_waiting_time(lane),
            'queue_length': self._get_lane_queue_length(lane),
            'speed_score': self._get_lane_speed_score(lane)
        } for lane in controlled_lanes}
        
        green_util = self._get_green_utilization(tl_id, controlled_lanes)
        has_emergency, emergency_score = self._check_emergency_vehicles(
            tl_id, controlled_lanes)
        
        components = {
            'waiting_time': -np.mean([m['waiting_time'] 
                                    for m in lane_metrics.values()]) / self.max_waiting_time,
            'queue_length': -np.mean([m['queue_length'] 
                                    for m in lane_metrics.values()]) / self.max_queue_length,
            'green_utilization': max(0, green_util - self.min_green_util),
            'speed': np.mean([m['speed_score'] 
                            for m in lane_metrics.values()]),
            'emergency': -emergency_score if has_emergency else 0.0
        }
        
        return components
    
    def save_metrics(self, filepath: str):
        """Save current metrics to file"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.previous_metrics, f, indent=4)
    
    def load_metrics(self, filepath: str):
        """Load metrics from file"""
        import json
        
        with open(filepath, 'r') as f:
            self.previous_metrics = defaultdict(dict, json.load(f))

def main():
    """Example usage and testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test local reward calculation')
    parser.add_argument('--net-file', required=True, help='Input network file')
    parser.add_argument('--route-file', required=True, help='Input route file')
    
    args = parser.parse_args()
    
    # Start SUMO
    sumo_cmd = [
        'sumo',
        '-n', args.net_file,
        '-r', args.route_file,
        '--no-warnings',
    ]
    
    traci.start(sumo_cmd)
    
    # Initialize reward calculator
    reward_calc = LocalRewardCalculator()
    
    # Run simulation for a few steps
    try:
        print("\nTesting reward calculation...")
        for step in range(100):
            # Get traffic lights
            for tl_id in traci.trafficlight.getIDList():
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                
                # Calculate reward
                reward = reward_calc.calculate_reward(tl_id, controlled_lanes)
                
                # Get detailed components
                components = reward_calc.get_reward_components(tl_id, controlled_lanes)
                
                if step % 10 == 0:
                    print(f"\nStep {step}, Traffic Light {tl_id}")
                    print(f"Total Reward: {reward:.3f}")
                    print("Reward Components:")
                    for component, value in components.items():
                        print(f"  {component}: {value:.3f}")
            
            traci.simulationStep()
    
    finally:
        traci.close()

if __name__ == "__main__":
    main()