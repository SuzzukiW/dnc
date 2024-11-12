# experiments/scenarios/rewards/global_rewards.py

import os
import sys
import numpy as np
import traci
import sumolib
import networkx as nx
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Part A

class NetworkState:
    """Monitor and analyze global network state"""
    
    def __init__(self,
                net_file: str,
                history_length: int = 100,
                congestion_threshold: float = 0.7,
                flow_window: int = 300):  # 5-minute window for flow calculation
        """
        Initialize network state monitor
        
        Args:
            net_file: Path to SUMO network file
            history_length: Number of time steps to keep in history
            congestion_threshold: Speed ratio below which traffic is considered congested
            flow_window: Time window for calculating flow rates
        """
        self.net = sumolib.net.readNet(net_file)
        self.history_length = history_length
        self.congestion_threshold = congestion_threshold
        self.flow_window = flow_window
        
        # Initialize state tracking
        self.vehicle_states = defaultdict(dict)
        self.edge_states = defaultdict(lambda: defaultdict(list))
        self.network_states = defaultdict(list)
        
        # Flow tracking
        self.vehicle_counts = defaultdict(int)
        self.flow_history = defaultdict(list)
        self.arrival_times = defaultdict(list)
        self.departure_times = defaultdict(list)
        
        # Congestion tracking
        self.congestion_zones = set()
        self.historical_congestion = defaultdict(list)
        
        # Travel time tracking
        self.travel_times = defaultdict(list)
        self.route_times = defaultdict(dict)
        
        # Performance metrics
        self.metrics = {
            'network_speed': [],
            'network_density': [],
            'network_flow': [],
            'congestion_level': [],
            'total_waiting_time': [],
            'emissions': [],
            'throughput': [],
            'travel_time_reliability': []
        }
    
    def update_state(self):
        """Update global network state"""
        current_time = traci.simulation.getTime()
        
        # Get current vehicles
        vehicles = traci.vehicle.getIDList()
        
        # Update vehicle states
        self._update_vehicle_states(vehicles)
        
        # Update edge states
        self._update_edge_states()
        
        # Update flow rates
        self._update_flow_rates(current_time)
        
        # Update congestion zones
        self._update_congestion_zones()
        
        # Update network-wide metrics
        self._update_network_metrics(current_time)
        
        # Trim history if needed
        self._trim_history()
    
    def _update_vehicle_states(self, vehicles: List[str]):
        """Update individual vehicle states"""
        current_time = traci.simulation.getTime()
        
        for vehicle_id in vehicles:
            try:
                # Get vehicle data
                edge_id = traci.vehicle.getRoadID(vehicle_id)
                speed = traci.vehicle.getSpeed(vehicle_id)
                waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                acc = traci.vehicle.getAcceleration(vehicle_id)
                
                # Store state
                self.vehicle_states[vehicle_id] = {
                    'edge': edge_id,
                    'speed': speed,
                    'waiting_time': waiting_time,
                    'acceleration': acc,
                    'time': current_time
                }
                
                # Update travel times if vehicle completed route
                if vehicle_id in self.departure_times:
                    if edge_id.startswith(':'):  # Vehicle on intersection
                        route = traci.vehicle.getRoute(vehicle_id)
                        if edge_id == route[-1]:  # Vehicle finished route
                            travel_time = current_time - self.departure_times[vehicle_id]
                            self.travel_times[vehicle_id] = travel_time
                            
                            # Store route-specific time
                            route_id = tuple(route)
                            self.route_times[route_id][vehicle_id] = travel_time
            
            except traci.exceptions.TraCIException:
                continue
    
    def _update_edge_states(self):
        """Update edge-level states"""
        for edge in self.net.getEdges():
            edge_id = edge.getID()
            try:
                # Calculate edge metrics
                mean_speed = traci.edge.getLastStepMeanSpeed(edge_id)
                occupancy = traci.edge.getLastStepOccupancy(edge_id)
                vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
                travel_time = traci.edge.getTraveltime(edge_id)
                
                # Calculate density
                length = edge.getLength()
                density = len(vehicles) / (length / 1000)  # vehicles per km
                
                # Store states
                self.edge_states[edge_id]['speed'].append(mean_speed)
                self.edge_states[edge_id]['occupancy'].append(occupancy)
                self.edge_states[edge_id]['density'].append(density)
                self.edge_states[edge_id]['travel_time'].append(travel_time)
                
                # Keep history within limit
                if len(self.edge_states[edge_id]['speed']) > self.history_length:
                    for key in self.edge_states[edge_id]:
                        self.edge_states[edge_id][key] = \
                            self.edge_states[edge_id][key][-self.history_length:]
            
            except traci.exceptions.TraCIException:
                continue
    
    def _update_flow_rates(self, current_time: float):
        """Update traffic flow rates"""
        # Update vehicle counts
        arrived = set(traci.simulation.getArrivedIDList())
        departed = set(traci.simulation.getDepartedIDList())
        
        # Record arrival and departure times
        for vehicle_id in arrived:
            self.arrival_times[vehicle_id] = current_time
        for vehicle_id in departed:
            self.departure_times[vehicle_id] = current_time
        
        # Calculate flow rates for each edge
        for edge in self.net.getEdges():
            edge_id = edge.getID()
            try:
                # Get vehicles that passed this edge
                passed_vehicles = set(traci.edge.getLastStepVehicleIDs(edge_id))
                
                # Update count
                self.vehicle_counts[edge_id] += len(passed_vehicles)
                
                # Calculate flow rate (vehicles per hour)
                flow_rate = (self.vehicle_counts[edge_id] * 3600) / self.flow_window
                self.flow_history[edge_id].append(flow_rate)
                
                # Reset count if window passed
                if current_time % self.flow_window == 0:
                    self.vehicle_counts[edge_id] = 0
            
            except traci.exceptions.TraCIException:
                continue
    
    def _update_congestion_zones(self):
        """Update congestion zone identification"""
        self.congestion_zones.clear()
        
        for edge in self.net.getEdges():
            edge_id = edge.getID()
            
            try:
                # Get current speed ratio
                current_speed = traci.edge.getLastStepMeanSpeed(edge_id)
                max_speed = traci.edge.getMaxSpeed(edge_id)
                speed_ratio = current_speed / max_speed if max_speed > 0 else 1.0
                
                # Check for congestion
                if speed_ratio < self.congestion_threshold:
                    self.congestion_zones.add(edge_id)
                
                # Store historical congestion
                self.historical_congestion[edge_id].append(
                    1 if speed_ratio < self.congestion_threshold else 0
                )
            
            except traci.exceptions.TraCIException:
                continue
    
    def _update_network_metrics(self, current_time: float):
        """Update network-wide performance metrics"""
        try:
            # Calculate network-wide metrics
            total_speed = 0
            total_vehicles = 0
            total_waiting = 0
            total_co2 = 0
            
            for vehicle_id in traci.vehicle.getIDList():
                try:
                    total_speed += traci.vehicle.getSpeed(vehicle_id)
                    total_waiting += traci.vehicle.getWaitingTime(vehicle_id)
                    total_co2 += traci.vehicle.getCO2Emission(vehicle_id)
                    total_vehicles += 1
                except:
                    continue
            
            # Calculate averages
            avg_speed = total_speed / max(total_vehicles, 1)
            network_density = total_vehicles / (self.get_network_length() / 1000)  # veh/km
            
            # Calculate throughput
            throughput = len(traci.simulation.getArrivedIDList())
            
            # Calculate travel time reliability
            reliability = self._calculate_travel_time_reliability()
            
            # Store metrics
            self.metrics['network_speed'].append(avg_speed)
            self.metrics['network_density'].append(network_density)
            self.metrics['network_flow'].append(throughput)
            self.metrics['congestion_level'].append(len(self.congestion_zones))
            self.metrics['total_waiting_time'].append(total_waiting)
            self.metrics['emissions'].append(total_co2)
            self.metrics['throughput'].append(throughput)
            self.metrics['travel_time_reliability'].append(reliability)
        
        except Exception as e:
            print(f"Error updating network metrics: {e}")

# Part B

class GlobalRewardCalculator:
    """Calculate global rewards based on network-wide performance"""
    
    def __init__(self,
                net_file: str,
                baseline_window: int = 600,    # 10-minute window for baseline
                smoothing_factor: float = 0.3,  # For exponential smoothing
                performance_weights: Optional[Dict[str, float]] = None):
        """
        Initialize global reward calculator
        
        Args:
            net_file: Path to SUMO network file
            baseline_window: Time window for baseline calculation
            smoothing_factor: Smoothing factor for metrics
            performance_weights: Custom weights for performance metrics
        """
        # Initialize network state monitor
        self.network_state = NetworkState(net_file)
        
        # Configuration
        self.baseline_window = baseline_window
        self.smoothing_factor = smoothing_factor
        
        # Default performance weights if not provided
        self.weights = performance_weights or {
            'flow_rate': 0.25,         # Network throughput
            'waiting_time': 0.20,      # Total waiting time
            'congestion': 0.20,        # Congestion levels
            'travel_time': 0.15,       # Travel time reliability
            'emissions': 0.10,         # Environmental impact
            'coordination': 0.10       # Signal coordination
        }
        
        # Initialize baseline metrics
        self.baseline_metrics = defaultdict(list)
        
        # Store historical rewards
        self.reward_history = []
        self.component_history = defaultdict(list)
        
        # Performance tracking
        self.best_performance = None
        self.worst_performance = None
        self.performance_trend = []
    
    def _calculate_flow_reward(self) -> float:
        """Calculate reward component based on network flow"""
        current_flow = np.mean(self.network_state.metrics['network_flow'][-10:])
        baseline_flow = np.mean(self.baseline_metrics['flow'])
        
        if baseline_flow == 0:
            return 0.0
        
        # Calculate relative improvement
        flow_ratio = current_flow / baseline_flow
        
        # Reward function with diminishing returns
        return np.tanh(flow_ratio - 1)
    
    def _calculate_waiting_time_reward(self) -> float:
        """Calculate reward component based on waiting times"""
        current_waiting = self.network_state.metrics['total_waiting_time'][-1]
        baseline_waiting = np.mean(self.baseline_metrics['waiting_time'])
        
        if baseline_waiting == 0:
            return 0.0
        
        # Calculate relative improvement (negative because lower is better)
        waiting_ratio = current_waiting / baseline_waiting
        
        # Penalty function
        return -np.tanh(waiting_ratio - 1)
    
    def _calculate_congestion_reward(self) -> float:
        """Calculate reward component based on congestion levels"""
        current_congestion = len(self.network_state.congestion_zones)
        total_edges = len(self.network_state.net.getEdges())
        
        if total_edges == 0:
            return 0.0
        
        # Calculate congestion ratio
        congestion_ratio = current_congestion / total_edges
        
        # Penalty function with threshold
        threshold = 0.3  # Consider 30% congested edges as threshold
        if congestion_ratio <= threshold:
            return 1.0 - (congestion_ratio / threshold)
        else:
            return -((congestion_ratio - threshold) / (1 - threshold))
    
    def _calculate_travel_time_reward(self) -> float:
        """Calculate reward component based on travel time reliability"""
        reliability = self.network_state.metrics['travel_time_reliability'][-1]
        baseline_reliability = np.mean(self.baseline_metrics['reliability'])
        
        if baseline_reliability == 0:
            return 0.0
        
        # Calculate relative improvement
        reliability_ratio = reliability / baseline_reliability
        
        # Reward function
        return np.tanh(reliability_ratio - 1)
    
    def _calculate_emissions_reward(self) -> float:
        """Calculate reward component based on emissions"""
        current_emissions = self.network_state.metrics['emissions'][-1]
        baseline_emissions = np.mean(self.baseline_metrics['emissions'])
        
        if baseline_emissions == 0:
            return 0.0
        
        # Calculate relative improvement (negative because lower is better)
        emission_ratio = current_emissions / baseline_emissions
        
        # Penalty function
        return -np.tanh(emission_ratio - 1)
    
    def _calculate_coordination_reward(self) -> float:
        """Calculate reward component based on signal coordination"""
        coordination_score = 0.0
        total_pairs = 0
        
        # Get all traffic lights
        traffic_lights = traci.trafficlight.getIDList()
        
        for i, tl1 in enumerate(traffic_lights):
            for tl2 in traffic_lights[i+1:]:
                try:
                    # Get phases for both lights
                    phase1 = traci.trafficlight.getPhase(tl1)
                    phase2 = traci.trafficlight.getPhase(tl2)
                    
                    # Get states
                    state1 = traci.trafficlight.getRedYellowGreenState(tl1)
                    state2 = traci.trafficlight.getRedYellowGreenState(tl2)
                    
                    # Calculate coordination score
                    green1 = sum(1 for c in state1 if c.lower() == 'g')
                    green2 = sum(1 for c in state2 if c.lower() == 'g')
                    
                    # Reward if green ratios are similar (indicating coordination)
                    ratio1 = green1 / len(state1) if state1 else 0
                    ratio2 = green2 / len(state2) if state2 else 0
                    
                    # Add to score if ratios are similar
                    if abs(ratio1 - ratio2) < 0.2:  # 20% threshold
                        coordination_score += 1
                    
                    total_pairs += 1
                
                except traci.exceptions.TraCIException:
                    continue
        
        if total_pairs == 0:
            return 0.0
        
        return coordination_score / total_pairs
    
    def calculate_reward(self) -> Tuple[float, Dict[str, float]]:
        """
        Calculate global reward based on all components
        
        Returns:
            Tuple of (final_reward, component_rewards)
        """
        try:
            # Update network state
            self.network_state.update_state()
            
            # Calculate reward components
            components = {
                'flow_rate': self._calculate_flow_reward(),
                'waiting_time': self._calculate_waiting_time_reward(),
                'congestion': self._calculate_congestion_reward(),
                'travel_time': self._calculate_travel_time_reward(),
                'emissions': self._calculate_emissions_reward(),
                'coordination': self._calculate_coordination_reward()
            }
            
            # Calculate weighted sum
            reward = sum(self.weights[component] * value 
                        for component, value in components.items())
            
            # Apply smoothing
            if self.reward_history:
                reward = (self.smoothing_factor * reward + 
                         (1 - self.smoothing_factor) * self.reward_history[-1])
            
            # Update history
            self.reward_history.append(reward)
            for component, value in components.items():
                self.component_history[component].append(value)
            
            # Update performance tracking
            self._update_performance_tracking(reward)
            
            # Clip final reward
            reward = np.clip(reward, -1.0, 1.0)
            
            return reward, components
        
        except Exception as e:
            print(f"Error calculating global reward: {e}")
            return 0.0, {}
    
    def _update_performance_tracking(self, reward: float):
        """Update performance tracking metrics"""
        # Update best/worst performance
        if self.best_performance is None or reward > self.best_performance:
            self.best_performance = reward
        if self.worst_performance is None or reward < self.worst_performance:
            self.worst_performance = reward
        
        # Update performance trend
        self.performance_trend.append(reward)
        if len(self.performance_trend) > 100:  # Keep last 100 steps
            self.performance_trend.pop(0)
    
    def update_baseline(self):
        """Update baseline metrics"""
        if len(self.network_state.metrics['network_flow']) >= self.baseline_window:
            self.baseline_metrics = {
                'flow': self.network_state.metrics['network_flow'][-self.baseline_window:],
                'waiting_time': self.network_state.metrics['total_waiting_time'][-self.baseline_window:],
                'reliability': self.network_state.metrics['travel_time_reliability'][-self.baseline_window:],
                'emissions': self.network_state.metrics['emissions'][-self.baseline_window:]
            }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.performance_trend:
            return {}
        
        return {
            'best_reward': self.best_performance,
            'worst_reward': self.worst_performance,
            'average_reward': np.mean(self.performance_trend),
            'current_trend': np.polyfit(range(len(self.performance_trend)),
                                      self.performance_trend, 1)[0]
        }
    
    def save_history(self, filepath: str):
        """Save reward history to file"""
        import json
        
        history = {
            'rewards': self.reward_history,
            'components': dict(self.component_history),
            'performance_stats': self.get_performance_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=4)

def main():
    """Example usage and testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test global reward calculation')
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
    reward_calc = GlobalRewardCalculator(args.net_file)
    
    # Run simulation for a few steps
    try:
        print("\nTesting global reward calculation...")
        for step in range(100):
            reward, components = reward_calc.calculate_reward()
            
            if step % 10 == 0:
                print(f"\nStep {step}")
                print(f"Global Reward: {reward:.3f}")
                print("Reward Components:")
                for component, value in components.items():
                    print(f"  {component}: {value:.3f}")
            
            traci.simulationStep()
        
        # Save history
        reward_calc.save_history("global_rewards_history.json")
        
        # Print performance stats
        print("\nPerformance Statistics:")
        for metric, value in reward_calc.get_performance_stats().items():
            print(f"  {metric}: {value:.3f}")
    
    finally:
        traci.close()

if __name__ == "__main__":
    main()