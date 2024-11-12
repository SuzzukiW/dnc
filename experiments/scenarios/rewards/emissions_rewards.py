# experiments/scenarios/rewards/emissions_rewards.py

import os
import sys
import numpy as np
import traci
import sumolib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Part A

class EmissionMonitor:
    """Monitor and analyze vehicle emissions in the network"""
    
    # Emission thresholds (mg/s) based on EURO 6 standards
    EMISSION_THRESHOLDS = {
        'CO2': 130.0,     # g/km (converted to mg/s in calculations)
        'CO': 1.0,        # g/km
        'NOx': 0.06,      # g/km
        'HC': 0.1,        # g/km
        'PMx': 0.005      # g/km
    }
    
    # Weights for different emission types in composite score
    EMISSION_WEIGHTS = {
        'CO2': 0.4,   # Primary greenhouse gas
        'CO': 0.15,   # Carbon monoxide
        'NOx': 0.2,   # Nitrogen oxides
        'HC': 0.15,   # Hydrocarbons
        'PMx': 0.1    # Particulate matter
    }
    
    def __init__(self,
                net_file: str,
                history_length: int = 100,
                window_size: int = 60,     # 1-minute window for averaging
                spatial_resolution: float = 50.0):  # 50m grid cells
        """
        Initialize emission monitor
        
        Args:
            net_file: Path to SUMO network file
            history_length: Number of time steps to keep in history
            window_size: Time window for averaging emissions
            spatial_resolution: Size of grid cells for spatial analysis
        """
        self.net = sumolib.net.readNet(net_file)
        self.history_length = history_length
        self.window_size = window_size
        self.spatial_resolution = spatial_resolution
        
        # Initialize tracking structures
        self.vehicle_emissions = defaultdict(lambda: defaultdict(list))
        self.edge_emissions = defaultdict(lambda: defaultdict(list))
        self.grid_emissions = defaultdict(lambda: defaultdict(float))
        self.network_emissions = defaultdict(list)
        
        # Track hotspots and patterns
        self.emission_hotspots = set()
        self.temporal_patterns = defaultdict(lambda: defaultdict(list))
        
        # Vehicle type specific tracking
        self.type_emissions = defaultdict(lambda: defaultdict(list))
        
        # Initialize spatial grid
        self.grid = self._initialize_spatial_grid()
        
        # Performance metrics
        self.metrics = {
            'total_emissions': defaultdict(list),
            'average_emissions': defaultdict(list),
            'hotspot_count': [],
            'violation_rate': [],
            'emission_reduction': []
        }
        
        # Baseline tracking
        self.baseline_emissions = None
        self.baseline_samples = []
    
    def _initialize_spatial_grid(self) -> Dict[Tuple[int, int], List[str]]:
        """Initialize spatial grid for emission mapping"""
        grid = defaultdict(list)
        
        # Get network boundaries
        bounds = self.net.getBoundary()
        min_x, min_y = bounds[0], bounds[1]
        
        # Map edges to grid cells
        for edge in self.net.getEdges():
            shape = edge.getShape()
            for i in range(len(shape) - 1):
                # Get line segment
                x1, y1 = shape[i]
                x2, y2 = shape[i + 1]
                
                # Calculate grid cells this segment passes through
                cell_x1 = int((x1 - min_x) / self.spatial_resolution)
                cell_y1 = int((y1 - min_y) / self.spatial_resolution)
                cell_x2 = int((x2 - min_x) / self.spatial_resolution)
                cell_y2 = int((y2 - min_y) / self.spatial_resolution)
                
                # Add edge to all cells it passes through
                for x in range(min(cell_x1, cell_x2), max(cell_x1, cell_x2) + 1):
                    for y in range(min(cell_y1, cell_y2), max(cell_y1, cell_y2) + 1):
                        grid[(x, y)].append(edge.getID())
        
        return grid
    
    def _calculate_vehicle_emissions(self, vehicle_id: str) -> Dict[str, float]:
        """Calculate emissions for a single vehicle"""
        try:
            # Get vehicle state
            speed = traci.vehicle.getSpeed(vehicle_id)
            acceleration = traci.vehicle.getAcceleration(vehicle_id)
            slope = 0.0  # Could be calculated from edge geometry if needed
            
            # Get vehicle type information
            v_type = traci.vehicle.getTypeID(vehicle_id)
            vehicle_class = traci.vehicle.getVehicleClass(vehicle_id)
            
            # Get emission values from SUMO
            emissions = {
                'CO2': traci.vehicle.getCO2Emission(vehicle_id),
                'CO': traci.vehicle.getCOEmission(vehicle_id),
                'NOx': traci.vehicle.getNOxEmission(vehicle_id),
                'HC': traci.vehicle.getHCEmission(vehicle_id),
                'PMx': traci.vehicle.getPMxEmission(vehicle_id)
            }
            
            # Store by vehicle type
            for emission_type, value in emissions.items():
                self.type_emissions[v_type][emission_type].append(value)
            
            return emissions
        
        except traci.exceptions.TraCIException:
            return {key: 0.0 for key in self.EMISSION_THRESHOLDS.keys()}
    
    def _update_spatial_emissions(self):
        """Update spatial emission distribution"""
        # Reset grid emissions
        self.grid_emissions.clear()
        
        # Get all vehicles
        vehicles = traci.vehicle.getIDList()
        
        for vehicle_id in vehicles:
            try:
                # Get vehicle position
                x, y = traci.vehicle.getPosition(vehicle_id)
                
                # Get network boundaries
                bounds = self.net.getBoundary()
                min_x, min_y = bounds[0], bounds[1]
                
                # Calculate grid cell
                cell_x = int((x - min_x) / self.spatial_resolution)
                cell_y = int((y - min_y) / self.spatial_resolution)
                
                # Get vehicle emissions
                emissions = self._calculate_vehicle_emissions(vehicle_id)
                
                # Add to grid cell
                for emission_type, value in emissions.items():
                    self.grid_emissions[(cell_x, cell_y)][emission_type] += value
            
            except traci.exceptions.TraCIException:
                continue
    
    def _identify_hotspots(self, threshold_factor: float = 2.0):
        """Identify emission hotspots"""
        self.emission_hotspots.clear()
        
        # Calculate average emissions per grid cell
        avg_emissions = defaultdict(float)
        for cell, emissions in self.grid_emissions.items():
            # Calculate weighted sum of emissions
            total = sum(value * self.EMISSION_WEIGHTS[e_type]
                       for e_type, value in emissions.items())
            avg_emissions[cell] = total
        
        if not avg_emissions:
            return
        
        # Calculate network-wide average
        network_avg = np.mean(list(avg_emissions.values()))
        
        # Identify hotspots
        for cell, value in avg_emissions.items():
            if value > network_avg * threshold_factor:
                self.emission_hotspots.add(cell)
    
    def _update_temporal_patterns(self):
        """Update temporal emission patterns"""
        current_time = traci.simulation.getTime()
        hour = int(current_time / 3600) % 24
        
        # Get current network-wide emissions
        total_emissions = {
            e_type: sum(self.grid_emissions[cell][e_type]
                       for cell in self.grid_emissions)
            for e_type in self.EMISSION_THRESHOLDS.keys()
        }
        
        # Store in temporal patterns
        for e_type, value in total_emissions.items():
            self.temporal_patterns[hour][e_type].append(value)
    
    def update_state(self):
        """Update emission monitoring state"""
        try:
            # Update vehicle and spatial emissions
            vehicles = traci.vehicle.getIDList()
            
            # Track vehicle emissions
            for vehicle_id in vehicles:
                emissions = self._calculate_vehicle_emissions(vehicle_id)
                edge_id = traci.vehicle.getRoadID(vehicle_id)
                
                # Store vehicle emissions
                for e_type, value in emissions.items():
                    self.vehicle_emissions[vehicle_id][e_type].append(value)
                    
                    # Store edge emissions if on regular edge
                    if not edge_id.startswith(':'):
                        self.edge_emissions[edge_id][e_type].append(value)
            
            # Update spatial distribution
            self._update_spatial_emissions()
            
            # Identify hotspots
            self._identify_hotspots()
            
            # Update temporal patterns
            self._update_temporal_patterns()
            
            # Update metrics
            self._update_metrics()
            
            # Trim history if needed
            self._trim_history()
        
        except Exception as e:
            print(f"Error updating emission state: {e}")
    
    def _update_metrics(self):
        """Update emission metrics"""
        # Calculate total emissions
        for e_type in self.EMISSION_THRESHOLDS.keys():
            total = sum(
                emissions[e_type][-1] if emissions[e_type] else 0
                for emissions in self.vehicle_emissions.values()
            )
            self.metrics['total_emissions'][e_type].append(total)
            
            # Calculate average per vehicle
            num_vehicles = len(self.vehicle_emissions)
            avg = total / num_vehicles if num_vehicles > 0 else 0
            self.metrics['average_emissions'][e_type].append(avg)
        
        # Update hotspot count
        self.metrics['hotspot_count'].append(len(self.emission_hotspots))
        
        # Calculate violation rate
        violations = 0
        total_checks = 0
        for vehicle_id in self.vehicle_emissions:
            for e_type, values in self.vehicle_emissions[vehicle_id].items():
                if values:  # If we have measurements
                    total_checks += 1
                    if values[-1] > self.EMISSION_THRESHOLDS[e_type]:
                        violations += 1
        
        violation_rate = violations / total_checks if total_checks > 0 else 0
        self.metrics['violation_rate'].append(violation_rate)
        
        # Calculate emission reduction (compared to baseline)
        if self.baseline_emissions is None and len(self.metrics['total_emissions']['CO2']) > self.window_size:
            # Set baseline after initial period
            self.baseline_emissions = {
                e_type: np.mean(values[-self.window_size:])
                for e_type, values in self.metrics['total_emissions'].items()
            }
        
        if self.baseline_emissions is not None:
            current_total = sum(
                values[-1] * self.EMISSION_WEIGHTS[e_type]
                for e_type, values in self.metrics['total_emissions'].items()
            )
            baseline_total = sum(
                value * self.EMISSION_WEIGHTS[e_type]
                for e_type, value in self.baseline_emissions.items()
            )
            
            reduction = (baseline_total - current_total) / baseline_total if baseline_total > 0 else 0
            self.metrics['emission_reduction'].append(reduction)
    
    def _trim_history(self):
        """Trim historical data to maintain memory usage"""
        # Trim vehicle emissions
        for vehicle_id in self.vehicle_emissions:
            for e_type in self.vehicle_emissions[vehicle_id]:
                if len(self.vehicle_emissions[vehicle_id][e_type]) > self.history_length:
                    self.vehicle_emissions[vehicle_id][e_type] = \
                        self.vehicle_emissions[vehicle_id][e_type][-self.history_length:]
        
        # Trim edge emissions
        for edge_id in self.edge_emissions:
            for e_type in self.edge_emissions[edge_id]:
                if len(self.edge_emissions[edge_id][e_type]) > self.history_length:
                    self.edge_emissions[edge_id][e_type] = \
                        self.edge_emissions[edge_id][e_type][-self.history_length:]
        
        # Trim metrics
        for metric_type in self.metrics:
            if isinstance(self.metrics[metric_type], dict):
                for key in self.metrics[metric_type]:
                    if len(self.metrics[metric_type][key]) > self.history_length:
                        self.metrics[metric_type][key] = \
                            self.metrics[metric_type][key][-self.history_length:]
            else:
                if len(self.metrics[metric_type]) > self.history_length:
                    self.metrics[metric_type] = \
                        self.metrics[metric_type][-self.history_length:]

# Part B

class EmissionRewardCalculator:
    """Calculate rewards based on emission patterns and reductions"""
    
    def __init__(self,
                net_file: str,
                reward_weights: Optional[Dict[str, float]] = None,
                spatial_weight: float = 0.3,
                temporal_weight: float = 0.3,
                reduction_weight: float = 0.4,
                smoothing_factor: float = 0.2):
        """
        Initialize emission reward calculator
        
        Args:
            net_file: Path to SUMO network file
            reward_weights: Custom weights for different emission types
            spatial_weight: Weight for spatial distribution component
            temporal_weight: Weight for temporal patterns component
            reduction_weight: Weight for emission reduction component
            smoothing_factor: Exponential smoothing factor
        """
        # Initialize emission monitor
        self.monitor = EmissionMonitor(net_file)
        
        # Weights configuration
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.reduction_weight = reduction_weight
        self.smoothing_factor = smoothing_factor
        
        # Custom emission weights or use defaults from monitor
        self.emission_weights = reward_weights or self.monitor.EMISSION_WEIGHTS
        
        # Store historical rewards
        self.reward_history = []
        self.component_history = defaultdict(list)
        
        # Performance tracking
        self.best_reward = None
        self.worst_reward = None
        self.reward_trend = []
        
        # Additional tracking
        self.violation_history = []
        self.hotspot_history = []
        self.reduction_history = []
    
    def _calculate_spatial_reward(self, tl_id: str) -> float:
        """Calculate reward component based on spatial emission distribution"""
        try:
            # Get controlled lanes
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            # Get grid cells for these lanes
            relevant_cells = set()
            for lane in controlled_lanes:
                edge = lane.split('_')[0]  # Get edge ID from lane ID
                for cell, edges in self.monitor.grid.items():
                    if edge in edges:
                        relevant_cells.add(cell)
            
            if not relevant_cells:
                return 0.0
            
            # Calculate emission scores for relevant cells
            cell_scores = []
            for cell in relevant_cells:
                cell_emissions = self.monitor.grid_emissions.get(cell, {})
                if cell_emissions:
                    # Calculate weighted emission score for cell
                    score = sum(value * self.emission_weights[e_type]
                              for e_type, value in cell_emissions.items())
                    cell_scores.append(score)
            
            if not cell_scores:
                return 0.0
            
            # Calculate metrics
            avg_score = np.mean(cell_scores)
            max_score = np.max(cell_scores)
            hotspot_ratio = sum(1 for cell in relevant_cells 
                              if cell in self.monitor.emission_hotspots) / len(relevant_cells)
            
            # Combine metrics (lower is better)
            spatial_score = (0.4 * (avg_score / self.monitor.EMISSION_THRESHOLDS['CO2']) +
                           0.3 * (max_score / self.monitor.EMISSION_THRESHOLDS['CO2']) +
                           0.3 * hotspot_ratio)
            
            # Convert to reward (-1 to 1)
            return -np.tanh(spatial_score)
        
        except Exception as e:
            print(f"Error calculating spatial reward: {e}")
            return 0.0
    
    def _calculate_temporal_reward(self, tl_id: str) -> float:
        """Calculate reward component based on temporal emission patterns"""
        try:
            current_time = traci.simulation.getTime()
            hour = int(current_time / 3600) % 24
            
            # Get controlled lanes
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            if not controlled_lanes:
                return 0.0
            
            # Calculate current emissions for controlled area
            current_emissions = defaultdict(float)
            for lane in controlled_lanes:
                if not lane.startswith(':'):  # Skip internal lanes
                    edge = lane.split('_')[0]
                    if edge in self.monitor.edge_emissions:
                        for e_type, values in self.monitor.edge_emissions[edge].items():
                            if values:
                                current_emissions[e_type] += values[-1]
            
            # Compare with historical patterns
            pattern_scores = []
            for e_type, value in current_emissions.items():
                if self.monitor.temporal_patterns[hour][e_type]:
                    avg_historical = np.mean(self.monitor.temporal_patterns[hour][e_type])
                    if avg_historical > 0:
                        # Calculate relative improvement
                        relative_change = (avg_historical - value) / avg_historical
                        pattern_scores.append(relative_change)
            
            if not pattern_scores:
                return 0.0
            
            # Average improvement across emission types
            temporal_score = np.mean(pattern_scores)
            
            # Convert to reward (-1 to 1)
            return np.clip(temporal_score, -1.0, 1.0)
        
        except Exception as e:
            print(f"Error calculating temporal reward: {e}")
            return 0.0
    
    def _calculate_reduction_reward(self, tl_id: str) -> float:
        """Calculate reward component based on emission reductions"""
        try:
            if self.monitor.baseline_emissions is None:
                return 0.0
            
            # Get controlled lanes
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            if not controlled_lanes:
                return 0.0
            
            # Calculate current total emissions for controlled area
            current_total = defaultdict(float)
            for lane in controlled_lanes:
                if not lane.startswith(':'):
                    edge = lane.split('_')[0]
                    if edge in self.monitor.edge_emissions:
                        for e_type, values in self.monitor.edge_emissions[edge].items():
                            if values:
                                current_total[e_type] += values[-1]
            
            # Calculate reduction scores for each emission type
            reduction_scores = []
            for e_type, baseline in self.monitor.baseline_emissions.items():
                if baseline > 0 and e_type in current_total:
                    reduction = (baseline - current_total[e_type]) / baseline
                    weighted_reduction = reduction * self.emission_weights[e_type]
                    reduction_scores.append(weighted_reduction)
            
            if not reduction_scores:
                return 0.0
            
            # Calculate final reduction score
            reduction_score = np.mean(reduction_scores)
            
            # Convert to reward (-1 to 1)
            return np.clip(reduction_score, -1.0, 1.0)
        
        except Exception as e:
            print(f"Error calculating reduction reward: {e}")
            return 0.0
    
    def calculate_reward(self, tl_id: str) -> Tuple[float, Dict[str, float]]:
        """
        Calculate emission-based reward for a traffic light
        
        Args:
            tl_id: Traffic light ID
            
        Returns:
            Tuple of (final_reward, component_rewards)
        """
        try:
            # Update emission monitor
            self.monitor.update_state()
            
            # Calculate reward components
            spatial_reward = self._calculate_spatial_reward(tl_id)
            temporal_reward = self._calculate_temporal_reward(tl_id)
            reduction_reward = self._calculate_reduction_reward(tl_id)
            
            # Store components
            components = {
                'spatial': spatial_reward,
                'temporal': temporal_reward,
                'reduction': reduction_reward
            }
            
            # Calculate weighted sum
            reward = (self.spatial_weight * spatial_reward +
                     self.temporal_weight * temporal_reward +
                     self.reduction_weight * reduction_reward)
            
            # Apply smoothing
            if self.reward_history:
                reward = (self.smoothing_factor * reward +
                         (1 - self.smoothing_factor) * self.reward_history[-1])
            
            # Update history
            self.reward_history.append(reward)
            for component, value in components.items():
                self.component_history[component].append(value)
            
            # Update performance tracking
            self._update_tracking(reward)
            
            # Store additional metrics
            self.violation_history.append(self.monitor.metrics['violation_rate'][-1])
            self.hotspot_history.append(len(self.monitor.emission_hotspots))
            if self.monitor.metrics['emission_reduction']:
                self.reduction_history.append(
                    self.monitor.metrics['emission_reduction'][-1])
            
            return np.clip(reward, -1.0, 1.0), components
        
        except Exception as e:
            print(f"Error calculating emission reward: {e}")
            return 0.0, {}
    
    def _update_tracking(self, reward: float):
        """Update performance tracking metrics"""
        # Update best/worst performance
        if self.best_reward is None or reward > self.best_reward:
            self.best_reward = reward
        if self.worst_reward is None or reward < self.worst_reward:
            self.worst_reward = reward
        
        # Update trend
        self.reward_trend.append(reward)
        if len(self.reward_trend) > 100:  # Keep last 100 steps
            self.reward_trend.pop(0)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.reward_trend:
            return {}
        
        return {
            'best_reward': self.best_reward,
            'worst_reward': self.worst_reward,
            'average_reward': np.mean(self.reward_trend),
            'current_trend': np.polyfit(range(len(self.reward_trend)),
                                      self.reward_trend, 1)[0],
            'average_violation_rate': np.mean(self.violation_history),
            'average_hotspots': np.mean(self.hotspot_history),
            'total_reduction': np.mean(self.reduction_history) if self.reduction_history else 0.0
        }
    
    def save_history(self, filepath: str):
        """Save reward history to file"""
        import json
        
        history = {
            'rewards': self.reward_history,
            'components': dict(self.component_history),
            'violations': self.violation_history,
            'hotspots': self.hotspot_history,
            'reductions': self.reduction_history,
            'performance_stats': self.get_performance_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=4)

def main():
    """Example usage and testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test emission reward calculation')
    parser.add_argument('--net-file', required=True, help='Input network file')
    parser.add_argument('--route-file', required=True, help='Input route file')
    
    args = parser.parse_args()
    
    # Start SUMO
    sumo_cmd = [
        'sumo',
        '-n', args.net_file,
        '-r', args.route_file,
        '--emission-output', 'emission_output.xml',
        '--no-warnings',
    ]
    
    traci.start(sumo_cmd)
    
    # Initialize reward calculator
    reward_calc = EmissionRewardCalculator(args.net_file)
    
    # Run simulation for a few steps
    try:
        print("\nTesting emission reward calculation...")
        for step in range(100):
            # Calculate rewards for all traffic lights
            for tl_id in traci.trafficlight.getIDList():
                reward, components = reward_calc.calculate_reward(tl_id)
                
                if step % 10 == 0:
                    print(f"\nStep {step}, Traffic Light {tl_id}")
                    print(f"Total Reward: {reward:.3f}")
                    print("Reward Components:")
                    for component, value in components.items():
                        print(f"  {component}: {value:.3f}")
            
            traci.simulationStep()
        
        # Save history
        reward_calc.save_history("emission_rewards_history.json")
        
        # Print performance stats
        print("\nPerformance Statistics:")
        for metric, value in reward_calc.get_performance_stats().items():
            print(f"  {metric}: {value:.3f}")
    
    finally:
        traci.close()

if __name__ == "__main__":
    main()