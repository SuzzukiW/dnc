# experiments/scenarios/rewards/hybrid_rewards.py

import os
import sys
import numpy as np
import traci
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import networkx as nx

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Part A

class NetworkAnalyzer:
    """Analyze traffic network structure and relationships"""
    
    def __init__(self, net_file: str):
        """
        Initialize network analyzer
        
        Args:
            net_file: Path to SUMO network file
        """
        self.net_file = net_file
        self.graph = self._build_network_graph()
        self.centrality_scores = self._calculate_centrality()
        self.influence_zones = self._calculate_influence_zones()
    
    def _build_network_graph(self) -> nx.Graph:
        """Build graph representation of the traffic network"""
        graph = nx.Graph()
        
        # Read network
        net = sumolib.net.readNet(self.net_file)
        
        # Add nodes (junctions)
        for junction in net.getNodes():
            graph.add_node(junction.getID(), 
                         pos=(junction.getCoord()))
        
        # Add edges
        for edge in net.getEdges():
            from_node = edge.getFromNode().getID()
            to_node = edge.getToNode().getID()
            length = edge.getLength()
            graph.add_edge(from_node, to_node, 
                         weight=length,
                         edge_id=edge.getID())
        
        return graph
    
    def _calculate_centrality(self) -> Dict[str, float]:
        """Calculate centrality measures for nodes"""
        # Calculate different centrality measures
        degree_cent = nx.degree_centrality(self.graph)
        between_cent = nx.betweenness_centrality(self.graph)
        close_cent = nx.closeness_centrality(self.graph)
        
        # Combine measures with weights
        centrality = {}
        for node in self.graph.nodes():
            centrality[node] = (
                0.4 * degree_cent[node] +      # Connectivity importance
                0.4 * between_cent[node] +     # Flow control importance
                0.2 * close_cent[node]         # Accessibility importance
            )
        
        return centrality
    
    def _calculate_influence_zones(self, max_distance: float = 200.0) -> Dict[str, List[str]]:
        """Calculate influence zones for traffic lights"""
        influence_zones = defaultdict(list)
        
        # Get all traffic lights
        tls = traci.trafficlight.getIDList()
        
        for tl1 in tls:
            # Get junction for this traffic light
            try:
                j1 = next(j for j in self.graph.nodes() 
                         if any(tl1 in edge for edge in 
                               self.graph.edges(j, data=True)))
            except StopIteration:
                continue
            
            for tl2 in tls:
                if tl1 != tl2:
                    try:
                        # Get junction for other traffic light
                        j2 = next(j for j in self.graph.nodes() 
                                if any(tl2 in edge for edge in 
                                      self.graph.edges(j, data=True)))
                        
                        # Calculate shortest path distance
                        try:
                            distance = nx.shortest_path_length(
                                self.graph, j1, j2, weight='weight')
                            
                            if distance <= max_distance:
                                influence_zones[tl1].append((tl2, distance))
                        except nx.NetworkXNoPath:
                            continue
                    except StopIteration:
                        continue
        
        return influence_zones
    
    def get_normalized_centrality(self, tl_id: str) -> float:
        """Get normalized centrality score for a traffic light"""
        try:
            junction = next(j for j in self.graph.nodes() 
                          if any(tl_id in edge for edge in 
                                self.graph.edges(j, data=True)))
            return self.centrality_scores[junction]
        except StopIteration:
            return 0.0
    
    def get_influenced_traffic_lights(self, tl_id: str) -> List[Tuple[str, float]]:
        """Get list of traffic lights influenced by given traffic light"""
        return self.influence_zones.get(tl_id, [])

class GlobalStateTracker:
    """Track global traffic state and performance metrics"""
    
    def __init__(self):
        """Initialize global state tracker"""
        self.metrics = defaultdict(list)
        self.current_state = {}
    
    def update_state(self):
        """Update global traffic state"""
        try:
            # Get overall network metrics
            vehicles = traci.vehicle.getIDList()
            
            # Calculate network-wide metrics
            total_waiting_time = sum(traci.vehicle.getAccumulatedWaitingTime(v) 
                                   for v in vehicles)
            total_speed = sum(traci.vehicle.getSpeed(v) for v in vehicles)
            avg_speed = total_speed / len(vehicles) if vehicles else 0
            
            stopped_vehicles = sum(1 for v in vehicles 
                                 if traci.vehicle.getSpeed(v) < 0.1)
            congestion_rate = stopped_vehicles / len(vehicles) if vehicles else 0
            
            # Calculate emissions (CO2 in mg)
            total_co2 = sum(traci.vehicle.getCO2Emission(v) for v in vehicles)
            
            # Update current state
            self.current_state = {
                'num_vehicles': len(vehicles),
                'total_waiting_time': total_waiting_time,
                'average_speed': avg_speed,
                'congestion_rate': congestion_rate,
                'total_co2': total_co2
            }
            
            # Store metrics history
            for key, value in self.current_state.items():
                self.metrics[key].append(value)
        
        except traci.exceptions.TraCIException as e:
            print(f"Error updating global state: {e}")
    
    def get_current_state(self) -> Dict[str, float]:
        """Get current global state"""
        return self.current_state
    
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Get historical metrics"""
        return dict(self.metrics)
    
    def get_performance_score(self) -> float:
        """Calculate overall network performance score"""
        if not self.current_state:
            return 0.0
        
        # Calculate score components
        waiting_score = np.exp(-self.current_state['total_waiting_time'] / 10000)
        speed_score = self.current_state['average_speed'] / 13.89  # Normalize by 50 km/h
        congestion_score = 1 - self.current_state['congestion_rate']
        emission_score = np.exp(-self.current_state['total_co2'] / 1000000)
        
        # Combine scores with weights
        score = (0.35 * waiting_score +
                0.25 * speed_score +
                0.25 * congestion_score +
                0.15 * emission_score)
        
        return np.clip(score, 0, 1)

# Part B

class HybridRewardCalculator:
    """Calculate hybrid rewards combining local and global components"""
    
    def __init__(self,
                net_file: str,
                local_weight: float = 0.6,
                global_weight: float = 0.4,
                neighbor_weight: float = 0.3,
                max_waiting_time: float = 180.0,
                max_queue_length: int = 20,
                coordinated_phases: bool = True):
        """
        Initialize hybrid reward calculator
        
        Args:
            net_file: Path to SUMO network file
            local_weight: Weight for local reward component
            global_weight: Weight for global reward component
            neighbor_weight: Weight for neighbor influence
            max_waiting_time: Maximum waiting time threshold
            max_queue_length: Maximum queue length threshold
            coordinated_phases: Whether to consider phase coordination
        """
        self.local_weight = local_weight
        self.global_weight = global_weight
        self.neighbor_weight = neighbor_weight
        self.max_waiting_time = max_waiting_time
        self.max_queue_length = max_queue_length
        self.coordinated_phases = coordinated_phases
        
        # Initialize network analyzer and global tracker
        self.network = NetworkAnalyzer(net_file)
        self.global_tracker = GlobalStateTracker()
        
        # Store previous states for calculating improvements
        self.previous_states = defaultdict(dict)
        
        # Phase coordination tracking
        self.phase_timings = defaultdict(dict)
        
        # Performance tracking
        self.performance_history = defaultdict(list)
    
    def _calculate_local_reward(self, 
                              tl_id: str, 
                              controlled_lanes: List[str]) -> float:
        """Calculate local reward component"""
        # Get basic metrics
        total_waiting_time = sum(traci.lane.getWaitingTime(lane) 
                               for lane in controlled_lanes)
        total_queue = sum(traci.lane.getLastStepHaltingNumber(lane) 
                         for lane in controlled_lanes)
        
        # Calculate utilization
        total_vehicles = sum(traci.lane.getLastStepVehicleNumber(lane) 
                           for lane in controlled_lanes)
        capacity = len(controlled_lanes) * self.max_queue_length
        utilization = total_vehicles / capacity if capacity > 0 else 0
        
        # Calculate speed efficiency
        avg_speed = np.mean([traci.lane.getLastStepMeanSpeed(lane) / 
                           max(traci.lane.getMaxSpeed(lane), 1.0)
                           for lane in controlled_lanes])
        
        # Normalize metrics
        waiting_score = -total_waiting_time / (len(controlled_lanes) * self.max_waiting_time)
        queue_score = -total_queue / (len(controlled_lanes) * self.max_queue_length)
        util_score = np.exp(-abs(utilization - 0.6))  # Optimal utilization around 60%
        
        # Weight components
        local_reward = (0.4 * waiting_score +
                       0.3 * queue_score +
                       0.2 * avg_speed +
                       0.1 * util_score)
        
        return np.clip(local_reward, -1.0, 1.0)
    
    def _calculate_neighbor_influence(self, 
                                   tl_id: str, 
                                   local_reward: float) -> float:
        """Calculate influence from neighboring traffic lights"""
        influenced_tls = self.network.get_influenced_traffic_lights(tl_id)
        if not influenced_tls:
            return 0.0
        
        total_influence = 0.0
        total_weight = 0.0
        
        for neighbor_id, distance in influenced_tls:
            try:
                # Get neighbor's controlled lanes
                neighbor_lanes = traci.trafficlight.getControlledLanes(neighbor_id)
                
                # Calculate neighbor's local reward
                neighbor_reward = self._calculate_local_reward(neighbor_id, neighbor_lanes)
                
                # Weight by distance (closer neighbors have more influence)
                weight = 1.0 / max(distance, 1.0)
                total_influence += neighbor_reward * weight
                total_weight += weight
            except:
                continue
        
        if total_weight > 0:
            return total_influence / total_weight
        return 0.0
    
    def _calculate_phase_coordination(self, tl_id: str) -> float:
        """Calculate reward component for phase coordination"""
        if not self.coordinated_phases:
            return 0.0
        
        try:
            # Get current phase info
            current_phase = traci.trafficlight.getPhase(tl_id)
            phase_state = traci.trafficlight.getRedYellowGreenState(tl_id)
            
            # Store phase timing
            current_time = traci.simulation.getTime()
            if tl_id not in self.phase_timings:
                self.phase_timings[tl_id] = {'last_phase': current_phase,
                                           'last_change': current_time}
            
            # Check phase progression
            coord_score = 0.0
            if current_phase != self.phase_timings[tl_id]['last_phase']:
                # Calculate phase duration
                duration = current_time - self.phase_timings[tl_id]['last_change']
                
                # Reward for maintaining green waves (simplified)
                influenced_tls = self.network.get_influenced_traffic_lights(tl_id)
                for neighbor_id, distance in influenced_tls:
                    try:
                        neighbor_phase = traci.trafficlight.getPhase(neighbor_id)
                        neighbor_state = traci.trafficlight.getRedYellowGreenState(neighbor_id)
                        
                        # Check if phases are coordinated
                        if (('g' in phase_state.lower() and 'g' in neighbor_state.lower()) or
                            ('r' in phase_state.lower() and 'r' in neighbor_state.lower())):
                            coord_score += 0.1
                    except:
                        continue
                
                # Update timing info
                self.phase_timings[tl_id].update({
                    'last_phase': current_phase,
                    'last_change': current_time
                })
            
            return min(coord_score, 1.0)
            
        except traci.exceptions.TraCIException:
            return 0.0
    
    def calculate_reward(self, tl_id: str) -> float:
        """
        Calculate hybrid reward for a traffic light
        
        Args:
            tl_id: Traffic light ID
            
        Returns:
            float: Combined reward value
        """
        try:
            # Update global state
            self.global_tracker.update_state()
            
            # Get controlled lanes
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            if not controlled_lanes:
                return 0.0
            
            # Calculate components
            local_reward = self._calculate_local_reward(tl_id, controlled_lanes)
            global_score = self.global_tracker.get_performance_score()
            neighbor_influence = self._calculate_neighbor_influence(tl_id, local_reward)
            coordination_score = self._calculate_phase_coordination(tl_id)
            
            # Get intersection importance
            centrality = self.network.get_normalized_centrality(tl_id)
            
            # Calculate improvement bonuses
            improvement_bonus = 0.0
            if tl_id in self.previous_states:
                prev_reward = self.previous_states[tl_id].get('reward', -float('inf'))
                if local_reward > prev_reward:
                    improvement_bonus = 0.1
            
            # Store current state
            self.previous_states[tl_id].update({
                'reward': local_reward,
                'time': traci.simulation.getTime()
            })
            
            # Combine components with weights
            hybrid_reward = (
                self.local_weight * local_reward +
                self.global_weight * global_score +
                self.neighbor_weight * neighbor_influence +
                0.1 * coordination_score +
                improvement_bonus
            )
            
            # Scale by intersection importance
            hybrid_reward *= (0.8 + 0.2 * centrality)
            
            # Track performance
            self.performance_history[tl_id].append({
                'time': traci.simulation.getTime(),
                'local_reward': local_reward,
                'global_score': global_score,
                'neighbor_influence': neighbor_influence,
                'coordination_score': coordination_score,
                'final_reward': hybrid_reward
            })
            
            return np.clip(hybrid_reward, -1.0, 1.0)
        
        except Exception as e:
            print(f"Error calculating reward for {tl_id}: {e}")
            return 0.0
    
    def get_reward_components(self, tl_id: str) -> Dict[str, float]:
        """Get detailed breakdown of reward components"""
        if not self.performance_history[tl_id]:
            return {}
        
        return self.performance_history[tl_id][-1]
    
    def save_performance_history(self, filepath: str):
        """Save performance history to file"""
        import json
        
        history = {tl_id: [dict(step) for step in tl_history]
                  for tl_id, tl_history in self.performance_history.items()}
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=4)

def main():
    """Example usage and testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test hybrid reward calculation')
    parser.add_argument('--net-file', required=True, help='Input network file')
    parser.add_argument('--route-file', required=True, help='Input route file')
    parser.add_argument('--local-weight', type=float, default=0.6,
                      help='Weight for local reward component')
    parser.add_argument('--global-weight', type=float, default=0.4,
                      help='Weight for global reward component')
    
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
    reward_calc = HybridRewardCalculator(
        args.net_file,
        local_weight=args.local_weight,
        global_weight=args.global_weight
    )
    
    # Run simulation for a few steps
    try:
        print("\nTesting hybrid reward calculation...")
        for step in range(100):
            # Calculate rewards for all traffic lights
            for tl_id in traci.trafficlight.getIDList():
                reward = reward_calc.calculate_reward(tl_id)
                components = reward_calc.get_reward_components(tl_id)
                
                if step % 10 == 0:
                    print(f"\nStep {step}, Traffic Light {tl_id}")
                    print(f"Total Reward: {reward:.3f}")
                    print("Reward Components:")
                    for component, value in components.items():
                        if isinstance(value, float):
                            print(f"  {component}: {value:.3f}")
            
            traci.simulationStep()
        
        # Save performance history
        reward_calc.save_performance_history("hybrid_rewards_history.json")
    
    finally:
        traci.close()

if __name__ == "__main__":
    main()