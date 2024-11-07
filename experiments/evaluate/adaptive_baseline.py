# experiments/evaluate/adaptive_baseline.py

import os
import sys
from pathlib import Path
import numpy as np
import yaml
import traci
from datetime import datetime
import json
from collections import defaultdict

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.environment.multi_agent_sumo_env import MultiAgentSumoEnvironment
from experiments.evaluate.evaluation_metrics import TrafficMetricsEvaluator

class AdaptiveController:
    """
    Adaptive traffic light controller
    Adjusts phase durations based on real-time traffic conditions
    """
    def __init__(self, config):
        """
        Args:
            config: Dictionary with controller configurations
                {
                    'min_phase_time': minimum phase duration,
                    'max_phase_time': maximum phase duration,
                    'yellow_time': yellow phase duration,
                    'vehicle_extension_time': time to extend green on vehicle detection,
                    'max_gap': maximum time gap to extend green,
                    'queue_threshold': vehicle count threshold for queue detection
                }
        """
        self.min_phase_time = config.get('min_phase_time', 10)
        self.max_phase_time = config.get('max_phase_time', 60)
        self.yellow_time = config.get('yellow_time', 3)
        self.vehicle_extension_time = config.get('vehicle_extension_time', 2)
        self.max_gap = config.get('max_gap', 3.0)
        self.queue_threshold = config.get('queue_threshold', 4)
        
        # Controller state
        self.current_phase = 0
        self.phase_time = 0
        self.yellow_active = False
        self.last_phase_change = 0
    
    def get_queue_lengths(self, intersection_id):
        """Get queue lengths for all approaches"""
        controlled_lanes = traci.trafficlight.getControlledLanes(intersection_id)
        queue_lengths = {}
        
        for lane in controlled_lanes:
            # Get number of halting vehicles
            queue = traci.lane.getLastStepHaltingNumber(lane)
            queue_lengths[lane] = queue
        
        return queue_lengths
    
    def get_approach_demands(self, intersection_id):
        """Calculate traffic demand for each approach"""
        controlled_lanes = traci.trafficlight.getControlledLanes(intersection_id)
        demands = {}
        
        for lane in controlled_lanes:
            # Consider both queued and approaching vehicles
            queue = traci.lane.getLastStepHaltingNumber(lane)
            vehicles = traci.lane.getLastStepVehicleNumber(lane)
            mean_speed = traci.lane.getLastStepMeanSpeed(lane)
            
            # Calculate demand based on queue and approaching vehicles
            demand = queue + (vehicles - queue) * (1 - min(1, mean_speed / 13.89))  # 13.89 m/s = 50 km/h
            demands[lane] = demand
        
        return demands
    
    def should_extend_green(self, intersection_id, current_lanes):
        """Determine if green phase should be extended"""
        # Check if any approaching vehicles in current green lanes
        for lane in current_lanes:
            vehicles = traci.lane.getLastStepVehicleNumber(lane)
            if vehicles > 0:
                mean_speed = traci.lane.getLastStepMeanSpeed(lane)
                time_to_stop = traci.lane.getLength(lane) / max(mean_speed, 1e-6)
                
                # Extend if vehicles are approaching and within reasonable time
                if time_to_stop < self.max_gap:
                    return True
        
        return False
    
    def get_competing_demand(self, intersection_id, phase_index):
        """Calculate demand for competing movements"""
        phase_def = traci.trafficlight.getAllProgramLogics(intersection_id)[0].phases[phase_index]
        phase_state = phase_def.state
        controlled_lanes = traci.trafficlight.getControlledLanes(intersection_id)
        
        total_demand = 0
        for i, state in enumerate(phase_state):
            if state == 'r':  # red signal
                if i < len(controlled_lanes):
                    lane = controlled_lanes[i]
                    queue = traci.lane.getLastStepHaltingNumber(lane)
                    vehicles = traci.lane.getLastStepVehicleNumber(lane)
                    total_demand += queue + (vehicles - queue) * 0.5  # Weight approaching vehicles less
        
        return total_demand
    
    def decide_next_phase(self, intersection_id, current_time):
        """Decide whether to change phase and what the next phase should be"""
        # Get current phase definition
        current_phase_def = traci.trafficlight.getAllProgramLogics(intersection_id)[0].phases[self.current_phase]
        current_lanes = [lane for i, lane in 
                        enumerate(traci.trafficlight.getControlledLanes(intersection_id))
                        if current_phase_def.state[i] == 'G']
        
        # Check if minimum green time has passed
        if self.phase_time < self.min_phase_time:
            return self.current_phase, False
        
        # Get current demands
        current_demand = sum(self.get_queue_lengths(intersection_id).values())
        
        # Check if should extend current green
        if (self.phase_time < self.max_phase_time and 
            self.should_extend_green(intersection_id, current_lanes)):
            return self.current_phase, False
        
        # Evaluate competing demands for all phases
        phase_demands = []
        for i in range(len(traci.trafficlight.getAllProgramLogics(intersection_id)[0].phases)):
            if i % 2 == 0:  # Skip yellow phases
                demand = self.get_competing_demand(intersection_id, i)
                phase_demands.append((i, demand))
        
        # Select phase with highest demand
        next_phase, max_demand = max(phase_demands, key=lambda x: x[1])
        
        # Change phase if competing demand is significant
        if max_demand > current_demand or self.phase_time >= self.max_phase_time:
            return next_phase, True
        
        return self.current_phase, False
    
    def get_action(self, intersection_id, current_time):
        """Get action for current time step"""
        if self.yellow_active:
            # Check if yellow phase is complete
            if current_time - self.last_phase_change >= self.yellow_time:
                self.yellow_active = False
                self.current_phase = (self.current_phase + 1) % len(traci.trafficlight.getAllProgramLogics(intersection_id)[0].phases)
                self.phase_time = 0
                self.last_phase_change = current_time
            return self.current_phase
        
        # Update phase time
        self.phase_time = current_time - self.last_phase_change
        
        # Decide if phase should change
        next_phase, should_change = self.decide_next_phase(intersection_id, current_time)
        
        if should_change:
            self.yellow_active = True
            self.last_phase_change = current_time
            return self.current_phase  # Will be yellow phase
        
        return self.current_phase

class AdaptiveBaseline:
    """
    Implementation of adaptive baseline for traffic control
    """
    def __init__(self, env_config, controller_config):
        self.env = MultiAgentSumoEnvironment(**env_config)
        self.traffic_lights = self.env.traffic_lights
        
        # Create adaptive controller for each intersection
        self.controllers = {
            tl_id: AdaptiveController(controller_config)
            for tl_id in self.traffic_lights
        }
    
    def run_episode(self, evaluator=None):
        """Run one episode with adaptive control"""
        states, _ = self.env.reset()
        done = False
        current_time = 0
        
        while not done:
            # Get actions from adaptive controllers
            actions = {}
            for tl_id in self.traffic_lights:
                action = self.controllers[tl_id].get_action(tl_id, current_time)
                actions[tl_id] = action
            
            # Take step in environment
            next_states, rewards, done, _, info = self.env.step(actions)
            
            # Collect metrics if evaluator is provided
            if evaluator is not None:
                evaluator.collect_step_metrics()
            
            states = next_states
            current_time += 1
        
        return info
    
    def close(self):
        """Clean up environment"""
        self.env.close()

def run_adaptive_evaluation(env_config, controller_config, num_episodes=5):
    """
    Run evaluation of adaptive baseline
    
    Args:
        env_config: Environment configuration
        controller_config: Adaptive controller configuration
        num_episodes: Number of evaluation episodes
    
    Returns:
        metrics: Evaluation metrics
        output_dir: Directory containing evaluation results
    """
    # Setup evaluation directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path("experiments/evaluation_results/adaptive") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configurations
    with open(output_dir / "env_config.yaml", 'w') as f:
        yaml.dump(env_config, f)
    with open(output_dir / "controller_config.yaml", 'w') as f:
        yaml.dump(controller_config, f)
    
    # Initialize baseline and evaluator
    baseline = AdaptiveBaseline(env_config, controller_config)
    evaluator = TrafficMetricsEvaluator(baseline.env, output_dir)
    
    try:
        # Run evaluation episodes
        all_metrics = []
        for episode in range(num_episodes):
            print(f"Running episode {episode + 1}/{num_episodes}")
            evaluator.reset_metrics()
            
            # Run episode
            info = baseline.run_episode(evaluator)
            
            # Calculate and save episode metrics
            episode_metrics = evaluator.calculate_final_metrics()
            all_metrics.append(episode_metrics)
            
            # Plot episode metrics
            fig = evaluator.plot_metrics(f"Adaptive Baseline - Episode {episode + 1}")
            fig.savefig(output_dir / f"metrics_episode_{episode + 1}.png")
            
            # Save episode metrics
            with open(output_dir / f"metrics_episode_{episode + 1}.json", 'w') as f:
                json.dump(episode_metrics, f, indent=4)
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            metric: {
                'mean': np.mean([m[metric] for m in all_metrics]),
                'std': np.std([m[metric] for m in all_metrics])
            }
            for metric in all_metrics[0].keys()
        }
        
        # Generate and save report
        report = evaluator.generate_report(aggregate_metrics, 
                                         "Adaptive Baseline Evaluation Report")
        
        # Save aggregate metrics
        with open(output_dir / "aggregate_metrics.json", 'w') as f:
            json.dump(aggregate_metrics, f, indent=4)
        
        return aggregate_metrics, output_dir
    
    finally:
        baseline.close()

if __name__ == "__main__":
    # Load environment configuration
    with open("config/env_config.yaml", 'r') as f:
        env_config = yaml.safe_load(f)
    
    # Define adaptive controller configuration
    controller_config = {
        'min_phase_time': 10,
        'max_phase_time': 60,
        'yellow_time': 3,
        'vehicle_extension_time': 2,
        'max_gap': 3.0,
        'queue_threshold': 4
    }
    
    # Run evaluation
    metrics, output_dir = run_adaptive_evaluation(env_config, controller_config)
    
    print(f"Evaluation completed. Results saved to: {output_dir}")
    print("\nSummary of Results:")
    print(f"Average Waiting Time: {metrics['mean_waiting_time']['mean']:.2f} ± "
          f"{metrics['mean_waiting_time']['std']:.2f} seconds")
    print(f"Total Throughput: {metrics['total_throughput']['mean']:.2f} ± "
          f"{metrics['total_throughput']['std']:.2f} vehicles")
    print(f"Average Speed: {metrics['mean_speed']['mean']:.2f} ± "
          f"{metrics['mean_speed']['std']:.2f} m/s")