# experiments/evaluate/fixed_time_baseline.py

import os
import sys
from pathlib import Path
import numpy as np
import yaml
import traci
from datetime import datetime
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.environment.multi_agent_sumo_env import MultiAgentSumoEnvironment
from experiments.evaluate.evaluation_metrics import TrafficMetricsEvaluator

class FixedTimeController:
    """
    Fixed-time traffic light controller
    Uses pre-defined phase durations regardless of traffic conditions
    """
    def __init__(self, phases_config):
        """
        Args:
            phases_config: Dictionary with traffic light configurations
                {
                    'cycle_time': total cycle time in seconds,
                    'phases': list of phase durations in seconds,
                    'yellow_time': yellow phase duration in seconds
                }
        """
        self.cycle_time = phases_config.get('cycle_time', 120)
        self.phases = phases_config.get('phases', [30, 30, 30, 30])  # Default equal phases
        self.yellow_time = phases_config.get('yellow_time', 3)
        
        # Validate configuration
        assert sum(self.phases) + (len(self.phases) * self.yellow_time) <= self.cycle_time, \
            "Total phase time plus yellow phases exceeds cycle time"
    
    def get_phase(self, time_in_cycle):
        """
        Determine the current phase based on cycle time
        
        Args:
            time_in_cycle: Current time position in the cycle
            
        Returns:
            phase_index: Current phase index
            is_yellow: Whether current phase is yellow
        """
        current_time = 0
        for i, phase_duration in enumerate(self.phases):
            # Check if in main phase
            if current_time <= time_in_cycle < (current_time + phase_duration):
                return i, False
            
            # Check if in yellow phase
            yellow_start = current_time + phase_duration
            yellow_end = yellow_start + self.yellow_time
            if yellow_start <= time_in_cycle < yellow_end:
                return i, True
            
            current_time = yellow_end
        
        # If we've exceeded cycle time, wrap around
        return self.get_phase(time_in_cycle % self.cycle_time)

class FixedTimeBaseline:
    """
    Implementation of fixed-time baseline for traffic control
    """
    def __init__(self, env_config, phases_config):
        self.env = MultiAgentSumoEnvironment(**env_config)
        self.traffic_lights = self.env.traffic_lights
        self.controllers = {
            tl_id: FixedTimeController(phases_config)
            for tl_id in self.traffic_lights
        }
    
    def run_episode(self, evaluator=None):
        """Run one episode with fixed-time control"""
        states, _ = self.env.reset()
        done = False
        step = 0
        
        while not done:
            # Calculate current phase for each traffic light
            actions = {}
            for tl_id in self.traffic_lights:
                controller = self.controllers[tl_id]
                time_in_cycle = step % controller.cycle_time
                phase, is_yellow = controller.get_phase(time_in_cycle)
                
                if is_yellow:
                    # Use yellow phase index from environment
                    phase = self.env.yellow_phase_dict[tl_id][0]
                
                actions[tl_id] = phase
            
            # Take step in environment
            next_states, rewards, done, _, info = self.env.step(actions)
            
            # Collect metrics if evaluator is provided
            if evaluator is not None:
                evaluator.collect_step_metrics()
            
            states = next_states
            step += 1
        
        return info
    
    def close(self):
        """Clean up environment"""
        self.env.close()

def run_fixed_time_evaluation(env_config, phases_config, num_episodes=5):
    """
    Run evaluation of fixed-time baseline
    
    Args:
        env_config: Environment configuration
        phases_config: Phase timing configuration
        num_episodes: Number of evaluation episodes
    
    Returns:
        metrics: Evaluation metrics
        output_dir: Directory containing evaluation results
    """
    # Setup evaluation directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path("experiments/evaluation_results/fixed_time") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configurations
    with open(output_dir / "env_config.yaml", 'w') as f:
        yaml.dump(env_config, f)
    with open(output_dir / "phases_config.yaml", 'w') as f:
        yaml.dump(phases_config, f)
    
    # Initialize baseline and evaluator
    baseline = FixedTimeBaseline(env_config, phases_config)
    evaluator = TrafficMetricsEvaluator(baseline.env, output_dir)
    
    try:
        # Run evaluation episodes
        for episode in range(num_episodes):
            print(f"Running episode {episode + 1}/{num_episodes}")
            evaluator.reset_metrics()
            
            # Run episode
            info = baseline.run_episode(evaluator)
            
            # Calculate and save episode metrics
            episode_metrics = evaluator.calculate_final_metrics()
            
            # Plot episode metrics
            fig = evaluator.plot_metrics(f"Fixed-Time Baseline - Episode {episode + 1}")
            fig.savefig(output_dir / f"metrics_episode_{episode + 1}.png")
            
            # Save episode metrics
            with open(output_dir / f"metrics_episode_{episode + 1}.json", 'w') as f:
                json.dump(episode_metrics, f, indent=4)
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            metric: {
                'mean': np.mean([m[metric] for m in evaluator.metrics]),
                'std': np.std([m[metric] for m in evaluator.metrics])
            }
            for metric in evaluator.metrics[0].keys()
        }
        
        # Generate and save report
        report = evaluator.generate_report(aggregate_metrics, 
                                         "Fixed-Time Baseline Evaluation Report")
        
        # Save aggregate metrics
        with open(output_dir / "aggregate_metrics.json", 'w') as f:
            json.dump(aggregate_metrics, f, indent=4)
        
        return aggregate_metrics, output_dir
    
    finally:
        baseline.close()

if __name__ == "__main__":
    # Load configurations
    with open("config/env_config.yaml", 'r') as f:
        env_config = yaml.safe_load(f)
    
    # Define fixed-time phase configuration
    phases_config = {
        'cycle_time': 120,  # Total cycle time in seconds
        'phases': [30, 30, 30, 30],  # Duration of each phase
        'yellow_time': 3  # Yellow phase duration
    }
    
    # Run evaluation
    metrics, output_dir = run_fixed_time_evaluation(env_config, phases_config)
    
    print(f"Evaluation completed. Results saved to: {output_dir}")
    print("\nSummary of Results:")
    print(f"Average Waiting Time: {metrics['mean_waiting_time']['mean']:.2f} ± "
          f"{metrics['mean_waiting_time']['std']:.2f} seconds")
    print(f"Total Throughput: {metrics['total_throughput']['mean']:.2f} ± "
          f"{metrics['total_throughput']['std']:.2f} vehicles")
    print(f"Average Speed: {metrics['mean_speed']['mean']:.2f} ± "
          f"{metrics['mean_speed']['std']:.2f} m/s")