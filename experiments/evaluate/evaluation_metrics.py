# experiments/evaluate/evaluation_metrics.py

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import traci
import json
import yaml
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.environment.multi_agent_sumo_env import MultiAgentSumoEnvironment

class TrafficMetricsEvaluator:
    """Evaluates traffic control performance using multiple metrics"""
    
    def __init__(self, env, output_dir):
        """
        Args:
            env: SUMO environment instance
            output_dir: Directory to save evaluation results
        """
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metric storage
        self.metrics = defaultdict(list)
        self.vehicle_data = defaultdict(dict)
        
    def reset_metrics(self):
        """Reset all metrics for new evaluation"""
        self.metrics = defaultdict(list)
        self.vehicle_data = defaultdict(dict)
    
    def collect_step_metrics(self):
        """Collect metrics for current simulation step"""
        # Get all vehicles currently in the simulation
        vehicle_ids = traci.vehicle.getIDList()
        
        # Collect per-vehicle metrics
        waiting_times = []
        speeds = []
        co2_emissions = []
        
        for vehicle_id in vehicle_ids:
            # Waiting time
            wait_time = traci.vehicle.getWaitingTime(vehicle_id)
            waiting_times.append(wait_time)
            
            # Speed
            speed = traci.vehicle.getSpeed(vehicle_id)
            speeds.append(speed)
            
            # CO2 emissions (in mg/s)
            co2 = traci.vehicle.getCO2Emission(vehicle_id)
            co2_emissions.append(co2)
            
            # Store individual vehicle data
            if vehicle_id not in self.vehicle_data:
                self.vehicle_data[vehicle_id] = {
                    'enter_time': traci.simulation.getTime(),
                    'total_wait_time': 0,
                    'total_co2': 0
                }
            
            # Update vehicle data
            self.vehicle_data[vehicle_id]['total_wait_time'] += wait_time
            self.vehicle_data[vehicle_id]['total_co2'] += co2
        
        # Calculate step metrics
        self.metrics['avg_waiting_time'].append(np.mean(waiting_times) if waiting_times else 0)
        self.metrics['avg_speed'].append(np.mean(speeds) if speeds else 0)
        self.metrics['total_co2'].append(sum(co2_emissions))
        self.metrics['vehicles_in_network'].append(len(vehicle_ids))
        
        # Track throughput (completed trips)
        arrived = traci.simulation.getArrivedNumber()
        self.metrics['arrived_vehicles'].append(arrived)
    
    def calculate_final_metrics(self):
        """Calculate final metrics after simulation"""
        # Calculate overall metrics
        final_metrics = {
            # Average metrics over time
            'mean_waiting_time': np.mean(self.metrics['avg_waiting_time']),
            'mean_speed': np.mean(self.metrics['avg_speed']),
            'total_emissions': sum(self.metrics['total_co2']),
            'total_throughput': sum(self.metrics['arrived_vehicles']),
            
            # Peak metrics
            'peak_waiting_time': max(self.metrics['avg_waiting_time']),
            'peak_vehicles': max(self.metrics['vehicles_in_network']),
            
            # Per-vehicle metrics
            'avg_trip_duration': np.mean([
                traci.simulation.getTime() - data['enter_time']
                for data in self.vehicle_data.values()
                if 'exit_time' in data
            ]),
            'avg_vehicle_total_wait': np.mean([
                data['total_wait_time']
                for data in self.vehicle_data.values()
            ]),
            'avg_vehicle_emissions': np.mean([
                data['total_co2']
                for data in self.vehicle_data.values()
            ])
        }
        
        return final_metrics
    
    def plot_metrics(self, title_prefix=""):
        """Plot evaluation metrics"""
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"{title_prefix} Traffic Control Performance Metrics")
        
        # Plot waiting times
        axes[0, 0].plot(self.metrics['avg_waiting_time'])
        axes[0, 0].set_title('Average Waiting Time')
        axes[0, 0].set_xlabel('Simulation Step')
        axes[0, 0].set_ylabel('Time (s)')
        
        # Plot vehicle speeds
        axes[0, 1].plot(self.metrics['avg_speed'])
        axes[0, 1].set_title('Average Vehicle Speed')
        axes[0, 1].set_xlabel('Simulation Step')
        axes[0, 1].set_ylabel('Speed (m/s)')
        
        # Plot emissions
        axes[1, 0].plot(self.metrics['total_co2'])
        axes[1, 0].set_title('Total CO2 Emissions')
        axes[1, 0].set_xlabel('Simulation Step')
        axes[1, 0].set_ylabel('CO2 (mg)')
        
        # Plot throughput
        cumulative_throughput = np.cumsum(self.metrics['arrived_vehicles'])
        axes[1, 1].plot(cumulative_throughput)
        axes[1, 1].set_title('Cumulative Throughput')
        axes[1, 1].set_xlabel('Simulation Step')
        axes[1, 1].set_ylabel('Number of Vehicles')
        
        plt.tight_layout()
        return fig
    
    def run_evaluation(self, model_path=None, num_episodes=5):
        """
        Run full evaluation of traffic control system
        
        Args:
            model_path: Path to trained model weights (if using RL agent)
            num_episodes: Number of evaluation episodes to run
        """
        all_episode_metrics = []
        
        for episode in range(num_episodes):
            self.reset_metrics()
            states, _ = self.env.reset()
            done = False
            
            while not done:
                # If using RL model, get actions from model
                if model_path:
                    # Load model and get actions
                    actions = {}  # Replace with actual model inference
                else:
                    # Use default SUMO traffic light logic
                    actions = {}
                
                # Take step in environment
                next_states, rewards, done, _, info = self.env.step(actions)
                
                # Collect metrics for this step
                self.collect_step_metrics()
            
            # Calculate final metrics for this episode
            episode_metrics = self.calculate_final_metrics()
            all_episode_metrics.append(episode_metrics)
            
            # Plot and save metrics for this episode
            fig = self.plot_metrics(f"Episode {episode + 1}")
            fig.savefig(self.output_dir / f"metrics_episode_{episode + 1}.png")
            plt.close(fig)
        
        # Calculate and save aggregate metrics across all episodes
        aggregate_metrics = {
            metric: {
                'mean': np.mean([em[metric] for em in all_episode_metrics]),
                'std': np.std([em[metric] for em in all_episode_metrics])
            }
            for metric in all_episode_metrics[0].keys()
        }
        
        # Save metrics to file
        metrics_file = self.output_dir / 'evaluation_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump({
                'aggregate_metrics': aggregate_metrics,
                'episode_metrics': all_episode_metrics
            }, f, indent=4)
        
        return aggregate_metrics
    
    def generate_report(self, metrics, title="Traffic Control Evaluation Report"):
        """Generate evaluation report with metrics and visualizations"""
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"""
{title}
Generated: {report_time}

Performance Metrics (averaged over {len(metrics)} episodes):

1. Traffic Flow Efficiency:
   - Average Waiting Time: {metrics['mean_waiting_time']['mean']:.2f} ± {metrics['mean_waiting_time']['std']:.2f} seconds
   - Average Vehicle Speed: {metrics['mean_speed']['mean']:.2f} ± {metrics['mean_speed']['std']:.2f} m/s
   - Total Throughput: {metrics['total_throughput']['mean']:.2f} ± {metrics['total_throughput']['std']:.2f} vehicles

2. Environmental Impact:
   - Total CO2 Emissions: {metrics['total_emissions']['mean']:.2f} ± {metrics['total_emissions']['std']:.2f} mg

3. Peak Performance:
   - Peak Waiting Time: {metrics['peak_waiting_time']['mean']:.2f} ± {metrics['peak_waiting_time']['std']:.2f} seconds
   - Peak Vehicle Count: {metrics['peak_vehicles']['mean']:.2f} ± {metrics['peak_vehicles']['std']:.2f} vehicles

4. Per-Vehicle Statistics:
   - Average Trip Duration: {metrics['avg_trip_duration']['mean']:.2f} ± {metrics['avg_trip_duration']['std']:.2f} seconds
   - Average Total Wait: {metrics['avg_vehicle_total_wait']['mean']:.2f} ± {metrics['avg_vehicle_total_wait']['std']:.2f} seconds
   - Average Emissions: {metrics['avg_vehicle_emissions']['mean']:.2f} ± {metrics['avg_vehicle_emissions']['std']:.2f} mg

Note: Values are presented as mean ± standard deviation
"""
        
        # Save report
        report_file = self.output_dir / 'evaluation_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        return report

def evaluate_model(env_config, model_path=None, num_episodes=5):
    """
    Evaluate a traffic control model or baseline
    
    Args:
        env_config: Environment configuration
        model_path: Path to model weights (optional)
        num_episodes: Number of evaluation episodes
    """
    # Setup environment
    env = MultiAgentSumoEnvironment(**env_config)
    
    # Create evaluator
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path("experiments/evaluation_results") / timestamp
    evaluator = TrafficMetricsEvaluator(env, output_dir)
    
    try:
        # Run evaluation
        metrics = evaluator.run_evaluation(model_path, num_episodes)
        
        # Generate and save report
        report = evaluator.generate_report(metrics)
        print(report)
        
        return metrics, output_dir
    
    finally:
        env.close()

if __name__ == "__main__":
    # Load environment configuration
    with open("config/env_config.yaml", 'r') as f:
        env_config = yaml.safe_load(f)
    
    # Evaluate with default SUMO traffic light logic (no model)
    metrics, output_dir = evaluate_model(env_config, num_episodes=5)
    print(f"Evaluation results saved to: {output_dir}")