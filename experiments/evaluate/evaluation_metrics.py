# experiments/evaluate/evaluation_metrics.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import traci
from collections import defaultdict
import json

class TrafficMetricsEvaluator:
    """Evaluator for traffic control systems"""
    
    def __init__(self, env, output_dir):
        """
        Initialize evaluator
        
        Args:
            env: SUMO environment
            output_dir: Directory for saving results
        """
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.reset_metrics()
        
    def reset_metrics(self):
        """Reset all metrics for new episode"""
        self.metrics = {
            'waiting_times': [],
            'queue_lengths': [],
            'speeds': [],
            'throughput': [],
            'travel_times': [],
            'emissions': [],
            'traffic_density': [],
            'stops_per_vehicle': []
        }
        
        # Per intersection metrics
        self.intersection_metrics = defaultdict(lambda: {
            'waiting_times': [],
            'queue_lengths': [],
            'throughput': []
        })
    
    def collect_step_metrics(self):
        """Collect metrics for current simulation step"""
        # Global metrics
        total_waiting_time = 0
        total_queue_length = 0
        total_vehicles = 0
        total_speed = 0
        total_co2 = 0
        
        # Collect metrics for each intersection
        for tl_id in self.env.traffic_lights:
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            intersection_waiting = 0
            intersection_queue = 0
            intersection_vehicles = 0
            
            for lane in controlled_lanes:
                # Basic metrics
                waiting_time = traci.lane.getWaitingTime(lane)
                queue_length = traci.lane.getLastStepHaltingNumber(lane)
                vehicles = traci.lane.getLastStepVehicleNumber(lane)
                speed = traci.lane.getLastStepMeanSpeed(lane)
                
                # Accumulate for intersection
                intersection_waiting += waiting_time
                intersection_queue += queue_length
                intersection_vehicles += vehicles
                
                # Accumulate for global
                total_waiting_time += waiting_time
                total_queue_length += queue_length
                total_vehicles += vehicles
                total_speed += speed * vehicles  # Weight by number of vehicles
                
                # Collect emissions
                for vehicle_id in traci.lane.getLastStepVehicleIDs(lane):
                    total_co2 += traci.vehicle.getCO2Emission(vehicle_id)
            
            # Store intersection metrics
            self.intersection_metrics[tl_id]['waiting_times'].append(intersection_waiting)
            self.intersection_metrics[tl_id]['queue_lengths'].append(intersection_queue)
            self.intersection_metrics[tl_id]['throughput'].append(intersection_vehicles)
        
        # Calculate and store global metrics
        if total_vehicles > 0:
            avg_speed = total_speed / total_vehicles
        else:
            avg_speed = 0
            
        self.metrics['waiting_times'].append(total_waiting_time)
        self.metrics['queue_lengths'].append(total_queue_length)
        self.metrics['speeds'].append(avg_speed)
        self.metrics['throughput'].append(total_vehicles)
        self.metrics['emissions'].append(total_co2)
        
        # Calculate traffic density
        network_length = sum(traci.lane.getLength(lane) 
                           for tl_id in self.env.traffic_lights
                           for lane in traci.trafficlight.getControlledLanes(tl_id))
        density = total_vehicles / (network_length / 1000)  # vehicles per km
        self.metrics['traffic_density'].append(density)
        
        # Calculate stops per vehicle
        total_stops = sum(traci.vehicle.getStopState(vid) > 0 
                         for vid in traci.vehicle.getIDList())
        if total_vehicles > 0:
            stops_per_vehicle = total_stops / total_vehicles
        else:
            stops_per_vehicle = 0
        self.metrics['stops_per_vehicle'].append(stops_per_vehicle)
    
    def calculate_final_metrics(self):
        """Calculate final metrics for episode"""
        final_metrics = {
            'mean_waiting_time': np.mean(self.metrics['waiting_times']),
            'mean_queue_length': np.mean(self.metrics['queue_lengths']),
            'mean_speed': np.mean(self.metrics['speeds']),
            'total_throughput': np.sum(self.metrics['throughput']),
            'mean_density': np.mean(self.metrics['traffic_density']),
            'total_emissions': np.sum(self.metrics['emissions']),
            'mean_stops': np.mean(self.metrics['stops_per_vehicle']),
            'intersection_metrics': {}
        }
        
        # Calculate per-intersection metrics
        for tl_id, metrics in self.intersection_metrics.items():
            final_metrics['intersection_metrics'][tl_id] = {
                'mean_waiting_time': np.mean(metrics['waiting_times']),
                'mean_queue_length': np.mean(metrics['queue_lengths']),
                'total_throughput': np.sum(metrics['throughput'])
            }
        
        return final_metrics
    
    def plot_metrics(self, title="Traffic Metrics"):
        """Plot evaluation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title)
        
        # Plot waiting times
        axes[0, 0].plot(self.metrics['waiting_times'])
        axes[0, 0].set_title('Total Waiting Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Time (s)')
        
        # Plot queue lengths
        axes[0, 1].plot(self.metrics['queue_lengths'])
        axes[0, 1].set_title('Total Queue Length')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Vehicles')
        
        # Plot speeds
        axes[1, 0].plot(self.metrics['speeds'])
        axes[1, 0].set_title('Average Speed')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Speed (m/s)')
        
        # Plot throughput
        axes[1, 1].plot(self.metrics['throughput'])
        axes[1, 1].set_title('Total Throughput')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Vehicles')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, metrics, title="Traffic Control Evaluation Report"):
        """Generate evaluation report"""
        report = {
            'title': title,
            'metrics': metrics,
            'summary_statistics': {
                'waiting_time_improvement': None,
                'throughput_improvement': None,
                'overall_performance': None
            }
        }
        
        # Calculate improvements if baseline metrics are available
        if hasattr(self, 'baseline_metrics'):
            report['summary_statistics']['waiting_time_improvement'] = (
                (self.baseline_metrics['mean_waiting_time'] - metrics['mean_waiting_time']['mean']) /
                self.baseline_metrics['mean_waiting_time'] * 100
            )
            report['summary_statistics']['throughput_improvement'] = (
                (metrics['total_throughput']['mean'] - self.baseline_metrics['total_throughput']) /
                self.baseline_metrics['total_throughput'] * 100
            )
        
        # Save report
        with open(self.output_dir / 'evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        return report