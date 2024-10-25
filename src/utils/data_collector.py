# src/utils/data_collector.py

import os
import traci
import numpy as np
import pandas as pd
from typing import Dict, List

class SUMODataCollector:
    """Minimal SUMO data collector using only basic commands"""
    
    def __init__(self):
        self.collected_data = {
            'traffic_metrics': []
        }

    def setup_collection(self):
        """Initialize data collection"""
        pass  # No setup needed for minimal version

    def collect_network_metrics(self) -> Dict:
        """Collect basic network metrics"""
        # Get list of vehicles
        vehicles = traci.vehicle.getIDList()
        
        # Initialize metrics
        metrics = {
            'timestamp': traci.simulation.getTime(),
            'vehicle_count': len(vehicles),
            'total_speed': 0.0,
            'total_waiting': 0.0
        }
        
        # If there are vehicles, collect their metrics
        if vehicles:
            for vehicle_id in vehicles:
                metrics['total_speed'] += traci.vehicle.getSpeed(vehicle_id)
                metrics['total_waiting'] += traci.vehicle.getWaitingTime(vehicle_id)
            
            # Calculate averages
            metrics['average_speed'] = metrics['total_speed'] / len(vehicles)
        else:
            metrics['average_speed'] = 0.0
            
        return metrics

    def step_collection(self):
        """Collect data for current simulation step"""
        network_metrics = self.collect_network_metrics()
        self.collected_data['traffic_metrics'].append(network_metrics)

    def export_data(self, output_dir: str):
        """Export collected data to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.collected_data['traffic_metrics']:
            metrics_df = pd.DataFrame(self.collected_data['traffic_metrics'])
            metrics_df.to_csv(os.path.join(output_dir, 'traffic_metrics.csv'), 
                            index=False)
            print(f"Exported {len(metrics_df)} records to {output_dir}")