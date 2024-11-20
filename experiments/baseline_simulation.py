import os
import sys
import numpy as np
import time
from collections import deque

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Set SUMO_HOME environment variable
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = '/opt/homebrew/opt/sumo/share/sumo'
    os.environ['SUMO_HOME_SET'] = 'true'  # Prevents retry messages

import traci
from sumolib import checkBinary
import pandas as pd

# Global hyperparameters that should be consistent across all models
HYPERPARAMETERS = {
    'simulation': {
        'total_episodes': 10,       # Number of episodes to run
        'episode_length': 3600,     # Each episode is 1 hour (in seconds)
        'step_length': 1.0,         # Each step is 1 second
        'yellow_duration': 3,       # Duration of yellow light (seconds)
        'min_green_duration': 5,    # Minimum green light duration (seconds)
        'max_green_duration': 60,   # Maximum green light duration (seconds)
        'waiting_time_memory': 1800 # How long to track waiting times (seconds)
    },
    'paths': {
        'net_file': "Version1/2024-11-05-18-42-37/osm.net.xml.gz",
        'route_file': "Version1/2024-11-05-18-42-37/osm.passenger.trips.xml"
    }
}

def run_baseline_simulation(config):
    """
    Run a baseline simulation without any traffic light optimization.
    This uses SUMO's default traffic light timing.
    """
    print("Starting baseline simulation...")
    
    # Configuration for the simulation
    net_file = os.path.join(project_root, config['paths']['net_file'])
    route_file = os.path.join(project_root, config['paths']['route_file'])
    
    # Simulation settings from config
    sim_duration = config['simulation']['episode_length']
    total_episodes = config['simulation']['total_episodes']
    step_length = config['simulation']['step_length']
    
    # Lists to store metrics for all episodes
    all_episodes_metrics = []
    
    for episode in range(total_episodes):
        print(f"\nStarting Episode {episode + 1}/{total_episodes}")
        
        # Set up the SUMO command with no GUI
        sumo_binary = checkBinary('sumo')
        sumo_cmd = [
            sumo_binary,
            '-n', net_file,
            '-r', route_file,
            '--step-length', str(step_length),
            '--waiting-time-memory', str(config['simulation']['waiting_time_memory']),  # Track waiting times
            '--no-warnings',
            '--no-step-log',  # Reduce console output
            '--random',  # Add randomness between episodes
        ]
        
        # Start SUMO
        traci.start(sumo_cmd)
        
        # Metrics to track for this episode
        step = 0
        episode_metrics = {
            'episode': episode + 1,
            'step': [],
            'vehicles_in_system': [],
            'avg_waiting_time': [],
            'avg_speed': []
        }
        
        print("\nSimulation progress:")
        try:
            while step < sim_duration:
                traci.simulationStep()
                
                # Get all vehicles currently in the simulation
                vehicles = traci.vehicle.getIDList()
                num_vehicles = len(vehicles)
                
                if num_vehicles > 0:
                    # Calculate metrics
                    waiting_times = [traci.vehicle.getWaitingTime(v) for v in vehicles]
                    speeds = [traci.vehicle.getSpeed(v) for v in vehicles]
                    
                    avg_waiting_time = sum(waiting_times) / num_vehicles
                    avg_speed = sum(speeds) / num_vehicles
                    
                    # Store metrics
                    episode_metrics['step'].append(step)
                    episode_metrics['vehicles_in_system'].append(num_vehicles)
                    episode_metrics['avg_waiting_time'].append(avg_waiting_time)
                    episode_metrics['avg_speed'].append(avg_speed)
                
                step += 1
                
                # Print progress every 10% of simulation
                if step % 360 == 0:  # Every 10% of simulation
                    progress = (step / sim_duration) * 100
                    print(f"Progress: {progress:.1f}% | Vehicles: {num_vehicles} | "
                          f"Avg Wait: {avg_waiting_time:.2f}s | Avg Speed: {avg_speed:.2f}m/s")
        
        except Exception as e:
            print(f"Error during simulation: {e}")
        
        finally:
            # Close SUMO
            traci.close()
            
            # Add episode metrics to all episodes list
            all_episodes_metrics.append(pd.DataFrame(episode_metrics))
            
            # Print episode summary
            if len(episode_metrics['avg_waiting_time']) > 0:
                print(f"\nEpisode {episode + 1} Summary:")
                print(f"Average waiting time: {np.mean(episode_metrics['avg_waiting_time']):.2f} seconds")
                print(f"Average speed: {np.mean(episode_metrics['avg_speed']):.2f} m/s")
                print(f"Average number of vehicles: {np.mean(episode_metrics['vehicles_in_system']):.2f}")
    
    # Combine all episodes data and save
    all_data = pd.concat(all_episodes_metrics, ignore_index=True)
    results_dir = os.path.join(project_root, 'logs', 'baseline')
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, 'baseline_metrics.csv')
    all_data.to_csv(output_file, index=False)
    
    # Print overall summary
    print("\nOverall Summary (across all episodes):")
    print(f"Average waiting time: {all_data['avg_waiting_time'].mean():.2f} seconds")
    print(f"Average speed: {all_data['avg_speed'].mean():.2f} m/s")
    print(f"Average number of vehicles: {all_data['vehicles_in_system'].mean():.2f}")
    print(f"\nMetrics saved to: {output_file}")

def main():
    # Use the global hyperparameters
    config = HYPERPARAMETERS.copy()
    config['simulation']['gui'] = False  # Add GUI setting
    
    run_baseline_simulation(config)

if __name__ == '__main__':
    main()