# scripts/test_boston_network.py

import os
import sys
import traci
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def test_network(scenario='morning_rush', gui=True, duration=100):
    """Test the Boston network and collect basic metrics"""
    
    network_dir = Path("data/sumo_nets/boston")
    config_file = network_dir / f"{scenario}.sumocfg"
    
    if not config_file.exists():
        print(f"Configuration file not found: {config_file}")
        return
    
    # Start SUMO
    sumo_binary = "sumo-gui" if gui else "sumo"
    sumo_cmd = [sumo_binary, "-c", str(config_file)]
    
    try:
        traci.start(sumo_cmd)
        
        metrics = {
            'vehicle_count': [],
            'mean_speed': [],
            'waiting_vehicles': [],
            'time_steps': []
        }
        
        print(f"\nRunning {scenario} simulation...")
        for step in range(duration):
            traci.simulationStep()
            
            # Collect metrics
            vehicles = traci.vehicle.getIDList()
            metrics['vehicle_count'].append(len(vehicles))
            
            if vehicles:
                speeds = [traci.vehicle.getSpeed(veh) for veh in vehicles]
                metrics['mean_speed'].append(np.mean(speeds))
            else:
                metrics['mean_speed'].append(0)
            
            waiting = len([veh for veh in vehicles if traci.vehicle.getWaitingTime(veh) > 0])
            metrics['waiting_vehicles'].append(waiting)
            metrics['time_steps'].append(step)
            
            if step % 10 == 0:
                print(f"Step {step}/{duration}: {len(vehicles)} vehicles, "
                      f"Mean speed: {metrics['mean_speed'][-1]:.2f} m/s, "
                      f"Waiting: {waiting}")
        
        # Plot metrics
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        ax1.plot(metrics['time_steps'], metrics['vehicle_count'])
        ax1.set_title('Vehicle Count over Time')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Number of Vehicles')
        
        ax2.plot(metrics['time_steps'], metrics['mean_speed'])
        ax2.set_title('Mean Speed over Time')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Speed (m/s)')
        
        ax3.plot(metrics['time_steps'], metrics['waiting_vehicles'])
        ax3.set_title('Waiting Vehicles over Time')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Number of Vehicles')
        
        plt.tight_layout()
        plot_file = network_dir / f"{scenario}_metrics.png"
        plt.savefig(plot_file)
        print(f"\nPlot saved to {plot_file}")
        
    finally:
        traci.close()

def main():
    # Test all scenarios
    scenarios = ['morning_rush', 'midday', 'evening_rush']
    
    for scenario in scenarios:
        print(f"\nTesting {scenario} scenario...")
        test_network(scenario=scenario, gui=True, duration=100)
        time.sleep(2)  # Wait between scenarios

if __name__ == "__main__":
    main()