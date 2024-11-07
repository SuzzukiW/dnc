# experiments/scenarios/rewards/emissions_rewards.py

import os
import sys
from pathlib import Path
import numpy as np
import yaml
from datetime import datetime
import json
import traci
from collections import defaultdict

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.agents import MultiAgentDQN
from src.environment import MultiAgentSumoEnvironment
from src.utils import setup_logger

class EmissionsRewardEnvironment(MultiAgentSumoEnvironment):
    """
    Environment with emissions-focused reward function
    Extends base environment to prioritize environmental impact
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Emissions reward weights
        self.reward_weights = {
            'co2': 1.0,      # Carbon dioxide
            'co': 2.0,       # Carbon monoxide
            'nox': 3.0,      # Nitrogen oxides
            'pmx': 3.0,      # Particulate matter
            'fuel': 1.5,     # Fuel consumption
            'noise': 0.5,    # Noise emissions
        }
        
        # Historical emissions tracking
        self.emissions_history = defaultdict(list)
        self.baseline_emissions = None
        
    def _get_vehicle_emissions(self, vehicle_id):
        """Get all emission values for a vehicle"""
        try:
            return {
                'co2': traci.vehicle.getCO2Emission(vehicle_id),
                'co': traci.vehicle.getCOEmission(vehicle_id),
                'nox': traci.vehicle.getNOxEmission(vehicle_id),
                'pmx': traci.vehicle.getPMxEmission(vehicle_id),
                'fuel': traci.vehicle.getFuelConsumption(vehicle_id),
                'noise': traci.vehicle.getNoiseEmission(vehicle_id)
            }
        except traci.exceptions.TraCIException:
            return None
            
    def _calculate_intersection_emissions(self, traffic_light_id):
        """Calculate total emissions for vehicles in an intersection"""
        # Get all vehicles in controlled lanes
        controlled_lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
        total_emissions = defaultdict(float)
        num_vehicles = 0
        
        for lane in controlled_lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            for vehicle_id in vehicle_ids:
                emissions = self._get_vehicle_emissions(vehicle_id)
                if emissions:
                    for key, value in emissions.items():
                        total_emissions[key] += value
                    num_vehicles += 1
        
        # Normalize by number of vehicles if any present
        if num_vehicles > 0:
            for key in total_emissions:
                total_emissions[key] /= num_vehicles
                
        return total_emissions
    
    def _get_reward(self, traffic_light_id):
        """
        Calculate reward based on emissions and traffic efficiency
        More negative reward for higher emissions
        """
        # Get emissions for current intersection
        current_emissions = self._calculate_intersection_emissions(traffic_light_id)
        
        # Calculate emissions-based reward component
        emissions_reward = 0
        for emission_type, value in current_emissions.items():
            weight = self.reward_weights.get(emission_type, 1.0)
            emissions_reward -= value * weight
        
        # Get traffic efficiency metrics
        controlled_lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
        waiting_time = 0
        queue_length = 0
        
        for lane in controlled_lanes:
            waiting_time += traci.lane.getWaitingTime(lane)
            queue_length += traci.lane.getLastStepHaltingNumber(lane)
        
        # Combine emissions and efficiency rewards
        # Higher weight on emissions (0.7) vs traffic efficiency (0.3)
        efficiency_reward = -(waiting_time + queue_length)
        total_reward = 0.7 * emissions_reward + 0.3 * efficiency_reward
        
        # Store emissions history
        self.emissions_history[traffic_light_id].append(current_emissions)
        
        return total_reward
    
    def _get_global_reward(self):
        """Calculate global reward based on overall emissions"""
        total_emissions = defaultdict(float)
        vehicle_count = 0
        
        # Sum emissions across all vehicles in the network
        for vehicle_id in traci.vehicle.getIDList():
            emissions = self._get_vehicle_emissions(vehicle_id)
            if emissions:
                for key, value in emissions.items():
                    total_emissions[key] += value
                vehicle_count += 1
        
        # Calculate global emissions reward
        if vehicle_count > 0:
            global_emissions_reward = 0
            for emission_type, total_value in total_emissions.items():
                weight = self.reward_weights.get(emission_type, 1.0)
                global_emissions_reward -= (total_value / vehicle_count) * weight
            
            return global_emissions_reward
        
        return 0
    
    def get_emissions_stats(self):
        """Get statistics about emissions"""
        stats = {
            'average_emissions': defaultdict(float),
            'peak_emissions': defaultdict(float),
            'total_emissions': defaultdict(float),
            'emissions_per_tl': {}
        }
        
        # Calculate per-intersection statistics
        for tl_id, history in self.emissions_history.items():
            tl_stats = defaultdict(list)
            for emission_record in history:
                for emission_type, value in emission_record.items():
                    tl_stats[emission_type].append(value)
            
            # Calculate statistics for this traffic light
            tl_averages = {
                emission_type: np.mean(values) 
                for emission_type, values in tl_stats.items()
            }
            tl_peaks = {
                emission_type: np.max(values) 
                for emission_type, values in tl_stats.items()
            }
            tl_totals = {
                emission_type: np.sum(values) 
                for emission_type, values in tl_stats.items()
            }
            
            stats['emissions_per_tl'][tl_id] = {
                'average': tl_averages,
                'peak': tl_peaks,
                'total': tl_totals
            }
            
            # Update global statistics
            for emission_type in tl_stats:
                stats['average_emissions'][emission_type] += tl_averages[emission_type]
                stats['peak_emissions'][emission_type] = max(
                    stats['peak_emissions'][emission_type],
                    tl_peaks[emission_type]
                )
                stats['total_emissions'][emission_type] += tl_totals[emission_type]
        
        # Normalize averages by number of traffic lights
        num_tl = len(self.emissions_history)
        if num_tl > 0:
            for emission_type in stats['average_emissions']:
                stats['average_emissions'][emission_type] /= num_tl
        
        return stats

def train_emissions_focused(env_config, agent_config, num_episodes=1000):
    """Train agents with emissions-focused rewards"""
    # Initialize custom environment
    env = EmissionsRewardEnvironment(**env_config)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path("experiments/logs/emissions") / timestamp
    model_dir = Path("experiments/models/emissions") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("emissions_training", log_dir / "training.log")
    
    # Initialize multi-agent system
    traffic_lights = env.traffic_lights
    state_size = env.observation_spaces[traffic_lights[0]].shape[0]
    action_size = env.action_spaces[traffic_lights[0]].n
    
    multi_agent_system = MultiAgentDQN(
        state_size=state_size,
        action_size=action_size,
        agent_ids=traffic_lights,
        neighbor_map=env.get_neighbor_map(),
        config=agent_config
    )
    
    # Training metrics
    metrics = {
        'episode_rewards': [],
        'global_rewards': [],
        'emissions_history': [],
        'agent_rewards': defaultdict(list)
    }
    
    try:
        for episode in range(num_episodes):
            states, _ = env.reset()
            episode_rewards = defaultdict(float)
            done = False
            
            while not done:
                # Get actions for all agents
                actions = multi_agent_system.act(states)
                
                # Execute actions
                next_states, rewards, done, _, info = env.step(actions)
                
                # Update agents
                losses = multi_agent_system.step(
                    states, actions, rewards, next_states,
                    {agent_id: done for agent_id in traffic_lights},
                    global_reward=info['global_reward']
                )
                
                # Update metrics
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward
                
                states = next_states
            
            # Calculate episode metrics
            emissions_stats = env.get_emissions_stats()
            
            # Log progress
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            logger.info(f"Average CO2: {emissions_stats['average_emissions']['co2']:.2f}")
            logger.info(f"Average NOx: {emissions_stats['average_emissions']['nox']:.2f}")
            logger.info(f"Total Fuel: {emissions_stats['total_emissions']['fuel']:.2f}")
            
            # Store metrics
            metrics['episode_rewards'].append(episode_rewards)
            metrics['global_rewards'].append(info['global_reward'])
            metrics['emissions_history'].append(emissions_stats)
            
            # Save models periodically
            if (episode + 1) % 100 == 0:
                save_dir = model_dir / f"episode_{episode + 1}"
                save_dir.mkdir(parents=True, exist_ok=True)
                multi_agent_system.save_agents(save_dir)
                
                # Save metrics
                metrics_path = log_dir / f"metrics_episode_{episode + 1}.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4, default=str)
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        env.close()
    
    return metrics

def main():
    # Load configurations
    with open("config/env_config.yaml", 'r') as f:
        env_config = yaml.safe_load(f)
    with open("config/agent_config.yaml", 'r') as f:
        agent_config = yaml.safe_load(f)
    
    # Add emissions-specific configurations
    env_config.update({
        'neighbor_distance': 100,  # meters
        'yellow_time': 2,
        'min_green': 5,
        'max_green': 50
    })
    
    agent_config.update({
        'memory_size': 100000,
        'communication_mode': 'shared_experience',
        'reward_type': 'emissions'
    })
    
    # Run training
    metrics = train_emissions_focused(env_config, agent_config)
    print("Training completed!")
    
    # Print final emissions statistics
    final_emissions = metrics['emissions_history'][-1]
    print("\nFinal Emissions Statistics:")
    print(f"Average CO2: {final_emissions['average_emissions']['co2']:.2f}")
    print(f"Average NOx: {final_emissions['average_emissions']['nox']:.2f}")
    print(f"Total Fuel Consumption: {final_emissions['total_emissions']['fuel']:.2f}")

if __name__ == "__main__":
    main()