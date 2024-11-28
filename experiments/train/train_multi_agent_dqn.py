# experiments/train/train_multi_agent_dqn.py

import os
import sys
import time
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import traci
from evaluation_sets.metrics import (
    average_waiting_time,
    total_throughput,
    average_speed,
    max_waiting_time
)

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.environment.multi_agent_sumo_env import MultiAgentSumoEnvironment
from src.agents.cooperative_dqn_agent import MultiAgentDQN
from src.utils.region_manager import RegionManager

def json_serialize(obj):
    """Helper function to serialize NumPy types for JSON"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                       np.int16, np.int32, np.int64, np.uint8,
                       np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        # Return default configuration
        return {
            'environment': {
                'net_file': os.path.join('Version1', '2024-11-05-18-42-37', 'osm.net.xml.gz'),
                'route_file': os.path.join('Version1', '2024-11-05-18-42-37', 'osm.passenger.trips.xml'),
                'num_seconds': 100,
                'delta_time': 8,
                'yellow_time': 3,
                'min_green': 12,
                'max_green': 45
            },
            'agent': {
                'num_episodes': 100,
                'batch_size': 64,
                'memory_size': 100000,
                'target_update': 10,
                'gamma': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay': 0.995,
                'learning_rate': 0.001,
                'regional_weight': 0.3
            },
            'training': {
                'num_episodes': 100
            }
        }

def filter_valid_traffic_lights(env):
    """Filter and validate traffic lights that have proper phase definitions"""
    all_traffic_lights = env.traffic_lights
    valid_traffic_lights = []
    skipped_traffic_lights = []
    
    print("\nValidating traffic lights...")
    print(f"Total traffic lights found: {len(all_traffic_lights)}")
    
    for tl_id in all_traffic_lights:
        try:
            # Get traffic light program
            program = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            phases = program.phases
            
            # Check if traffic light has phases
            if not phases:
                skipped_traffic_lights.append((tl_id, "No phases defined"))
                continue
                
            # Check if traffic light has valid phases (with green states)
            valid_phases = [i for i, phase in enumerate(phases) 
                          if any(c in 'gG' for c in phase.state)]
            
            if not valid_phases:
                skipped_traffic_lights.append((tl_id, "No valid green phases"))
                continue
            
            # Check if traffic light controls any lanes
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            if not controlled_lanes:
                skipped_traffic_lights.append((tl_id, "No controlled lanes"))
                continue
                
            # If we get here, the traffic light is valid
            valid_traffic_lights.append(tl_id)
            
        except Exception as e:
            skipped_traffic_lights.append((tl_id, str(e)))
            continue
    
    # Print summary
    print("\nTraffic Light Validation Summary:")
    print(f"Valid traffic lights: {len(valid_traffic_lights)}")
    print(f"Skipped traffic lights: {len(skipped_traffic_lights)}")
    
    if skipped_traffic_lights:
        print("\nSkipped Traffic Lights Details:")
        for tl_id, reason in skipped_traffic_lights:
            print(f"- {tl_id}: {reason}")
    
    if not valid_traffic_lights:
        raise ValueError("No valid traffic lights found in the network!")
    
    return valid_traffic_lights

def calculate_rewards(env, traffic_lights):
    """Balanced reward calculation with stability focus"""
    rewards = {}
    MAX_WAITING_TIME = 150.0  # Reduced back
    MIN_GREEN_UTIL = 0.35
    MAX_QUEUE_LENGTH = 10
    
    # Get global traffic state
    total_network_vehicles = sum(
        traci.lane.getLastStepVehicleNumber(lane)
        for tl_id in traffic_lights
        for lane in traci.trafficlight.getControlledLanes(tl_id)
    )
    
    for tl_id in traffic_lights:
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        num_lanes = len(controlled_lanes)
        if num_lanes == 0:
            continue
        
        lane_metrics = defaultdict(dict)
        total_vehicles = 0
        green_lanes = []
        
        # Get current phase info
        phase_state = traci.trafficlight.getRedYellowGreenState(tl_id)
        green_indices = [i for i, state in enumerate(phase_state) if state in 'Gg']
        
        # Get neighbor states for coordination
        neighbor_pressure = 0
        if hasattr(env, 'neighbor_map'):
            neighbors = env.neighbor_map.get(tl_id, [])
            for neighbor in neighbors:
                neighbor_lanes = traci.trafficlight.getControlledLanes(neighbor)
                if neighbor_lanes:
                    neighbor_pressure += sum(
                        traci.lane.getLastStepVehicleNumber(lane) / 
                        max(traci.lane.getLength(lane), 1)
                        for lane in neighbor_lanes
                    ) / len(neighbor_lanes)
        
        # Collect enhanced per-lane metrics
        for lane in controlled_lanes:
            metrics = lane_metrics[lane]
            
            # Basic metrics
            metrics['queue'] = min(traci.lane.getLastStepHaltingNumber(lane), MAX_QUEUE_LENGTH)
            metrics['waiting_time'] = min(traci.lane.getWaitingTime(lane), MAX_WAITING_TIME)
            metrics['vehicles'] = traci.lane.getLastStepVehicleNumber(lane)
            metrics['speed'] = traci.lane.getLastStepMeanSpeed(lane)
            metrics['max_speed'] = traci.lane.getMaxSpeed(lane)
            metrics['is_green'] = lane in green_lanes
            
            # Advanced metrics
            metrics['occupancy'] = traci.lane.getLastStepOccupancy(lane)
            metrics['relative_speed'] = metrics['speed'] / max(metrics['max_speed'], 1)
            
            # Calculate lane pressure
            metrics['pressure'] = metrics['vehicles'] / max(traci.lane.getLength(lane), 1)
            
            total_vehicles += metrics['vehicles']
        
        # Skip if no vehicles
        if total_vehicles == 0:
            rewards[tl_id] = 0.0
            continue
        
        # Calculate enhanced lane utilization
        green_utilization = sum(1 for l in green_lanes if lane_metrics[l]['vehicles'] > 0) / max(len(green_lanes), 1)
        
        # Dynamic weights based on traffic conditions
        traffic_density = total_vehicles / (num_lanes * MAX_QUEUE_LENGTH)
        base_weights = {
            'queue': 0.28,        # Balanced weights
            'waiting': 0.28,      
            'speed': 0.18,
            'throughput': 0.18,   
            'coordination': 0.04,
            'green_util': 0.04
        }

        # Adjust weights based on traffic density and neighbor pressure
        metrics_weights = base_weights.copy()
        if traffic_density > 0.7:  # Heavy traffic
            metrics_weights.update({
                'queue': 0.30,
                'waiting': 0.30,
                'throughput': 0.20,
                'speed': 0.20
            })
        elif traffic_density < 0.3:  # Light traffic
            metrics_weights.update({
                'speed': 0.25,
                'throughput': 0.25,
                'waiting': 0.25,
                'queue': 0.25
            })
        
        # Calculate enhanced component rewards
        reward_components = {
            'queue': -sum(m['queue'] for m in lane_metrics.values()) / (num_lanes * MAX_QUEUE_LENGTH),
            'waiting': -sum(m['waiting_time'] for m in lane_metrics.values()) / (num_lanes * MAX_WAITING_TIME),
            'speed': sum(m['relative_speed'] for m in lane_metrics.values()) / num_lanes,
            'throughput': total_vehicles / (num_lanes * MAX_QUEUE_LENGTH),
            'coordination': -abs(neighbor_pressure - total_vehicles/num_lanes) if neighbor_pressure > 0 else 0,
            'green_util': max(0, green_utilization - MIN_GREEN_UTIL) * 2
        }
        
        # Progressive scaling based on multiple factors
        network_load = total_network_vehicles / (len(traffic_lights) * num_lanes * MAX_QUEUE_LENGTH)
        scale_factors = [
            1.0 + (traffic_density * 0.3),
            1.0 + (network_load * 0.2),
            1.0 + (green_utilization * 0.1)
        ]
        scale = np.mean(scale_factors)
        
        # Calculate final reward with components
        reward = sum(component * metrics_weights[name] 
                    for name, component in reward_components.items())
        reward = reward * scale
        
        # Add improvement bonuses
        if hasattr(env, 'previous_metrics') and tl_id in env.previous_metrics:
            prev = env.previous_metrics[tl_id]
            curr = reward_components
            
            waiting_improvement = curr['waiting'] - prev['waiting']
            queue_improvement = curr['queue'] - prev['queue']
            
            # Single combined bonus
            if waiting_improvement > 0.05 and queue_improvement > 0.05:
                reward += 0.2
            
            # Significant improvement bonus
            if (curr['waiting'] > prev['waiting'] * 1.2 and 
                curr['queue'] > prev['queue'] * 1.2):
                reward += 0.3
            
            # Sustained performance bonus
            elif (curr['waiting'] >= prev['waiting'] * 0.9 and 
                  curr['queue'] >= prev['queue'] * 0.9):
                reward += 0.1
        
        # Store current metrics for next comparison
        if not hasattr(env, 'previous_metrics'):
            env.previous_metrics = {}
        env.previous_metrics[tl_id] = reward_components
        
        # Final scaling and clipping
        rewards[tl_id] = float(np.clip(reward, -1.0, 1.0))
    
    return rewards

def plot_metrics(metrics, save_path):
    """Plot training metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot average waiting time
    ax1.plot(metrics['average_waiting_time'])
    ax1.set_title('Average Waiting Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Time (s)')
    
    # Plot throughput
    ax2.plot(metrics['total_throughput'])
    ax2.set_title('Total Throughput')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Vehicles')
    
    # Plot average speed
    ax3.plot(metrics['average_speed'])
    ax3.set_title('Average Speed')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Speed (km/h)')
    
    # Plot max waiting time
    ax4.plot(metrics['max_waiting_time'])
    ax4.set_title('Maximum Waiting Time')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_metrics(metrics, path):
    """Save metrics with proper JSON serialization"""
    # No need to convert to list since metrics are already lists
    serialized_metrics = {
        'average_waiting_time': metrics['average_waiting_time'],
        'total_throughput': metrics['total_throughput'],
        'average_speed': metrics['average_speed'],
        'max_waiting_time': metrics['max_waiting_time']
    }
    
    try:
        # Create metrics directory if it doesn't exist
        metrics_dir = os.path.dirname(path)
        os.makedirs(metrics_dir, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(serialized_metrics, f, indent=4, default=json_serialize)
            
    except Exception as e:
        print(f"Error saving metrics: {str(e)}")

def train_multi_agent(env_config, agent_config, num_episodes=100):
    """Train multiple cooperative DQN agents with enhanced traffic management"""
    
    # Ensure network and route files exist
    if not os.path.exists(env_config['net_file']):
        raise FileNotFoundError(f"Network file {env_config['net_file']} not found")
    
    if not os.path.exists(env_config['route_file']):
        raise FileNotFoundError(f"Route file {env_config['route_file']} not found")
    
    # Initialize environment
    env = MultiAgentSumoEnvironment(config=env_config)
    
    # Filter valid traffic lights
    valid_traffic_lights = filter_valid_traffic_lights(env)
    print(f"\nProceeding with training using {len(valid_traffic_lights)} valid traffic lights")
    
    # Update environment's traffic lights to only include valid ones
    env.traffic_lights = valid_traffic_lights
    
    # Get neighbor map for valid traffic lights only
    full_neighbor_map = env.get_neighbor_map()
    neighbor_map = {tl: [n for n in neighbors if n in valid_traffic_lights] 
                   for tl, neighbors in full_neighbor_map.items()
                   if tl in valid_traffic_lights}
    
    print("\nNetwork Configuration:")
    print(f"Average neighbors per agent: {np.mean([len(n) for n in neighbor_map.values()]):.1f}")
    print(f"Total connections: {sum(len(n) for n in neighbor_map.values())}")
    
    # Get steps per episode from environment config
    steps_per_episode = env_config['max_steps']
    print(f"\nTraining Configuration:")
    print(f"Steps per episode: {steps_per_episode}")
    print(f"Action interval: {env_config['delta_time']} seconds")
    
    # Initialize traffic pressure tracking
    if not hasattr(env, 'traffic_pressure'):
        env.traffic_pressure = defaultdict(float)
    
    # Initialize multi-agent system
    state_size = env.observation_spaces[valid_traffic_lights[0]].shape[0]
    
    # Modify action size handling for continuous action spaces
    valid_traffic_lights = filter_valid_traffic_lights(env)
    
    # Get the first valid traffic light's action space
    first_tl = valid_traffic_lights[0]
    action_space = env.action_spaces[first_tl]
    
    # For continuous action spaces, use the shape of the action space
    action_size = action_space.shape[0]
    
    print("\nAgent Configuration:")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    
    # Initialize multi-agent system with proper parameters
    multi_agent_system = MultiAgentDQN(
        valid_traffic_lights=valid_traffic_lights,
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        neighbor_map=neighbor_map,
        config=agent_config
    )
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = Path("experiments/models") / f"dqn_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (model_dir / "checkpoints").mkdir(exist_ok=True)
    (model_dir / "final").mkdir(exist_ok=True)
    (model_dir / "metrics").mkdir(exist_ok=True)
    
    # Initialize metrics dictionary for all episodes
    all_metrics = {
        'average_waiting_time': [],
        'total_throughput': [],
        'average_speed': [],
        'max_waiting_time': []
    }
    
    # Initialize metrics for current episode
    episode_metrics = {
        'average_waiting_time': [],
        'total_throughput': [],
        'average_speed': [],
        'max_waiting_time': []
    }
    
    # Track vehicles for metrics
    vehicles_seen = set()
    vehicles_completed = set()
    
    try:
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            try:
                states, _ = env.reset(return_info=True)
                states = {tl: state for tl, state in states.items() 
                         if tl in valid_traffic_lights}
                
                episode_rewards = defaultdict(float)
                episode_losses = defaultdict(list)
                episode_queues = defaultdict(list)
                episode_waiting_times = defaultdict(list)
                episode_throughput = []
                episode_pressure = defaultdict(list)
                done = False
                step = 0
                
                # Create progress bar for this episode
                pbar = tqdm(total=steps_per_episode, desc=f'Episode {episode + 1}/{num_episodes}')
                
                while step < steps_per_episode:
                    try:
                        # Update traffic pressure for each intersection
                        for tl_id in valid_traffic_lights:
                            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                            total_pressure = sum(
                                traci.lane.getLastStepVehicleNumber(lane) / 
                                traci.lane.getLength(lane)
                                for lane in controlled_lanes
                            )
                            env.traffic_pressure[tl_id] = total_pressure / len(controlled_lanes)
                            episode_pressure[tl_id].append(env.traffic_pressure[tl_id])
                        
                        # Get adaptive actions based on traffic pressure
                        actions = {}
                        for tl_id in valid_traffic_lights:
                            # Adjust epsilon based on traffic pressure
                            if env.traffic_pressure[tl_id] > 0.8:  # High pressure
                                multi_agent_system.agents[tl_id].epsilon = max(
                                    multi_agent_system.agents[tl_id].epsilon_min,
                                    multi_agent_system.agents[tl_id].epsilon * 0.5
                                )
                            elif env.traffic_pressure[tl_id] < 0.2:  # Low pressure
                                multi_agent_system.agents[tl_id].epsilon = min(
                                    1.0,
                                    multi_agent_system.agents[tl_id].epsilon * 1.5
                                )
                            
                            # Get action for this agent
                            actions[tl_id] = multi_agent_system.agents[tl_id].act(states[tl_id])
                        
                        # Apply actions to each traffic light
                        actions = {}
                        for tl_id, agent in zip(valid_traffic_lights, multi_agent_system.agents.values()):
                            # Get continuous action from the agent
                            action = agent.act(states[tl_id])
                            
                            # Clip action to the environment's action space
                            action_space = env.action_spaces[tl_id]
                            
                            # Ensure action matches the action space shape
                            if isinstance(action, (list, np.ndarray)):
                                # If action is an array, ensure it matches action space
                                if len(action) != len(action_space.low):
                                    # Resize or adjust action to match action space
                                    action = np.zeros_like(action_space.low)
                            else:
                                # If action is a scalar, convert to array
                                action = np.full_like(action_space.low, action)
                            
                            # Clip action to the environment's action space
                            action = np.clip(action, action_space.low, action_space.high)
                            actions[tl_id] = action
                        
                        # Execute actions
                        next_states, rewards, dones, info = env.step(actions)
                        
                        # Calculate custom rewards with pressure consideration
                        custom_rewards = calculate_rewards(env, valid_traffic_lights)
                        
                        # Calculate traffic metrics
                        vehicle_data = [
                            {
                                'waiting_time': traci.vehicle.getWaitingTime(vehicle_id),
                                'speed': traci.vehicle.getSpeed(vehicle_id)
                            }
                            for vehicle_id in traci.vehicle.getIDList()
                        ]
                        
                        # Update episode metrics
                        episode_metrics['average_waiting_time'].append(average_waiting_time(vehicle_data))
                        episode_metrics['total_throughput'].append(len(vehicles_completed))
                        episode_metrics['average_speed'].append(average_speed(vehicle_data))
                        episode_metrics['max_waiting_time'].append(max_waiting_time(vehicle_data))
                        
                        # Update agents with custom rewards
                        losses = multi_agent_system.step(
                            states, actions, custom_rewards, next_states, dones
                        )
                        
                        # Update states for next iteration
                        states = next_states
                        
                        # Update metrics
                        current_vehicles = set(traci.vehicle.getIDList())
                        vehicles_seen.update(current_vehicles)
                        
                        # Track completed trips
                        completed = vehicles_seen - current_vehicles
                        vehicles_completed.update(completed)
                        
                        # Update progress bar with current metrics
                        avg_wait = np.mean([d['waiting_time'] for d in vehicle_data]) if vehicle_data else 0
                        pbar.set_description(
                            f'Episode {episode + 1}/{num_episodes} '
                            f'[Vehicles: {len(current_vehicles)}, '
                            f'Completed: {len(vehicles_completed)}, '
                            f'Wait: {avg_wait:.1f}s]'
                        )
                        pbar.update(1)
                        
                        step += 1
                    
                    except Exception as e:
                        print(f"Error during step {step}: {str(e)}")
                        # Try to reset the environment
                        states, _ = env.reset(return_info=True)
                        states = {tl: state for tl, state in states.items() 
                                 if tl in valid_traffic_lights}
                        continue
                
                # Close progress bar at the end of episode
                pbar.close()
                
                # Store final metrics for this episode
                for metric in episode_metrics:
                    if len(episode_metrics[metric]) > 0:
                        all_metrics[metric].append(episode_metrics[metric][-1])
                    else:
                        all_metrics[metric].append(0.0)

                # Print episode summary
                print(f"\nEpisode {episode + 1}/{num_episodes} Summary:")
                print(f"Steps completed: {step}/{steps_per_episode}")
                print(f"Mean reward: {np.mean([r for r in rewards.values()]):.2f}")
                print(f"Global reward: {sum(rewards.values()):.2f}")
                print("\nTraffic Metrics:")
                if len(episode_metrics['average_waiting_time']) > 0:
                    print(f"Average Waiting Time: {episode_metrics['average_waiting_time'][-1]:.2f} seconds")
                    print(f"Total Throughput: {episode_metrics['total_throughput'][-1]} vehicles")
                    print(f"Average Speed: {episode_metrics['average_speed'][-1]:.2f} km/h")
                    print(f"Maximum Waiting Time: {episode_metrics['max_waiting_time'][-1]:.2f} seconds")
                else:
                    print("No traffic metrics available for this episode")
                
                # Reset episode metrics for next episode
                episode_metrics = {
                    'average_waiting_time': [],
                    'total_throughput': [],
                    'average_speed': [],
                    'max_waiting_time': []
                }
            
            except Exception as e:
                print(f"Error during episode {episode + 1}: {str(e)}")
                continue
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Critical training error: {str(e)}")
    finally:
        try:
            # Print final summary across all episodes
            print("\nDQN Training Results - Averaged over {} episodes:".format(num_episodes))
            print(f"Average Waiting Time: {np.mean(all_metrics['average_waiting_time']):.2f} ± {np.std(all_metrics['average_waiting_time']):.2f} seconds")
            print(f"Average Total Throughput: {np.mean(all_metrics['total_throughput']):.2f} ± {np.std(all_metrics['total_throughput']):.2f} vehicles")
            print(f"Average Speed: {np.mean(all_metrics['average_speed']):.2f} ± {np.std(all_metrics['average_speed']):.2f} km/h")
            print(f"Average Max Waiting Time: {np.mean(all_metrics['max_waiting_time']):.2f} ± {np.std(all_metrics['max_waiting_time']):.2f} seconds")
            
            # Plot and save final metrics
            plot_metrics(all_metrics, model_dir / "final" / "metrics.png")
            
            # Save metrics to JSON
            with open(model_dir / "final" / "metrics.json", 'w') as f:
                json.dump(all_metrics, f, indent=4, default=json_serialize)
                
            print("Final state saved successfully")
            print("\nTraining Summary:")
            print(f"Episodes completed: {num_episodes}")
            print(f"Best waiting time: {min(all_metrics['average_waiting_time']):.2f} seconds")
            print(f"Final waiting time: {all_metrics['average_waiting_time'][-1]:.2f} seconds")
            print(f"Best throughput: {max(all_metrics['total_throughput']):.1f} vehicles")
            print(f"Final throughput: {all_metrics['total_throughput'][-1]:.1f} vehicles")
            
            # Calculate improvements
            initial_waiting_time = all_metrics['average_waiting_time'][0]
            final_waiting_time = all_metrics['average_waiting_time'][-1]
            initial_throughput = all_metrics['total_throughput'][0]
            final_throughput = all_metrics['total_throughput'][-1]
            
            waiting_time_improvement = ((initial_waiting_time - final_waiting_time) / 
                                      initial_waiting_time * 100 if initial_waiting_time > 0 else 0)
            throughput_improvement = ((final_throughput - initial_throughput) / 
                                    initial_throughput * 100 if initial_throughput > 0 else 0)
            
            print("\nPerformance Improvements:")
            print(f"Waiting time reduction: {waiting_time_improvement:.1f}%")
            print(f"Throughput improvement: {throughput_improvement:.1f}%")
            
            print(f"\nResults saved to: {model_dir}")
        
        except Exception as e:
            print(f"Error saving final state: {str(e)}")
        finally:
            env.close()
    
    return all_metrics

def main():
    """Main training function"""
    try:
        # Close any existing SUMO connections
        try:
            traci.close()
        except:
            pass
            
        # Load configuration with absolute path
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / 'config' / 'dqn_config.yaml'
        config = load_config(config_path)
        
        # Initialize environment
        env = MultiAgentSumoEnvironment(config=config)
        
        # Get valid traffic lights
        valid_traffic_lights = filter_valid_traffic_lights(env)
        
        # Initialize multi-agent system
        multi_agent_system = MultiAgentDQN(
            valid_traffic_lights=valid_traffic_lights,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            neighbor_map=env.neighbor_map,
            config=config['agent']
        )
        
        # Train agents
        metrics = train_multi_agent(
            env_config=config['environment'],
            agent_config=config['agent'],
            num_episodes=config['training']['num_episodes']
        )
        
        # Save metrics
        save_metrics(metrics, "experiments/results/metrics.json")
        
        # Plot training results
        plot_metrics(metrics, "experiments/results/training_plots.png")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        # Ensure SUMO connection is closed on error
        try:
            traci.close()
        except:
            pass
        raise e
    finally:
        # Always try to close SUMO connection
        try:
            traci.close()
        except:
            pass

if __name__ == "__main__":
    main()