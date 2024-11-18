#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import argparse
import yaml
import torch
import numpy as np
from datetime import datetime
import logging
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import traci

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.environment import MultiAgentSumoEnvironment
from experiments.scenarios.communication.state_sharing import StateSharedAgent
from src.utils import setup_logger

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
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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

def calculate_state_sharing_rewards(env, traffic_lights, communication_type):
    """Calculate rewards with state sharing considerations"""
    rewards = {}
    MAX_WAITING_TIME = 150.0
    MIN_GREEN_UTIL = 0.35
    MAX_QUEUE_LENGTH = 10
    
    # Get global traffic state
    total_network_vehicles = sum(
        traci.lane.getLastStepVehicleNumber(lane)
        for tl_id in traffic_lights
        for lane in traci.trafficlight.getControlledLanes(tl_id)
    )
    
    # Calculate per-communication type metrics
    comm_metrics = defaultdict(lambda: defaultdict(float))
    
    # Calculate individual rewards with communication context
    for tl_id in traffic_lights:
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        num_lanes = len(controlled_lanes)
        if num_lanes == 0:
            rewards[tl_id] = 0.0
            continue

        # Compute lane-level metrics
        lane_metrics = {
            'waiting_time': min(sum(traci.lane.getWaitingTime(lane) for lane in controlled_lanes), MAX_WAITING_TIME),
            'queue_length': min(sum(traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes), MAX_QUEUE_LENGTH),
            'vehicles': sum(traci.lane.getLastStepVehicleNumber(lane) for lane in controlled_lanes)
        }
        
        # Reward calculation based on communication type
        if communication_type == 'none':
            # Local reward: Penalize waiting time and queue length
            reward = -lane_metrics['waiting_time'] / MAX_WAITING_TIME - lane_metrics['queue_length'] / MAX_QUEUE_LENGTH
        elif communication_type == 'local':
            # Local reward with neighborhood context
            neighborhood_metrics = lane_metrics.copy()
            reward = -neighborhood_metrics['waiting_time'] / MAX_WAITING_TIME
        else:  # full communication
            # Global reward considering network-wide metrics
            global_factor = lane_metrics['vehicles'] / total_network_vehicles
            reward = global_factor * (1 - lane_metrics['waiting_time'] / MAX_WAITING_TIME)
        
        rewards[tl_id] = reward
    
    return rewards

def calculate_metrics(env, traffic_lights, communication_type, step):
    """Calculate current performance metrics"""
    metrics = defaultdict(float)
    
    # Get global metrics
    total_waiting = 0
    total_queue = 0
    total_vehicles = 0
    total_speed = 0
    
    for tl_id in traffic_lights:
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        
        for lane in controlled_lanes:
            # Cap metrics at reasonable values
            waiting_time = min(traci.lane.getWaitingTime(lane), 150.0)
            queue_length = min(traci.lane.getLastStepHaltingNumber(lane), 10)
            vehicles = traci.lane.getLastStepVehicleNumber(lane)
            speed = traci.lane.getLastStepMeanSpeed(lane)
            max_speed = traci.lane.getMaxSpeed(lane)
            
            total_waiting += waiting_time
            total_queue += queue_length
            total_vehicles += vehicles
            total_speed += speed / max_speed if max_speed > 0 else 0
    
    # Normalize metrics
    num_lanes = sum(len(traci.trafficlight.getControlledLanes(tl_id)) 
                   for tl_id in traffic_lights)
    metrics.update({
        'waiting_time': total_waiting / max(num_lanes, 1),
        'queue_length': total_queue / max(num_lanes, 1),
        'throughput': total_vehicles / max(num_lanes, 1),
        'avg_speed': total_speed / max(num_lanes, 1),
        'communication_type': communication_type,
        'simulation_step': step
    })
    
    return metrics

def plot_training_progress(metrics, save_path):
    """Plot training progress metrics"""
    plt.figure(figsize=(12, 8))
    
    # Plotting different metrics
    plt.subplot(2, 2, 1)
    plt.plot(metrics['mean_rewards'], label='Mean Reward')
    plt.title('Mean Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(metrics['waiting_times'], label='Waiting Time')
    plt.title('Mean Waiting Times')
    plt.xlabel('Episode')
    plt.ylabel('Waiting Time')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(metrics['queue_lengths'], label='Queue Length')
    plt.title('Mean Queue Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Queue Length')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(metrics['throughput'], label='Throughput')
    plt.title('Throughput')
    plt.xlabel('Episode')
    plt.ylabel('Throughput')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def prepare_sumo_config(env_config, route_file):
    """
    Prepare SUMO configuration command line arguments
    
    Args:
        env_config (dict): Environment configuration dictionary
        route_file (str): Path to the route file
    
    Returns:
        str: SUMO command line arguments as a single string
    """
    # Basic SUMO configuration
    sumo_binary = "sumo-gui" if env_config.get('use_gui', False) else "sumo"
    
    # Construct command arguments
    sumo_cmd_list = [
        sumo_binary,
        "-n", str(Path(project_root) / "data" / "grid.net.xml"),  # Network file
        "-r", route_file,  # Route file
        "--step-length", str(env_config.get('delta_time', 1.0)),
        "--no-warnings", "true",
        "--no-step-log", "true",
        "--duration-log.disable", "true",
        "--max-depart-delay", str(env_config.get('max_depart_delay', 100000)),
        "--time-to-teleport", str(env_config.get('time_to_teleport', -1))
    ]
    
    # Optional additional parameters
    if not env_config.get('use_gui', False):
        sumo_cmd_list.extend(["--quit-on-end"])
    
    # Convert to a single string command
    sumo_cmd = " ".join(map(str, sumo_cmd_list))
    
    return sumo_cmd

def train_state_sharing(
    net_file, route_file, out_csv_name, 
    use_gui=False, num_seconds=20000, 
    max_depart_delay=100000, time_to_teleport=-1, 
    delta_time=5, yellow_time=2, min_green=5, 
    max_green=50, neighbor_distance=100,
    communication_types=['none'], num_episodes=5
):
    """Train multi-agent system with different state sharing strategies"""
    # Setup logging
    log_dir = Path(project_root) / 'logs' / 'state_sharing'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('state_sharing_training', log_dir / 'training.log')
    
    # Experiment tracking
    all_experiment_results = {}
    
    # Iterate through communication types
    for comm_type in communication_types:
        logger.info(f"Training with communication type: {comm_type}")
        
        # Initialize metrics dictionary
        metrics = {
            'mean_rewards': [],
            'waiting_times': [],
            'queue_lengths': [],
            'throughput': [],
            'coordination_rate': []
        }
        
        # Initialize environment
        env = MultiAgentSumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            out_csv_name=out_csv_name,
            use_gui=use_gui,
            num_seconds=num_seconds,
            max_depart_delay=max_depart_delay,
            time_to_teleport=time_to_teleport,
            delta_time=delta_time,
            yellow_time=yellow_time,
            min_green=min_green,
            max_green=max_green,
            neighbor_distance=neighbor_distance
        )
        
        # Validate and select traffic lights
        traffic_lights = filter_valid_traffic_lights(env)
        
        # Initialize agents with specific communication strategy
        agents = {
            tl_id: StateSharedAgent(
                state_dim=env.observation_spaces[tl_id].shape[0],
                action_dim=len(env.valid_phases[tl_id]),
                communication_type=comm_type
            ) for tl_id in traffic_lights
        }
        
        # Training loop
        for episode in range(num_episodes):
            logger.info(f"Episode {episode + 1}/{num_episodes} - Communication Type: {comm_type}")
            
            # Reset environment
            env.reset()
            done = False
            step = 0
            
            # Episode-level metrics tracking
            episode_metrics = {
                'rewards': [],
                'waiting_time': [],
                'queue_length': [],
                'throughput': []
            }
            
            while not done:
                # Get current states
                states = {
                    tl_id: env._get_state(tl_id)  
                    for tl_id in traffic_lights
                }
                
                # Select actions
                actions = {
                    tl_id: agents[tl_id].select_action(states[tl_id]) 
                    for tl_id in traffic_lights
                }
                
                # Execute actions
                next_states, rewards, done, _, info = env.step(actions)
                
                # Update agents
                for tl_id in traffic_lights:
                    agents[tl_id].update(
                        states[tl_id], 
                        actions[tl_id], 
                        rewards[tl_id], 
                        next_states[tl_id], 
                        done
                    )
                
                # Calculate and store metrics
                current_metrics = calculate_metrics(
                    env, traffic_lights, comm_type, step
                )
                
                # Detailed logging for debugging
                logger.debug(f"Step {step} Metrics: {current_metrics}")
                
                # Track episode-level metrics
                episode_metrics['rewards'].append(np.mean(list(rewards.values())))
                episode_metrics['waiting_time'].append(current_metrics.get('waiting_time', 0))
                episode_metrics['queue_length'].append(current_metrics.get('queue_length', 0))
                episode_metrics['throughput'].append(current_metrics.get('throughput', 0))
                
                # Periodic logging
                if step % 100 == 0:
                    avg_reward = np.mean(episode_metrics['rewards'][-100:]) if episode_metrics['rewards'] else 0
                    avg_wait = np.mean(episode_metrics['waiting_time'][-100:]) if episode_metrics['waiting_time'] else 0
                    avg_queue = np.mean(episode_metrics['queue_length'][-100:]) if episode_metrics['queue_length'] else 0
                    
                    logger.info(f"\nStep {step}/{num_seconds // delta_time} of Episode {episode+1}")
                    logger.info(f"Average reward: {avg_reward:.3f}")
                    logger.info(f"Average waiting time: {avg_wait:.2f}")
                    logger.info(f"Average queue length: {avg_queue:.2f}")
                    logger.info(f"Simulation time: {step * delta_time} seconds")
                
                step += 1
                
                # Optional: Early stopping or max steps
                if step >= num_seconds // delta_time:
                    break
            
            # Calculate and store episode summary metrics
            metrics['mean_rewards'].append(np.mean(episode_metrics['rewards']) if episode_metrics['rewards'] else 0)
            metrics['waiting_times'].append(np.mean(episode_metrics['waiting_time']) if episode_metrics['waiting_time'] else 0)
            metrics['queue_lengths'].append(np.mean(episode_metrics['queue_length']) if episode_metrics['queue_length'] else 0)
            metrics['throughput'].append(np.mean(episode_metrics['throughput']) if episode_metrics['throughput'] else 0)
            
            # Log episode summary with detailed metrics
            logger.info(f"\nEpisode {episode+1} Summary:")
            logger.info(f"Steps completed: {step}/{num_seconds // delta_time}")
            logger.info(f"Mean reward: {metrics['mean_rewards'][-1]:.2f}")
            logger.info(f"Mean waiting time: {metrics['waiting_times'][-1]:.2f}")
            logger.info(f"Mean queue length: {metrics['queue_lengths'][-1]:.2f}")
            logger.info(f"Average throughput: {metrics['throughput'][-1]:.2f}")
            
            # Debug: Print raw episode metrics
            logger.info("Raw Episode Metrics:")
            logger.info(f"Rewards: {episode_metrics['rewards']}")
            logger.info(f"Waiting Times: {episode_metrics['waiting_time']}")
            logger.info(f"Queue Lengths: {episode_metrics['queue_length']}")
            logger.info(f"Throughput: {episode_metrics['throughput']}")
            logger.info("-" * 50)
            
            # End of episode cleanup
            env.close()
        
        # Plot training progress
        plot_path = log_dir / f'training_progress_{comm_type}.png'
        plot_training_progress(metrics, plot_path)
        
        # Save results
        results_path = log_dir / f'results_{comm_type}.json'
        with open(results_path, 'w') as f:
            json.dump({k: json_serialize(v) for k, v in metrics.items()}, f, indent=4)
        
        # Store results for this communication type
        all_experiment_results[comm_type] = metrics
    
    return all_experiment_results

def main():
    """
    Main function to run state sharing experiment
    """
    # Load configuration
    config_path = Path(project_root) / 'config' / 'state_sharing.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging directory
    log_dir = Path(project_root) / 'logs' / 'state_sharing'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract environment and agent configurations
    env_config = config.get('environment', {})
    agent_config = config.get('agent', {})
    
    # Determine network and route file paths
    net_file = str(Path(project_root) / 'Version1' / '2024-11-05-18-42-37' / 'osm.net.xml.gz')
    route_file = str(Path(project_root) / 'Version1' / '2024-11-05-18-42-37' / 'osm.passenger.trips.xml')
    out_csv_name = str(log_dir / 'state_sharing_results.csv')
    
    # Communication types to experiment with
    communication_types = ['none', 'local', 'full']
    
    # Training configuration
    num_episodes = env_config.get('num_episodes', 15)
    
    # Run experiment
    results = train_state_sharing(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=out_csv_name,
        use_gui=env_config.get('use_gui', False),
        num_seconds=env_config.get('num_seconds', 3600),
        max_depart_delay=env_config.get('max_depart_delay', 100000),
        time_to_teleport=env_config.get('time_to_teleport', -1),
        delta_time=env_config.get('delta_time', 5),
        yellow_time=env_config.get('yellow_time', 2),
        min_green=env_config.get('min_green', 5),
        max_green=env_config.get('max_green', 50),
        neighbor_distance=env_config.get('neighbor_distance', 100),
        communication_types=communication_types,
        num_episodes=num_episodes
    )
    
    # Optional: Visualize or save results
    print("Experiment completed. Results saved in:", log_dir)
    return results

if __name__ == "__main__":
    main()
