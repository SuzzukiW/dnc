# experiments/train/train_state_sharing.py

import os
import sys
import logging
import yaml
import numpy as np
from typing import Dict, List
import torch
from datetime import datetime
import time

from src.environment.multi_agent_sumo_env_state_sharing import MultiAgentSumoEnvStateSharing
from src.agents.dqn_agent import DQNAgent
from experiments.scenarios.communication.state_sharing import StateSharing
import traci

from evaluation_sets.metrics import (
    average_waiting_time, 
    total_throughput, 
    average_speed
)

def setup_logging(config: Dict):
    """Set up logging configuration"""
    log_dir = config['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_agents(env: MultiAgentSumoEnvStateSharing, config: Dict) -> Dict[str, DQNAgent]:
    """Create DQN agents for each traffic light"""
    agents = {}
    
    # Create scenario instance
    scenario = StateSharing(
        net_file=config['env']['net_file'],
        route_file=config['env']['route_file'],
        use_gui=config['env']['use_gui'],
        num_episodes=config['training']['num_episodes'],
        episode_length=config['env']['episode_length'],
        delta_time=config['env']['delta_time']
    )
    
    # Start simulation temporarily to get state dimensions
    env._start_simulation()
    
    # Get a sample state for a traffic light
    first_tl = next(iter(env.traffic_signals))
    sample_state = env._get_state(first_tl)
    
    for tl_id in env.traffic_signals:
        # Get dimensionality of combined state
        state_dim = len(scenario.preprocess_state(sample_state))
        
        # Action space is number of green phases
        action_dim = env.traffic_signals[tl_id]['num_green_phases']
        
        # Create agent
        agents[tl_id] = DQNAgent(
            state_size=state_dim,
            action_size=action_dim,
            config=config['agent']
        )
    
    traci.close()
    return agents

def _calculate_metrics(env, episode_rewards, episode_start_time, episode_losses):
    """Calculate metrics matching the Baseline simulation."""
    metrics = {}
    
    # Calculate average waiting time
    total_waiting_time = 0
    max_waiting_time = 0
    vehicle_count = 0
    
    # Collect metrics across all lanes controlled by traffic lights
    for tl_id, tl_data in env.traffic_signals.items():
        for lane_id in tl_data['lanes']:
            try:
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                for vid in vehicles:
                    wait_time = traci.vehicle.getAccumulatedWaitingTime(vid)
                    total_waiting_time += wait_time
                    max_waiting_time = max(max_waiting_time, wait_time)
                    vehicle_count += 1
            except traci.exceptions.TraCIException:
                continue  # Skip invalid lanes
    
    # Calculate metrics
    metrics['average_waiting_time'] = total_waiting_time / max(1, vehicle_count)
    metrics['max_waiting_time'] = max_waiting_time
    metrics['total_throughput'] = vehicle_count
    
    # Calculate average speed across all valid lanes
    total_speed = 0
    valid_lanes = 0
    for tl_id, tl_data in env.traffic_signals.items():
        for lane_id in tl_data['lanes']:
            try:
                speed = traci.lane.getLastStepMeanSpeed(lane_id)
                total_speed += speed * 3.6  # Convert m/s to km/h
                valid_lanes += 1
            except traci.exceptions.TraCIException:
                continue
    
    metrics['average_speed'] = total_speed / max(1, valid_lanes)
    
    return metrics

def _log_episode_metrics(logger, episode, metrics):
    """Log metrics in a format similar to Baseline."""
    print(f"\nEpisode {episode + 1} Metrics:")
    print(f"Average Waiting Time: {metrics['average_waiting_time']:.2f} seconds")
    print(f"Total Throughput: {metrics['total_throughput']} vehicles")
    print(f"Average Speed: {metrics['average_speed']:.2f} km/h")
    print(f"Max Waiting Time: {metrics['max_waiting_time']:.2f} seconds")

def setup_sumo_path():
    """Attempt to set up SUMO path"""
    possible_sumo_paths = [
        '/opt/homebrew/bin/sumo',  # Homebrew path
        '/usr/local/bin/sumo',     # Standard Unix path
        os.path.expanduser('~/sumo/bin/sumo')  # User home directory
    ]
    
    # Check if SUMO_HOME is set
    if 'SUMO_HOME' not in os.environ:
        for path in possible_sumo_paths:
            if os.path.exists(path):
                sumo_home = os.path.dirname(os.path.dirname(path))
                os.environ['SUMO_HOME'] = sumo_home
                logging.info(f"Set SUMO_HOME to {sumo_home}")
                break
        else:
            logging.error("Could not find SUMO installation. Please install SUMO and set SUMO_HOME.")
            sys.exit(1)
    
    # Add SUMO tools to Python path
    tools_path = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools_path not in sys.path:
        sys.path.append(tools_path)

def _share_states(agents: Dict[str, DQNAgent], scenario: StateSharing, states: Dict[str, np.ndarray]) -> (Dict[str, np.ndarray], int):
    """
    Share states between agents using SUMO metrics to prioritize state exchange.
    
    Args:
        agents (Dict[str, DQNAgent]): Dictionary of agents
        scenario (StateSharing): Scenario configuration
        states (Dict[str, np.ndarray]): Current states of all agents
    
    Returns:
        Dict[str, np.ndarray]: Updated states after sharing
        int: Number of shared states
    """
    # Prepare vehicle data for metrics calculation
    vehicle_data = [{'waiting_time': traci.vehicle.getWaitingTime(vid), 
                     'speed': traci.vehicle.getSpeed(vid)} 
                    for vid in traci.vehicle.getIDList()]
    
    # Calculate metrics for each agent's state
    agent_metrics = {}
    for tl_id, state in states.items():
        metrics = {
            'avg_waiting_time': average_waiting_time(vehicle_data),
            'throughput': total_throughput(vehicle_data),
            'avg_speed': average_speed(vehicle_data)
        }
        
        # Store metrics with the state
        agent_metrics[tl_id] = {
            'metrics': metrics,
            'state': state
        }
    
    # Determine state sharing weights
    def calculate_state_sharing_weight(metrics):
        """
        Calculate a weight for state sharing based on traffic performance metrics.
        
        Args:
            metrics (dict): Calculated SUMO metrics for an agent
        
        Returns:
            float: Weight for state sharing
        """
        # Normalize metrics
        waiting_time_weight = 1 / (metrics['avg_waiting_time'] + 1)  # Lower waiting time is better
        throughput_weight = metrics['throughput'] / 100.0  # Normalize throughput
        speed_weight = metrics['avg_speed'] / 50.0  # Normalize speed (assuming typical urban speeds)
        
        # Combine weights with different importance
        return 0.4 * waiting_time_weight + 0.3 * throughput_weight + 0.3 * speed_weight
    
    # Sort agents by their state sharing weights
    sorted_agents = sorted(
        agent_metrics.items(), 
        key=lambda x: calculate_state_sharing_weight(x[1]['metrics']), 
        reverse=True
    )
    
    # Share states between top-performing agents
    shared_states = states.copy()
    num_agents = len(sorted_agents)
    shared_count = 0
    
    for i in range(num_agents):
        for j in range(i+1, num_agents):
            agent_id_1, agent_1_data = sorted_agents[i]
            agent_id_2, agent_2_data = sorted_agents[j]
            
            # Blend states using a weighted average
            blended_state = (
                0.6 * agent_1_data['state'] + 
                0.4 * agent_2_data['state']
            )
            
            # Update states for both agents
            shared_states[agent_id_1] = blended_state
            shared_states[agent_id_2] = blended_state
            
            shared_count += 1
    
    return shared_states, shared_count

def train(config_path: str):
    """Main training loop with simplified metrics collection"""
    setup_sumo_path()
    # Load configuration
    config = load_config(config_path)
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Create environment
    env = MultiAgentSumoEnvStateSharing(
        net_file=config['env']['net_file'],
        route_file=config['env']['route_file'],
        use_gui=config['env']['use_gui'],
        num_seconds=config['env']['episode_length'],
        delta_time=config['env']['delta_time'],
        max_steps=config['training'].get('max_steps', None)  # Get max_steps from config if available
    )
    
    # Create scenario
    scenario = StateSharing(
        net_file=config['env']['net_file'],
        route_file=config['env']['route_file'],
        use_gui=config['env']['use_gui'],
        num_episodes=config['training']['num_episodes'],
        episode_length=config['env']['episode_length'],
        delta_time=config['env']['delta_time']
    )
    
    # Create agents
    agents = create_agents(env, config)
    
    # Training metrics tracking
    all_metrics = {
        'average_waiting_time': [],
        'total_throughput': [],
        'average_speed': [],
        'max_waiting_time': []
    }
    
    # Progress tracking
    total_episodes = config['training']['num_episodes']
    print(f"\n{'='*50}")
    print(f"Starting Multi-Agent State Sharing Training")
    print(f"Total Episodes: {total_episodes}")
    print(f"{'='*50}\n")
    
    # Training loop
    for episode in range(config['training']['num_episodes']):
        # Progress bar
        progress_percentage = (episode + 1) / total_episodes * 100
        print(f"\rTraining Progress: [{'#'*int(progress_percentage/5)}{'-'*(20-int(progress_percentage/5))}] "
              f"{progress_percentage:.1f}% (Episode {episode+1}/{total_episodes})", end='', flush=True)
        
        states = env.reset()
        episode_rewards = {tl_id: 0 for tl_id in env.traffic_signals}
        episode_losses = {tl_id: [] for tl_id in env.traffic_signals}
        done = {'__all__': False}
        step = 0
        episode_start_time = time.time()
        
        # State sharing tracking
        total_state_shares = 0
        
        while not done['__all__']:
            # Process states through scenario
            processed_states = {
                tl_id: scenario.preprocess_state(state) 
                for tl_id, state in states.items()
            }
            
            # Share states between agents
            shared_states, shared_count = _share_states(agents, scenario, processed_states)
            total_state_shares += shared_count
            
            # Agent actions
            actions = {}
            for tl_id, agent in agents.items():
                state = shared_states[tl_id]
                action = agent.act(state)
                actions[tl_id] = action
            
            # Environment step
            next_states, rewards, done, info = env.step(actions)
            
            # Update rewards and losses
            for tl_id in env.traffic_signals:
                episode_rewards[tl_id] += rewards[tl_id]
                
            step += 1
            states = next_states
        
        # Calculate and log metrics
        metrics = _calculate_metrics(env, episode_rewards, episode_start_time, episode_losses)
        all_metrics['average_waiting_time'].append(metrics['average_waiting_time'])
        all_metrics['total_throughput'].append(metrics['total_throughput'])
        all_metrics['average_speed'].append(metrics['average_speed'])
        all_metrics['max_waiting_time'].append(metrics['max_waiting_time'])
        
        # Log episode metrics
        _log_episode_metrics(logger, episode, metrics)
    
    print("\n\nState Sharing Training Results - Averaged over all episodes:")
    print(f"Average Waiting Time: {np.mean(all_metrics['average_waiting_time']):.2f} ± {np.std(all_metrics['average_waiting_time']):.2f} seconds")
    print(f"Average Total Throughput: {np.mean(all_metrics['total_throughput']):.2f} ± {np.std(all_metrics['total_throughput']):.2f} vehicles")
    print(f"Average Speed: {np.mean(all_metrics['average_speed']):.2f} ± {np.std(all_metrics['average_speed']):.2f} km/h")
    print(f"Average Max Waiting Time: {np.mean(all_metrics['max_waiting_time']):.2f} ± {np.std(all_metrics['max_waiting_time']):.2f} seconds")
    
    return all_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train traffic signal control agents")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    results = train(args.config)
    logging.info("Training completed successfully")