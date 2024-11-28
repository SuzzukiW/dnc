# experiments/train/train_independent_agents.py

import os
import sys
import yaml
import torch
import numpy as np
import logging
import time
from datetime import datetime
from pathlib import Path
import traci
import argparse
from tqdm.auto import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.environment.multi_agent_sumo_env_independent import MultiAgentSumoIndependentEnvironment
from experiments.scenarios.communication.independent_agents import IndependentAgentsManager
from evaluation_sets.metrics import (
    average_waiting_time, 
    total_throughput, 
    average_speed, 
    max_waiting_time
)

def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def calculate_metrics(env, episode_reward, episode_start_time, loss, past_metrics):
    """
    Calculate performance metrics using evaluation sets metrics
    
    Args:
        env (MultiAgentSumoIndependentEnvironment): Training environment
        episode_reward (float): Total reward for the episode
        episode_start_time (float): Start time of the episode
        loss (float): Training loss
        past_metrics (list): Previous metrics for trend calculation
    
    Returns:
        dict: Comprehensive metrics for the episode
    """
    current_time = time.time()
    
    # Collect vehicle data for metrics calculation
    vehicle_data = [
        {
            'id': vid, 
            'waiting_time': traci.vehicle.getWaitingTime(vid),
            'speed': traci.vehicle.getSpeed(vid),
            'route': traci.vehicle.getRoute(vid)
        }
        for vid in traci.vehicle.getIDList()
    ]
    
    # Calculate metrics using evaluation sets functions
    metrics = {
        'avg_score': episode_reward,
        'episode_length': env.episode_step,
        'processing_time': (current_time - episode_start_time) / env.episode_step,
        'loss': loss if loss is not None else 0,
        
        # Metrics from evaluation sets
        'total_waiting_time': average_waiting_time(vehicle_data),
        'total_throughput': total_throughput(vehicle_data),
        'average_speed': average_speed(vehicle_data),
        'max_waiting_time': max_waiting_time(vehicle_data),
        
        # Additional derived metrics
        'vehicles_in_network': len(vehicle_data),
        'completed_trips': traci.simulation.getArrivedNumber()
    }
    
    # Calculate trends if past metrics exist
    if past_metrics:
        metrics['waiting_time_trend'] = (
            metrics['total_waiting_time'] - past_metrics[-1].get('total_waiting_time', 0)
        )
        metrics['throughput_trend'] = (
            metrics['total_throughput'] - past_metrics[-1].get('total_throughput', 0)
        )
        metrics['speed_trend'] = (
            metrics['average_speed'] - past_metrics[-1].get('average_speed', 0)
        )
    else:
        metrics['waiting_time_trend'] = 0
        metrics['throughput_trend'] = 0
        metrics['speed_trend'] = 0
    
    return metrics

def log_episode_metrics(logger, episode, total_episodes, episode_reward, epsilon, metrics):
    """
    Log detailed metrics after each episode with evaluation sets metrics
    
    Args:
        logger (logging.Logger): Logger instance
        episode (int): Current episode number
        total_episodes (int): Total number of episodes
        episode_reward (float): Total reward for the episode
        epsilon (float): Current exploration rate
        metrics (dict): Metrics dictionary
    """
    logger.info("=" * 80)
    logger.info(f"Episode {episode}/{total_episodes} [{episode/total_episodes*100:.1f}%]")
    logger.info("=" * 80)
    
    # Training Metrics
    logger.info("Training Metrics:")
    logger.info(f"Average Score:                  {metrics['avg_score']:.2f}")
    logger.info(f"Episode Length:                 {metrics['episode_length']:.1f} steps")
    logger.info(f"Processing Time:                {metrics['processing_time']:.2f} sec/step")
    
    # Learning Progress
    logger.info("Learning Progress:")
    logger.info(f"Loss:                          {metrics['loss']:.4f}")
    logger.info(f"Epsilon:                       {epsilon:.4f}")
    
    # Traffic Metrics from Evaluation Sets
    logger.info("Traffic Metrics:")
    logger.info(f"Total Waiting Time:            {metrics['total_waiting_time']:.2f} sec")
    logger.info(f"Waiting Time Trend:            {metrics['waiting_time_trend']:.4f} sec/episode")
    logger.info(f"Total Throughput:              {metrics['total_throughput']:.2f} vehicles/time")
    logger.info(f"Throughput Trend:              {metrics['throughput_trend']:.4f} veh/episode")
    logger.info(f"Average Speed:                 {metrics['average_speed']:.2f} m/s")
    logger.info(f"Speed Trend:                   {metrics['speed_trend']:.4f} m/s/episode")
    logger.info(f"Max Waiting Time:              {metrics['max_waiting_time']:.2f} sec")
    
    # Network Statistics
    logger.info("Network Statistics:")
    logger.info(f"Vehicles in Network:           {metrics['vehicles_in_network']}")
    logger.info(f"Completed Trips:               {metrics['completed_trips']}")
    
    logger.info("=" * 80)

def train(config):
    """Main training loop with complete metrics tracking"""
    # Setup devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup directories
    run_dir = os.path.join(config['paths']['output_dir'], f"run_{timestamp}")
    log_dir = os.path.join(run_dir, "logs")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info(f"Starting training with config: {config}")
    
    # Initialize environment
    env = MultiAgentSumoIndependentEnvironment(
        net_file=config['paths']['net_file'],
        route_file=config['paths']['route_file'],
        use_gui=config['environment']['use_gui'],
        num_seconds=config['environment']['num_seconds'],
        delta_time=config['environment']['delta_time'],
        yellow_time=config['environment']['yellow_time'],
        min_green=config['environment']['min_green'],
        max_green=config['environment']['max_green'],
        sumo_seed=config['environment']['sumo_seed'],
        sumo_warnings=config['environment']['sumo_warnings']
    )
    
    # Get traffic light IDs and action/state spaces
    traffic_light_ids = env.get_traffic_light_ids()
    state_size = env.observation_space_size
    action_size = env.action_space_size
    
    # Initialize agents
    agents = IndependentAgentsManager(
        traffic_light_ids=traffic_light_ids,
        state_size=state_size,
        action_size=action_size,
        device=device
    )
    
    # Training parameters
    total_episodes = config['training']['num_episodes']
    batch_size = config['training']['batch_size']
    epsilon_start = config['training']['epsilon_start']
    epsilon_end = config['training']['epsilon_end']
    epsilon_decay = config['training']['epsilon_decay']
    train_freq = config['training']['train_freq']
    save_freq = config['training']['save_freq']
    max_steps_per_episode = config['training'].get('max_steps_per_episode', float('inf'))
    
    # Create a progress bar for the entire training process
    progress_bar = tqdm(
        total=total_episodes, 
        desc="🚀 Training Independent Agents", 
        unit="episode", 
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        colour='green'
    )
    
    # Metrics tracking
    past_metrics = []
    best_average_reward = float('-inf')
    training_start_time = time.time()
    
    logger.info("Starting training...")
    
    for episode in range(total_episodes):
        states = env.reset()
        done = False
        episode_reward = 0
        episode_start_time = time.time()
        episode_length = 0
        episode_loss = 0
        total_steps = 0
        total_vehicles = 0
        total_waiting_time = 0
        total_queue_length = 0
        
        # Calculate epsilon for this episode
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * episode / epsilon_decay)
        
        while not done and episode_length < max_steps_per_episode:
            # Select actions for all agents
            actions = agents.select_actions(states, epsilon)
            
            # Execute actions in environment
            next_states, rewards, done, _ = env.step(actions)
            
            # Store transitions for all agents
            transitions = {
                tl_id: (
                    states[tl_id],
                    actions[tl_id],
                    rewards[tl_id],
                    next_states[tl_id],
                    done
                )
                for tl_id in traffic_light_ids
            }
            agents.store_transitions(transitions)
            
            # Train agents
            if total_steps % train_freq == 0:
                loss = agents.train_agents(batch_size)
                if loss is not None:
                    episode_loss = loss  # Store the latest loss for metrics
                    logger.info(f"Episode {episode}, Step {total_steps}, Loss: {loss:.4f}")
            
            # Update states and accumulate rewards
            states = next_states
            episode_reward += sum(rewards.values())
            
            # Collect step metrics
            total_steps += 1
            episode_length += 1
            for tl_id in traffic_light_ids:
                lanes = env.traffic_lights[tl_id]['lanes']
                for lane in lanes:
                    waiting_time = traci.lane.getWaitingTime(lane)
                    total_waiting_time += waiting_time
                    
                    queue = sum(1 for veh_id in traci.lane.getLastStepVehicleIDs(lane)
                              if traci.vehicle.getSpeed(veh_id) < 0.1)
                    total_queue_length += queue
                    
                    total_vehicles += traci.lane.getLastStepVehicleNumber(lane)
            
            if episode_length >= max_steps_per_episode:
                logger.info(f"Episode {episode} reached maximum steps ({max_steps_per_episode})")
                break
        
        # Calculate episode metrics
        metrics = calculate_metrics(env, episode_reward, episode_start_time, episode_loss, past_metrics)
        
        past_metrics.append(metrics)
        
        # Update progress bar with episode metrics
        progress_bar.set_postfix({
            'Reward': f'{episode_reward:.2f}', 
            'Avg Score': f'{metrics["avg_score"]:.2f}', 
            'Loss': f'{episode_loss:.4f}',
            'ε': f'{epsilon:.3f}'
        })
        progress_bar.update(1)
        
        # Optional: log episode metrics
        log_episode_metrics(logger, episode, total_episodes, episode_reward, epsilon, metrics)
        
        # Save checkpoints
        if (episode + 1) % save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode+1}")
            os.makedirs(checkpoint_path, exist_ok=True)
            agents.save_agents(checkpoint_path)
            logger.info(f"Saved checkpoint at episode {episode+1}")
        
        # Save best model
        current_avg_reward = np.mean([m['avg_score'] for m in past_metrics[-100:]])
        if current_avg_reward > best_average_reward:
            best_average_reward = current_avg_reward
            best_model_path = os.path.join(checkpoint_dir, "best_model")
            os.makedirs(best_model_path, exist_ok=True)
            agents.save_agents(best_model_path)
            logger.info(f"New best average reward: {best_average_reward:.2f}")
    
    # Close the progress bar
    progress_bar.close()
    
    # Log final training summary
    log_final_summary(logger, past_metrics)
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    agents.save_agents(final_path)
    logger.info("Training completed. Final model saved.")
    
    env.close()
    
    return past_metrics  # Return metrics for potential further analysis

def log_final_summary(logger, past_metrics):
    """Log final training summary with all metrics"""
    if not past_metrics:
        logger.warning("No metrics available for summary")
        return
        
    logger.info("=" * 80)
    logger.info("                            📊 Final Training Summary                            ")
    logger.info("=" * 80)
    
    # Time Statistics
    avg_episode_length = np.mean([m['episode_length'] for m in past_metrics])
    avg_processing_time = np.mean([m['processing_time'] for m in past_metrics])
    total_time = sum(m['episode_length'] * m['processing_time'] for m in past_metrics) / 3600
    
    logger.info("⏱️  Time Statistics:")
    logger.info(f"Total Episodes:                      {len(past_metrics)}")
    logger.info(f"Total Training Time:                 {total_time:.2f} hours")
    logger.info(f"Average Episode Length:              {avg_episode_length:.1f} steps")
    logger.info(f"Average Processing Time:             {avg_processing_time:.2f} sec/step")
    
    # Performance Statistics
    window_size = min(100, len(past_metrics))  # Adjust window size based on available episodes
    final_avg_score = np.mean([m['avg_score'] for m in past_metrics[-window_size:]])
    
    if len(past_metrics) >= window_size:
        windows = [past_metrics[i:i+window_size] for i in range(len(past_metrics) - window_size + 1)]
        best_avg_score = max(np.mean([m['avg_score'] for m in window]) for window in windows)
    else:
        best_avg_score = final_avg_score
    
    final_loss = past_metrics[-1]['loss']
    
    logger.info("🎯 Performance Statistics:")
    logger.info(f"Final Average Score:                {final_avg_score:.2f}")
    logger.info(f"Best Average Score:                 {best_avg_score:.2f}")
    logger.info(f"Final Loss:                        {final_loss:.4f}")
    
    # Traffic Statistics
    final_wait_time = np.mean([m['total_waiting_time'] for m in past_metrics[-window_size:]])
    best_wait_time = min(m['total_waiting_time'] for m in past_metrics)
    final_queue = np.mean([m['total_throughput'] for m in past_metrics[-window_size:]])
    min_queue = min(m['total_throughput'] for m in past_metrics)
    
    logger.info("🚦 Traffic Statistics:")
    logger.info(f"Final Average Wait Time:            {final_wait_time:.2f} seconds")
    logger.info(f"Best Wait Time:                     {best_wait_time:.2f} seconds")
    logger.info(f"Final Queue Length:                 {final_queue:.2f} vehicles")
    logger.info(f"Min Queue Length:                   {min_queue:.2f} vehicles")
    
    # Efficiency Metrics
    final_flow = np.mean([m['average_speed'] for m in past_metrics[-window_size:]])
    best_flow = max(m['average_speed'] for m in past_metrics)
    
    logger.info("⚡ Efficiency Metrics:")
    logger.info(f"Final Flow Rate:                    {final_flow:.2f} m/s")
    logger.info(f"Best Flow Rate:                     {best_flow:.2f} m/s")
    logger.info("=" * 80)

def main():
    """
    Main function to run independent agents training
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Independent Agents Traffic Signal Control Training')
    parser.add_argument('-c', '--config', 
                        type=str, 
                        default='config/independent_agents.yaml', 
                        help='Path to the configuration file')
    parser.add_argument('-l', '--log_level', 
                        type=str, 
                        default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}")
        sys.exit(1)
    
    # Log configuration file path
    logging.info(f"Using configuration file: {args.config}")
    
    try:
        # Run training
        train(config)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()