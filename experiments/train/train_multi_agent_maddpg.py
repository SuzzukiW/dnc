# experiments/train/train_multi_agent_maddpg.py

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
from typing import List, Dict
import traci

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.agents.maddpg_agent import MADDPGAgent
from src.environment.multi_agent_sumo_env_maddpg import MultiAgentSumoEnvironmentMADDPG as MultiAgentSumoEnvironment
from evaluation_sets.metrics import average_waiting_time, total_throughput, average_speed, max_waiting_time

def setup_logging(config: Dict):
    """Set up logging configuration.
    
    Args:
        config: Configuration dictionary containing logging settings
    """
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.WARNING,  
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.getLogger('traci').setLevel(logging.ERROR)
    logging.getLogger('sumolib').setLevel(logging.ERROR)
    
def create_agents(config_path: str, env: MultiAgentSumoEnvironment) -> List[MADDPGAgent]:
    """Create MADDPG agents.
    
    Args:
        config_path: Path to configuration file
        env: SUMO environment with observation and action spaces
        
    Returns:
        List of initialized MADDPG agents
    """
    # Find maximum action size across all agents
    max_action_size = max(space.shape[0] for space in env.action_spaces.values())
    
    agents = []
    for tl_id in env.traffic_lights:
        agent = MADDPGAgent(
            config_path=config_path,
            agent_id=tl_id,
            observation_space=env.observation_spaces[tl_id],
            action_space=env.action_spaces[tl_id],
            max_action_size=max_action_size
        )
        agents.append(agent)
    return agents

def evaluate_agents(env: MultiAgentSumoEnvironment, agents: List[MADDPGAgent],
                   num_episodes: int, config: Dict) -> Dict:
    """Evaluate agents' performance using standardized metrics.
    
    Args:
        env: SUMO environment
        agents: List of MADDPG agents
        num_episodes: Number of evaluation episodes
        config: Configuration dictionary
        
    Returns:
        Dictionary containing evaluation metrics
    """
    total_rewards = np.zeros(len(env.traffic_lights))
    total_metrics = {
        'avg_waiting_time': 0,
        'max_waiting_time': 0,
        'avg_speed': 0,
        'throughput': 0,
        'queue_length': 0
    }
    
    for episode in range(num_episodes):
        states = env.reset()
        episode_rewards = np.zeros(len(env.traffic_lights))
        
        vehicle_data = []  # List to store vehicle data for metrics calculation
        vehicles_seen = set()  # Track unique vehicles
        
        for step in range(config['training']['max_steps_per_episode']):
            # Get vehicle data for metrics
            vehicle_ids = traci.vehicle.getIDList()
            step_vehicle_data = []
            
            for vid in vehicle_ids:
                vehicles_seen.add(vid)  # Track unique vehicle
                vehicle_info = {
                    'waiting_time': traci.vehicle.getWaitingTime(vid),
                    'speed': traci.vehicle.getSpeed(vid),
                    'time_loss': traci.vehicle.getTimeLoss(vid)
                }
                step_vehicle_data.append(vehicle_info)
            
            vehicle_data.extend(step_vehicle_data)
            
            # Get actions and step environment
            actions_list = [agent.act(states[tl_id], add_noise=False) 
                          for tl_id, agent in zip(list(states.keys()), agents)]
            actions = {tl_id: action for tl_id, action in zip(list(states.keys()), actions_list)}
            next_states, rewards, dones, info = env.step(actions)
            
            for i, tl_id in enumerate(list(states.keys())):
                episode_rewards[i] += rewards[tl_id]
            
            if any(dones.values()):
                break
                
            states = next_states
        
        # Calculate metrics using the metrics module
        if vehicle_data:
            total_metrics['avg_waiting_time'] += average_waiting_time(vehicle_data)
            total_metrics['max_waiting_time'] += max_waiting_time(vehicle_data)
            total_metrics['avg_speed'] += average_speed(vehicle_data)
            total_metrics['throughput'] += len(vehicles_seen)  # Use count of unique vehicles
            
            # Calculate queue length (vehicles with speed < 0.1 m/s)
            queue_length = len([v for v in vehicle_data if v['speed'] < 0.1])
            total_metrics['queue_length'] += queue_length / len(vehicle_data)
        
        total_rewards += episode_rewards
    
    # Calculate averages
    avg_rewards = total_rewards / num_episodes
    for key in total_metrics:
        total_metrics[key] /= num_episodes
    
    return {
        'avg_rewards': avg_rewards,
        'mean_reward': np.mean(avg_rewards),
        **total_metrics
    }

def print_metrics_summary(metrics: Dict, episode: int = None, final: bool = False, config: Dict = None):
    """Print a summary of the metrics."""
    total_episodes = config['training']['num_episodes'] if config else 'N/A'
    
    if final:
        print("\nFinal Training Results:")
        print(f"Episodes completed: {total_episodes}")
        # Convert numpy arrays to float for proper formatting
        avg_waiting = float(np.mean(metrics['avg_waiting_time']))
        std_waiting = float(np.std(metrics['avg_waiting_time']))
        avg_throughput = float(np.mean(metrics['throughput']))
        std_throughput = float(np.std(metrics['throughput']))
        avg_speed = float(np.mean(metrics['avg_speed']))
        std_speed = float(np.std(metrics['avg_speed']))
        avg_max_waiting = float(np.mean(metrics['max_waiting_time']))
        std_max_waiting = float(np.std(metrics['max_waiting_time']))
        
        print(f"Average Waiting Time: {avg_waiting:.2f} ± {std_waiting:.2f} seconds")
        print(f"Average Throughput: {avg_throughput:.2f} ± {std_throughput:.2f} vehicles")
        print(f"Average Speed: {avg_speed:.2f} ± {std_speed:.2f} m/s")
        print(f"Average Max Waiting Time: {avg_max_waiting:.2f} ± {std_max_waiting:.2f} seconds")
    else:
        print(f"\nEpisode {episode}/{total_episodes} Results:")
        # Convert single episode metrics to float
        print(f"Average Waiting Time: {float(metrics['avg_waiting_time']):.2f} seconds")
        print(f"Throughput: {float(metrics['throughput']):.2f} vehicles")
        print(f"Average Speed: {float(metrics['avg_speed']):.2f} m/s")
        print(f"Max Waiting Time: {float(metrics['max_waiting_time']):.2f} seconds")

def train(config_path: str):
    """Main training loop for MADDPG agents.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with configuration: {config}")
    
    # Set random seeds
    np.random.seed(config['evaluation']['eval_seed'])
    torch.manual_seed(config['evaluation']['eval_seed'])
    
    # Initialize metrics storage
    all_episodes_metrics = {
        'avg_waiting_time': [],
        'max_waiting_time': [],
        'avg_speed': [],
        'throughput': [],
        'queue_length': [],
        'mean_reward': [],
        'avg_rewards': []
    }
    
    # Resolve SUMO network paths relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    net_file = os.path.join(project_root, config['sumo_network']['net_file'])
    
    # Handle backward compatibility for route files
    if 'route_files' in config['sumo_network']:
        route_file = config['sumo_network']['route_files']['train']
    else:
        route_file = config['sumo_network']['route_file']
    
    route_file = os.path.join(project_root, route_file)
    
    logger.info(f"Using SUMO network file: {net_file}")
    logger.info(f"Using SUMO route file: {route_file}")
    
    # Initialize environment
    env = MultiAgentSumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=config['environment']['use_gui'],
        num_seconds=config['environment']['num_seconds'],
        delta_time=config['environment']['delta_time'],
        yellow_time=config['environment']['yellow_time'],
        min_green=config['environment']['min_green'],
        max_green=config['environment']['max_green']
    )
    
    # Create agents after environment initialization
    agents = create_agents(config_path, env)
    
    # Set up TensorBoard
    if config['logging']['tensorboard']:
        writer = SummaryWriter(config['logging']['log_dir'])
    
    # Training loop
    best_eval_reward = float('-inf')
    pbar = tqdm(total=config['training']['num_episodes'],
                desc='Training Episodes',
                position=0,
                leave=True)
    
    for episode in range(config['training']['num_episodes']):
        states = env.reset()
        episode_rewards = np.zeros(len(env.traffic_lights))
        
        # Get list of traffic light IDs for consistent ordering
        tl_ids = list(states.keys())
        
        # Episode progress bar
        episode_pbar = tqdm(total=config['training']['max_steps_per_episode'],
                           desc=f'Episode {episode + 1} Steps',
                           position=1,
                           leave=False)
        
        for step in range(config['training']['max_steps_per_episode']):
            # Get actions from all agents
            actions_list = [agent.act(states[tl_id], add_noise=True)
                          for tl_id, agent in zip(tl_ids, agents)]
            
            actions = {tl_id: action for tl_id, action in zip(tl_ids, actions_list)}
            next_states, rewards, dones, info = env.step(actions)
            
            # Store experiences for each agent
            for i, (tl_id, agent) in enumerate(zip(tl_ids, agents)):
                other_agents_states = [states[other_id] for other_id in tl_ids if other_id != tl_id]
                other_agents_actions = [actions[other_id] for other_id in tl_ids if other_id != tl_id]
                other_agents_next_states = [next_states[other_id] for other_id in tl_ids if other_id != tl_id]
                
                agent.step(
                    states[tl_id], actions[tl_id], rewards[tl_id], next_states[tl_id], dones[tl_id],
                    other_agents_states, other_agents_actions, other_agents_next_states
                )
                
                episode_rewards[i] += rewards[tl_id]
            
            # Update metrics in progress bar
            if step % 10 == 0:  # Update every 10 steps to reduce overhead
                metrics = info.get('traffic_lights', {})
                avg_wait = np.mean([tl.get('total_waiting_time', 0) for tl in metrics.values()])
                avg_speed = np.mean([tl.get('avg_speed', 0) for tl in metrics.values()])
                episode_pbar.set_postfix({
                    'Avg Wait': f'{avg_wait:.1f}s',
                    'Avg Speed': f'{avg_speed:.1f}m/s',
                    'Vehicles': info.get('total_vehicles', 0)
                })
            
            episode_pbar.update(1)
            
            if any(dones.values()):
                break
            
            states = next_states
        
        episode_pbar.close()
        
        # Logging
        avg_reward = np.mean(episode_rewards)
        logger.info("=" * 80)
        logger.info(f"Episode {episode + 1}/{config['training']['num_episodes']}")
        logger.info("=" * 80)
        
        # Training Metrics
        logger.info("Training Metrics:")
        logger.info(f"Average Reward: {avg_reward:.2f}")
        
        # Evaluation
        if (episode + 1) % config['training']['evaluate_every'] == 0:
            eval_metrics = evaluate_agents(
                env, agents,
                config['evaluation']['eval_episodes'],
                config
            )
            
            # Print episode summary
            print_metrics_summary(eval_metrics, episode + 1, config=config)
            
            # Store metrics for final summary
            for key in eval_metrics:
                if key != 'avg_rewards':  # Skip the per-agent rewards array
                    all_episodes_metrics[key].append(eval_metrics[key])
            
            # TensorBoard logging
            if config['logging']['tensorboard']:
                for key, value in eval_metrics.items():
                    if key != 'avg_rewards':  # Skip the per-agent rewards array
                        writer.add_scalar(f'eval/{key}', value, episode)
            
            # Save best model
            if eval_metrics['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_metrics['mean_reward']
                logger.info(f"New best reward: {best_eval_reward:.2f}")
                for agent in agents:
                    agent.save(os.path.join(
                        config['logging']['model_dir'],
                        f'best_model_episode_{episode + 1}'
                    ))
        
        pbar.update(1)
        
        # Regular checkpoints
        if episode % config['logging']['save_frequency'] == 0:
            for agent in agents:
                agent.save(os.path.join(
                    config['logging']['model_dir'],
                    f'checkpoint_episode_{episode}'
                ))
    
    pbar.close()
    
    # Calculate and print final summary
    final_metrics = {
        'avg_waiting_time': np.array(all_episodes_metrics['avg_waiting_time']),
        'throughput': np.array(all_episodes_metrics['throughput']),
        'avg_speed': np.array(all_episodes_metrics['avg_speed']),
        'max_waiting_time': np.array(all_episodes_metrics['max_waiting_time'])
    }
    print_metrics_summary(final_metrics, final=True, config=config)
    
    # Close environment
    env.close()
    
    if config['logging']['tensorboard']:
        writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train MADDPG agents')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    args = parser.parse_args()
    
    train(args.config)

if __name__ == "__main__":
    main()