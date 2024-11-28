# experiments/train/train_single_agent.py

import os
import sys
import time
import json
import yaml
import torch
import logging
import numpy as np
from pathlib import Path
import traceback
from datetime import datetime
import traci
import sumolib
import warnings
import argparse
from evaluation_sets.metrics import average_waiting_time, total_throughput, average_speed, max_waiting_time
from tqdm.auto import tqdm, trange

# Suppress all warnings
warnings.filterwarnings('ignore')

# Go up two levels to get to the project root (from experiments/train to project root)
project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, str(project_root))

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = '/opt/homebrew/opt/sumo/share/sumo'
    os.environ['SUMO_HOME_SET'] = 'true'  # this should prevent retry messages

from src.agents import DQNAgent
from src.environment import SingleAgentSumoEnvironment
from src.utils.logger import get_logger

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_tl_position(net, tl_id):
    """Get traffic light position from its controlled lanes"""
    try:
        # Get controlled lanes
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        if not controlled_lanes:
            return None, None
        
        # Get the first controlled lane
        lane_id = controlled_lanes[0]
        try:
            # Get the edge from the lane ID
            edge = net.getLane(lane_id).getEdge()
            # Use the center of the edge as the traffic light position
            pos = edge.getCenter()
            return pos[0], pos[1]
        except:
            # If we can't get the position from the lane, try getting it from the junction
            try:
                junctions = net.getNode(tl_id)
                if junctions:
                    pos = junctions.getCoord()
                    return pos[0], pos[1]
            except:
                pass
            
            return None, None
    except Exception as e:
        print(f"Error getting position for traffic light {tl_id}: {e}")
        return None, None

def select_strategic_traffic_light(net_file):
    """Select a strategic traffic light from the network"""
    # Read SUMO network
    net = sumolib.net.readNet(net_file)
    
    # Start SUMO temporarily to analyze traffic lights
    sumo_cmd = [
        "sumo", 
        "-n", net_file,
        "--no-warnings",
        "--no-step-log",
    ]
    traci.start(sumo_cmd)
    
    # Get all traffic lights
    traffic_lights = traci.trafficlight.getIDList()
    print(f"\nFound {len(traffic_lights)} traffic lights")
    
    # Get network bounds for normalization
    net_bounds = net.getBoundary()
    net_width = net_bounds[2] - net_bounds[0]
    net_height = net_bounds[3] - net_bounds[1]
    center_x = (net_bounds[0] + net_bounds[2]) / 2
    center_y = (net_bounds[1] + net_bounds[3]) / 2
    
    print(f"Network center: ({center_x:.2f}, {center_y:.2f})")
    print(f"Network size: {net_width:.2f}m x {net_height:.2f}m")
    
    # Analyze each traffic light
    tl_scores = {}
    for tl_id in traffic_lights:
        print(f"\nAnalyzing traffic light: {tl_id}")
        
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        controlled_links = traci.trafficlight.getControlledLinks(tl_id)
        
        # Get position
        x, y = get_tl_position(net, tl_id)
        if x is None:
            print(f"Skipping {tl_id} - couldn't determine position")
            continue
            
        # Calculate metrics
        num_lanes = len(controlled_lanes)
        num_approaches = len(set([lane.split('_')[0] for lane in controlled_lanes]))
        
        # Calculate distance from center
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        normalized_distance = distance_from_center / (np.sqrt(net_width**2 + net_height**2) / 2)
        
        # Calculate traffic density
        total_vehicles = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in controlled_lanes])
        
        print(f"  Lanes: {num_lanes}")
        print(f"  Approaches: {num_approaches}")
        print(f"  Position: ({x:.2f}, {y:.2f})")
        print(f"  Distance from center: {distance_from_center:.2f}m")
        print(f"  Current vehicles: {total_vehicles}")
        
        # Calculate score
        score = (
            num_lanes * 2 +                     # More lanes = more important
            num_approaches * 3 +                # More approaches = more important
            (1 - normalized_distance) * 4 +     # Closer to center = more important
            total_vehicles * 0.5                # More traffic = more important
        )
        
        print(f"  Score: {score:.2f}")
        
        tl_scores[tl_id] = {
            'score': score,
            'num_lanes': num_lanes,
            'num_approaches': num_approaches,
            'position': (x, y),
            'distance_from_center': distance_from_center,
            'total_vehicles': total_vehicles
        }
    
    traci.close()
    
    if not tl_scores:
        raise ValueError("No valid traffic lights found in the network")
    
    # Get top 5 traffic lights by score
    top_5 = sorted(tl_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:5]
    
    print("\nTop 5 Traffic Lights:")
    for i, (tl_id, info) in enumerate(top_5, 1):
        print(f"\n{i}. Traffic Light {tl_id}")
        print(f"   Score: {info['score']:.2f}")
        print(f"   Lanes: {info['num_lanes']}")
        print(f"   Approaches: {info['num_approaches']}")
        print(f"   Vehicles: {info['total_vehicles']}")
        print(f"   Distance from center: {info['distance_from_center']:.2f}m")
    
    # Select traffic light with highest score
    selected_tl = top_5[0][0]
    return selected_tl, tl_scores[selected_tl]

def train_dqn(config):
    """Train DQN agent on SUMO environment"""
    env_config = config['env']
    agent_config = config['agent']
    training_config = config['training']
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Suppress SUMO output
    traci.setLegacyGetLeader(True)  # Suppress some SUMO warnings
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    
    # Set SUMO environment variables to suppress output
    os.environ['SUMO_WARNINGS'] = "false"
    os.environ['SUMO_QUIET'] = "true"
    
    # Initialize environment
    env = SingleAgentSumoEnvironment(
        net_file=env_config['net_file'],
        route_file=env_config['route_file'],
        out_csv_name=f"outputs/single_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        use_gui=env_config['use_gui'],
        num_seconds=env_config['episode_length'],
        delta_time=env_config['delta_time'],
        yellow_time=env_config['yellow_time'],
        min_green=env_config['min_green_time'],
        max_green=env_config['max_green_time'],
        sumo_warnings=False  # Disable SUMO warnings
    )
    
    # Initialize agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        config=agent_config
    )
    
    logger = get_logger('single_agent_training')
    best_reward = float('-inf')
    patience_counter = 0
    
    # Lists to track metrics across all episodes
    episode_metrics = []
    
    # Create progress bar for episodes
    episode_iterator = trange(training_config['num_episodes'], desc='Training Episodes', unit='episode')
    
    for episode in episode_iterator:
        state = env.reset()[0]  # env.reset() returns (state, info)
        total_reward = 0
        episode_losses = []
        
        # Create progress bar for steps
        step_iterator = trange(training_config['max_steps'], 
                             desc=f'Episode {episode + 1} Steps', 
                             unit='step',
                             leave=False)
        
        # Track episode-specific metrics
        episode_info = {
            'waiting_times': [],
            'queue_lengths': [],
            'speeds': [],
            'vehicles_completed': 0,
            'total_vehicles': 0
        }
        
        for step in step_iterator:
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store metrics from info dictionary
            if info:
                waiting_time = info.get('average_waiting_time', 0)
                queue_length = info.get('queue_length', 0)
                avg_speed = info.get('average_speed', 0)
                vehicles_completed = info.get('vehicles_completed', 0)
                total_vehicles = info.get('total_vehicles', 0)
                
                # Update episode metrics
                episode_info['waiting_times'].append(waiting_time)
                episode_info['queue_lengths'].append(queue_length)
                episode_info['speeds'].append(avg_speed)
                episode_info['vehicles_completed'] += vehicles_completed
                episode_info['total_vehicles'] = max(episode_info['total_vehicles'], total_vehicles)
                
                # Update step progress bar with current metrics
                step_iterator.set_postfix({
                    'Reward': f'{reward:.2f}',
                    'Waiting': f'{waiting_time:.1f}s',
                    'Speed': f'{avg_speed * 3.6:.1f}km/h',
                    'Vehicles': total_vehicles
                })
            
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            
            if done or truncated:
                break
        
        # Calculate episode metrics
        avg_waiting_time = np.mean(episode_info['waiting_times']) if episode_info['waiting_times'] else 0
        avg_queue_length = np.mean(episode_info['queue_lengths']) if episode_info['queue_lengths'] else 0
        avg_speed = np.mean(episode_info['speeds']) if episode_info['speeds'] else 0
        total_completed = episode_info['vehicles_completed']
        max_vehicles = episode_info['total_vehicles']
        
        # Store episode metrics for final summary
        episode_metrics.append({
            'avg_waiting_time': avg_waiting_time,
            'avg_queue_length': avg_queue_length,
            'avg_speed': avg_speed,
            'throughput': total_completed,
            'total_vehicles': max_vehicles
        })
        
        # Log episode results
        logger.info(f"Episode {episode + 1}/{training_config['num_episodes']}, "
                   f"Total Reward: {total_reward:.2f}, "
                   f"Average Loss: {np.mean(episode_losses):.4f}, "
                   f"Epsilon: {agent.epsilon:.3f}")
        
        print("\nTraffic Metrics:")
        print(f"  - Avg Waiting Time: {avg_waiting_time:.2f}s")
        print(f"  - Queue Length: {avg_queue_length:.0f} vehicles")
        print(f"  - Throughput: {total_completed} vehicles")
        print(f"  - Avg Speed: {avg_speed * 3.6:.2f} km/h")  # Convert m/s to km/h
        print(f"  - Total Vehicles Seen: {max_vehicles}")
        
        # Save model periodically
        if (episode + 1) % training_config['save_frequency'] == 0:
            agent.save(f"models/single_agent_episode_{episode + 1}.pth")
        
        # Early stopping
        if training_config['early_stopping']:
            if total_reward > best_reward + training_config['min_improvement']:
                best_reward = total_reward
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= training_config['patience']:
                logger.info("Early stopping triggered")
                break
    
    # Calculate and log final summary metrics
    if episode_metrics:
        avg_waiting_times = [m['avg_waiting_time'] for m in episode_metrics]
        avg_queue_lengths = [m['avg_queue_length'] for m in episode_metrics]
        avg_speeds = [m['avg_speed'] for m in episode_metrics]
        throughputs = [m['throughput'] for m in episode_metrics]
        
        logger.info("\nTraining Complete - Final Traffic Metrics Summary:")
        logger.info(f"Average Waiting Time: {np.mean(avg_waiting_times):.2f}s (±{np.std(avg_waiting_times):.2f})")
        logger.info(f"Average Queue Length: {np.mean(avg_queue_lengths):.1f} vehicles (±{np.std(avg_queue_lengths):.1f})")
        logger.info(f"Average Throughput: {np.mean(throughputs):.1f} vehicles (±{np.std(throughputs):.1f})")
        logger.info(f"Average Speed: {np.mean(avg_speeds) * 3.6:.2f} km/h (±{np.std(avg_speeds) * 3.6:.2f})")
    
    env.close()
    return agent

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a single agent for traffic signal control')
    parser.add_argument('--config', type=str, default='config/single_agent_config.yaml',
                      help='Path to configuration file')
    return parser.parse_args()

def main():
    """Main function"""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Load configuration
        config = load_config(args.config)
        
        # Set up logging
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        logger = get_logger('single_agent_training')
        
        logger.info(f"Loading configuration from: {args.config}")
        
        # Train the agent
        agent = train_dqn(config)
        
        # Save the final model
        os.makedirs("models", exist_ok=True)
        model_path = "models/single_agent_final.pth"
        agent.save(model_path)
        logger.info(f"Final model saved to: {model_path}")
        
    except Exception as e:
        logger = get_logger('single_agent_training')
        logger.error(f"Training failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()