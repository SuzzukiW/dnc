# quick_tests/test_single_agent.py

import os
import sys
from pathlib import Path

# Add project root and quick_tests to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))
sys.path.append(str(current_dir))

import torch
import numpy as np
import sumolib
from collections import defaultdict
import traci

from src.agents import DQNAgent
from src.environment import MultiAgentSumoEnvironment
from src.utils.logger import setup_logger
from config_loader import load_config

def normalize_reward(reward, scale=100.0):
    """Normalize reward to help with training stability"""
    return reward / scale


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
        
        # Calculate distance from center (normalized by network size)
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        normalized_distance = distance_from_center / (np.sqrt(net_width**2 + net_height**2) / 2)
        
        # Calculate traffic density
        total_vehicles = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in controlled_lanes])
        
        print(f"  Lanes: {num_lanes}")
        print(f"  Approaches: {num_approaches}")
        print(f"  Position: ({x:.2f}, {y:.2f})")
        print(f"  Distance from center: {distance_from_center:.2f}m")
        print(f"  Current vehicles: {total_vehicles}")
        
        # Calculate score (higher is better)
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

def test_single_agent_training(env_config, agent_config, num_episodes=10):
    """Run a quick test training for single traffic light"""
    
    # Select strategic traffic light
    selected_tl, tl_info = select_strategic_traffic_light(env_config['net_file'])
    print(f"\nSelected Traffic Light Info:")
    print(f"ID: {selected_tl}")
    print(f"Number of controlled lanes: {tl_info['num_lanes']}")
    print(f"Number of approaches: {tl_info['num_approaches']}")
    print(f"Position: ({tl_info['position'][0]:.2f}, {tl_info['position'][1]:.2f})")
    print(f"Distance from network center: {tl_info['distance_from_center']:.2f}m")
    print("-" * 50)
    
    # Initialize environment
    env = MultiAgentSumoEnvironment(**env_config)
    
    # Initialize agent
    state_size = env.observation_spaces[selected_tl].shape[0]
    action_size = env.action_spaces[selected_tl].n
    agent = DQNAgent(state_size, action_size, agent_config)
    
    # Setup logging
    log_dir = current_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    logger = setup_logger("single_agent_test", log_dir / "single_agent_test.log")
    
    # Track metrics
    metrics = defaultdict(list)
    
    # Training loop
    for episode in range(num_episodes):
        states, _ = env.reset()
        state = states[selected_tl]
        episode_reward = 0
        episode_losses = []
        episode_queues = []
        episode_waiting_times = []
        done = False
        step = 0
        
        while not done:
            # Select action
            action = agent.act(state)
            actions = {selected_tl: action}
            
            # Take step in environment
            next_states, rewards, done, _, info = env.step(actions)
            next_state = next_states[selected_tl]
            reward = rewards[selected_tl]
            
            # Normalize and clip reward
            reward = normalize_reward(reward)
            reward = np.clip(reward, -10, 10)
            
            # Store transition and train agent
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > agent.batch_size:
                agent.optimizer.zero_grad()
                loss = agent.replay()
                
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), max_norm=1.0)
                
                episode_losses.append(loss)
                if step % 20 == 0:
                    logger.info(f"Episode {episode+1}, Step {step}, Loss: {loss:.4f}")
            
            # Collect metrics
            controlled_lanes = traci.trafficlight.getControlledLanes(selected_tl)
            total_queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes)
            total_waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in controlled_lanes)
            
            episode_queues.append(total_queue)
            episode_waiting_times.append(total_waiting_time)
            
            state = next_state
            episode_reward += reward
            step += 1
            
            # Update target network less frequently
            if step % 200 == 0:
                agent.update_target_network()
        
        # Store episode metrics
        metrics['rewards'].append(episode_reward)
        metrics['losses'].append(np.mean(episode_losses) if episode_losses else 0)
        metrics['avg_queue_length'].append(np.mean(episode_queues))
        metrics['avg_waiting_time'].append(np.mean(episode_waiting_times))
        
        # Log episode results
        logger.info(f"\nEpisode {episode+1}/{num_episodes}")
        logger.info(f"Total Reward: {episode_reward:.2f}")
        logger.info(f"Average Reward: {episode_reward/step:.2f}")
        logger.info(f"Average Loss: {metrics['losses'][-1]:.4f}")
        logger.info(f"Average Queue Length: {metrics['avg_queue_length'][-1]:.2f}")
        logger.info(f"Average Waiting Time: {metrics['avg_waiting_time'][-1]:.2f}")
        logger.info(f"Epsilon: {agent.epsilon:.4f}")
        logger.info("-" * 50)
        
        # Early stopping if we achieve good performance
        if episode > 2:
            recent_rewards = metrics['rewards'][-3:]
            recent_waiting_times = metrics['avg_waiting_time'][-3:]
            if np.mean(recent_rewards) > -100 and np.mean(recent_waiting_times) < 30:
                logger.info("Achieved good performance, stopping early")
                break
    
    env.close()
    
    # Print final training summary
    print("\nTraining Summary:")
    print(f"Average reward over all episodes: {np.mean(metrics['rewards']):.2f}")
    print(f"Best episode reward: {max(metrics['rewards']):.2f}")
    print(f"Final average loss: {np.mean(metrics['losses'][-5:]):.2f}")
    print(f"Final average queue length: {np.mean(metrics['avg_queue_length'][-5:]):.2f}")
    print(f"Final average waiting time: {np.mean(metrics['avg_waiting_time'][-5:]):.2f}")
    
    return agent, metrics, selected_tl

if __name__ == "__main__":
    # Load configurations from quick_tests/configs
    config_dir = current_dir / "configs"
    env_config = load_config(config_dir / "env_config.yaml")
    agent_config = load_config(config_dir / "agent_config.yaml")
    
    # Make paths absolute
    env_config['net_file'] = str(project_root / env_config['net_file'])
    env_config['route_file'] = str(project_root / env_config['route_file'])
    env_config['out_csv_name'] = str(project_root / env_config['out_csv_name'])
    
    # Create output directory
    out_dir = current_dir / "outputs"
    out_dir.mkdir(exist_ok=True)
    
    # Run test training
    agent, metrics, selected_tl = test_single_agent_training(env_config, agent_config)
    
    # Save model and metrics
    model_path = out_dir / f"model_{selected_tl}.pt"
    metrics_path = out_dir / f"metrics_{selected_tl}.json"
    
    # Save model
    agent.save(model_path)
    
    # Save metrics
    import json
    with open(metrics_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in metrics.items()}, f, indent=4)
    
    print("\nTest training completed!")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")