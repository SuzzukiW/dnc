# experiments/train_single_agent.py

import os
import sys
import yaml
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import json
import traci
import sumolib

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Add SUMO_HOME to path if not already there
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise ValueError("Please declare SUMO_HOME environment variable")

from src.agents import DQNAgent
from src.environment import SingleAgentSumoEnvironment
from src.utils.logger import setup_logger

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

def train_dqn(env_config, agent_config, num_episodes=20):
    """Train DQN agent on SUMO environment"""
    
    # Select strategic traffic light
    print("\nSelecting strategic traffic light...")
    selected_tl, tl_info = select_strategic_traffic_light(env_config['net_file'])
    print(f"\nSelected traffic light: {selected_tl}")
    print("Traffic light info:")
    for key, value in tl_info.items():
        print(f"  {key}: {value}")
    
    # Initialize environment with selected traffic light
    env = SingleAgentSumoEnvironment(
        net_file=env_config['net_file'],
        route_file=env_config['route_file'],
        out_csv_name=env_config['out_csv_name'],
        use_gui=env_config['use_gui'],
        num_seconds=env_config['num_seconds'],
        delta_time=env_config['delta_time'],
        yellow_time=env_config.get('yellow_time', 2),
        min_green=env_config.get('min_green', 8),
        max_green=env_config.get('max_green', 30),
        tl_id=selected_tl
    )
    
    # Initialize agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, agent_config)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path("experiments/logs") / f"single_agent_{selected_tl}_{timestamp}"
    model_dir = Path("experiments/models") / f"single_agent_{selected_tl}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("dqn_training", log_dir / "training.log")
    
    # Save configurations and traffic light info
    with open(log_dir / "env_config.yaml", 'w') as f:
        yaml.dump(env_config, f)
    with open(log_dir / "agent_config.yaml", 'w') as f:
        yaml.dump(agent_config, f)
    with open(log_dir / "traffic_light_info.yaml", 'w') as f:
        yaml.dump(tl_info, f)
    
    # Training metrics
    metrics = {
        'traffic_light_info': tl_info,
        'episode_rewards': [],
        'episode_lengths': [],
        'waiting_times': [],
        'queue_lengths': [],
        'throughput': [],
        'losses': [],
        'traffic_pressure': []
    }
    
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_losses = []
            step = 0
            done = False
            
            while not done:
                # Select and take action
                action = agent.act(state)
                next_state, reward, done, _, info = env.step(action)
                
                # Store transition
                agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                if len(agent.memory) > agent.batch_size:
                    loss = agent.replay()
                    if loss is not None:
                        episode_losses.append(float(loss))
                
                state = next_state
                episode_reward += reward
                step += 1
                
                # Update target network
                if step % agent_config.get('target_update_frequency', 50) == 0:
                    agent.update_target_network()
                
                # Adjust epsilon based on performance
                if episode > 0 and step % 100 == 0:
                    if metrics['waiting_times'][-1] > np.mean(metrics['waiting_times'][-5:]):
                        agent.epsilon = min(1.0, agent.epsilon * 1.1)
            
            # Update metrics
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_lengths'].append(step)
            env_metrics = env.get_metrics()
            metrics['waiting_times'].append(np.mean(env_metrics['waiting_times']))
            metrics['queue_lengths'].append(np.mean(env_metrics['queue_lengths']))
            metrics['throughput'].append(np.mean(env_metrics['throughput']))
            metrics['traffic_pressure'].append(np.mean(env_metrics['traffic_pressure']))
            metrics['losses'].append(np.mean(episode_losses) if episode_losses else 0)
            
            # Log progress
            logger.info(f"\nEpisode {episode+1}/{num_episodes}")
            logger.info(f"Reward: {episode_reward:.2f}")
            logger.info(f"Average Loss: {metrics['losses'][-1]:.4f}")
            logger.info(f"Epsilon: {agent.epsilon:.4f}")
            logger.info(f"Average Waiting Time: {metrics['waiting_times'][-1]:.2f}")
            logger.info(f"Average Queue Length: {metrics['queue_lengths'][-1]:.2f}")
            logger.info(f"Average Throughput: {metrics['throughput'][-1]:.2f}")
            
            # Save checkpoint
            if (episode + 1) % 2 == 0:
                checkpoint_dir = model_dir / f"checkpoint_{episode+1}"
                checkpoint_dir.mkdir(exist_ok=True)
                
                agent.save(checkpoint_dir / "model.pt")
                with open(checkpoint_dir / "metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=4)
            
            # Early stopping check
            if episode >= 4:
                recent_rewards = metrics['episode_rewards'][-5:]
                reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
                avg_waiting = np.mean(metrics['waiting_times'][-5:])
                avg_queue = np.mean(metrics['queue_lengths'][-5:])
                
                if (reward_trend > 0.005 and 
                    avg_waiting < 180 and 
                    avg_queue < 8 and
                    episode > num_episodes // 3):
                    logger.info("Performance targets achieved, stopping early")
                    break
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.exception("Training error details:")
    finally:
        # Save final model and metrics
        try:
            agent.save(model_dir / "final_model.pt")
            with open(log_dir / "metrics_final.json", 'w') as f:
                json.dump(metrics, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving final state: {str(e)}")
        finally:
            env.close()
    
    return metrics

def main():
    """Main training function"""
    # Load base configurations
    env_config = load_config(project_root / 'config/env_config.yaml')
    agent_config = load_config(project_root / 'config/agent_config.yaml')
    
    # Update environment configuration
    env_config.update({
        'yellow_time': 2,          # Shorter yellow time
        'min_green': 8,           # More flexible timing
        'max_green': 30,          # Shorter maximum green
        'num_seconds': 1500,      # 25 minutes per episode
        'delta_time': 5,          # More frequent decisions
        'use_gui': False,
        'max_depart_delay': 100,  # Shorter delay
        'time_to_teleport': 100   # Quicker teleport
    })
    
    # Update agent configuration
    agent_config.update({
        # Memory and batch settings
        'memory_size': 100000,    # Smaller memory for single agent
        'batch_size': 64,         # Smaller batches
        'prioritized_replay': True,
        'priority_alpha': 0.6,
        'priority_beta': 0.4,
        
        # Learning parameters
        'learning_rate': 0.0001,  # Lower learning rate
        'learning_rate_decay': 0.999,
        'min_learning_rate': 0.00001,
        'epsilon_start': 1.0,
        'epsilon_min': 0.05,
        'epsilon_decay': 0.95,    # Slower decay
        'gamma': 0.95,           # Shorter horizon
        
        # Network architecture
        'hidden_size': 256,       # Smaller network
        'num_hidden_layers': 2,   # Fewer layers
        'activation': 'relu',
        'dropout_rate': 0.1,
        
        # Training stability
        'target_update_frequency': 50,
        'gradient_clip': 0.5,     # More clipping
        'double_dqn': True,
        'dueling_dqn': True
    })
    
    print("\nStarting single-agent traffic light control training...")
    print("\nEnvironment Configuration:")
    for key, value in env_config.items():
        if key not in ['net_file', 'route_file', 'out_csv_name']:
            print(f"- {key}: {value}")
    
    print("\nAgent Configuration:")
    for key, value in agent_config.items():
        print(f"- {key}: {value}")
    
    try:
        print("\nStarting training...")
        metrics = train_dqn(env_config, agent_config, num_episodes=20)
        
        # Print results
        print("\nTraining completed!")
        if metrics:
            print("\nFinal Metrics Summary:")
            # Print traffic light info
            print("\nTraffic Light Information:")
            for key, value in metrics['traffic_light_info'].items():
                print(f"- {key}: {value}")
            
            print("\nPerformance Metrics:")
            print(f"Average reward per episode: {np.mean(metrics['episode_rewards']):.4f}")
            print(f"Best episode reward: {max(metrics['episode_rewards']):.4f}")
            print(f"Final average waiting time: {metrics['waiting_times'][-1]:.2f} seconds")
            print(f"Best waiting time: {min(metrics['waiting_times']):.2f} seconds")
            print(f"Average queue length: {np.mean(metrics['queue_lengths']):.2f} vehicles")
            if metrics.get('traffic_pressure'):
                print(f"Average traffic pressure: {np.mean(metrics['traffic_pressure']):.3f}")
            if metrics.get('throughput'):
                print(f"Average throughput: {np.mean(metrics['throughput']):.2f} vehicles")
            
            # Calculate improvements if we have at least 2 episodes
            if len(metrics['waiting_times']) > 1:
                waiting_time_improvement = ((metrics['waiting_times'][0] - metrics['waiting_times'][-1]) / 
                                         metrics['waiting_times'][0] * 100)
                queue_improvement = ((metrics['queue_lengths'][0] - metrics['queue_lengths'][-1]) / 
                                   metrics['queue_lengths'][0] * 100)
                
                print("\nPerformance Improvements:")
                print(f"Waiting time reduction: {waiting_time_improvement:.1f}%")
                print(f"Queue length reduction: {queue_improvement:.1f}%")
            
            # Print training statistics
            print("\nTraining Statistics:")
            print(f"Total episodes completed: {len(metrics['episode_rewards'])}")
            print(f"Average episode length: {np.mean(metrics['episode_lengths']):.0f} steps")
            if metrics['losses']:
                print(f"Final training loss: {metrics['losses'][-1]:.6f}")
            
            # Print early stopping info if training stopped early
            if len(metrics['episode_rewards']) < 20:
                print("\nNote: Training stopped early due to achieving performance targets")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        print("\nTraining session completed")

if __name__ == "__main__":
    main()