#!/usr/bin/env python3
# experiments/run_independent_agents.py

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
print(f"Project root set to: {project_root}")

try:
    print("Importing local modules...")
    from src.environment import MultiAgentSumoEnvironment
    from experiments.scenarios.communication.independent_agents import IndependentAgentManager
    from src.utils import setup_logger
    print("Local modules imported successfully")
except Exception as e:
    print(f"Error importing local modules: {e}")
    raise

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
    print(f"Attempting to load config from: {config_path}")
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Config file not found at: {config_path.absolute()}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        print("Config loaded successfully")
        return config

def filter_valid_traffic_lights(env):
    """Filter and validate traffic lights that have proper phase definitions"""
    print("Starting traffic light validation...")
    all_traffic_lights = env.traffic_lights
    valid_traffic_lights = []
    skipped_traffic_lights = []
    
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
            print(f"Error validating traffic light {tl_id}: {e}")
            skipped_traffic_lights.append((tl_id, str(e)))
            continue
    
    print(f"\nTraffic Light Validation Summary:")
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
    """Calculate rewards for all traffic lights"""
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
        
        # Collect per-lane metrics
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

        # Calculate green utilization
        green_utilization = sum(1 for l in green_lanes if lane_metrics[l]['vehicles'] > 0) / max(len(green_lanes), 1)

        # Calculate metrics
        avg_metrics = {
            'waiting_time': np.mean([m['waiting_time'] for m in lane_metrics.values()]),
            'queue_length': np.mean([m['queue'] for m in lane_metrics.values()]),
            'speed': np.mean([m['relative_speed'] for m in lane_metrics.values()]),
            'pressure': np.mean([m['pressure'] for m in lane_metrics.values()])
        }

        # Calculate reward components
        reward_components = {
            'waiting': -avg_metrics['waiting_time'] / MAX_WAITING_TIME,
            'queue': -avg_metrics['queue_length'] / MAX_QUEUE_LENGTH,
            'speed': avg_metrics['speed'],
            'green_util': max(0, green_utilization - MIN_GREEN_UTIL) * 2,
            'pressure': -avg_metrics['pressure'] 
        }

        # Calculate weighted reward
        reward = sum(0.25 * component for component in reward_components.values())
        rewards[tl_id] = float(np.clip(reward, -1.0, 1.0))

    return rewards

def plot_metrics(metrics, save_path):
    """Plot and save training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot average rewards
    axes[0, 0].plot(metrics['mean_rewards'])
    axes[0, 0].set_title('Mean Reward per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    
    # Plot global rewards
    axes[0, 1].plot(metrics['global_rewards'])
    axes[0, 1].set_title('Global Reward per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Global Reward')
    
    # Plot average waiting times
    axes[1, 0].plot(metrics['mean_waiting_times'])
    axes[1, 0].set_title('Mean Waiting Time per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Waiting Time (s)')
    
    # Plot average queue lengths
    axes[1, 1].plot(metrics['mean_queue_lengths'])
    axes[1, 1].set_title('Mean Queue Length per Episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Queue Length')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Part II: Main execution

def train_independent_agents(env_config, agent_config, num_episodes=10):
    """Train multiple independent DQN agents"""
    
    # Ensure GUI is disabled for training
    env_config['use_gui'] = False
    
    # Initialize environment
    env = MultiAgentSumoEnvironment(**env_config)
    
    # Filter valid traffic lights
    valid_traffic_lights = filter_valid_traffic_lights(env)
    print(f"\nProceeding with training using {len(valid_traffic_lights)} valid traffic lights")
    
    # Update environment's traffic lights
    env.traffic_lights = valid_traffic_lights
    
    # Initialize agent manager
    state_size = env.observation_spaces[valid_traffic_lights[0]].shape[0]
    action_size = env.action_spaces[valid_traffic_lights[0]].n
    
    manager = IndependentAgentManager(
        state_size=state_size,
        action_size=action_size,
        num_agents=len(valid_traffic_lights),
        config=agent_config
    )
    
    print("\nAgent Configuration:")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path("experiments/logs") / f"independent_{timestamp}"
    model_dir = Path("experiments/models") / f"independent_{timestamp}"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "final_models").mkdir(exist_ok=True)
    
    logger = setup_logger("independent_training", log_dir / "training.log")
    
    # Initialize metrics
    metrics = {
        'mean_rewards': [],
        'global_rewards': [],
        'mean_waiting_times': [],
        'mean_queue_lengths': [],
        'throughput': [],
        'agent_rewards': defaultdict(list),
        'losses': defaultdict(list)
    }
    
    try:
        for episode in range(num_episodes):
            try:
                states, _ = env.reset()
                states = {tl: state for tl, state in states.items() 
                         if tl in valid_traffic_lights}
                
                episode_rewards = defaultdict(float)
                episode_losses = defaultdict(list)
                episode_metrics = defaultdict(list)
                done = False
                step = 0
                
                while not done:
                    try:
                        # Get actions for each agent
                        actions = manager.select_actions(states)
                        
                        # Execute actions
                        next_states, rewards, done, _, info = env.step(actions)
                        
                        # Calculate rewards
                        rewards = calculate_rewards(env, valid_traffic_lights)
                        
                        # Store transitions and update agents
                        manager.store_transitions(states, actions, rewards, next_states,
                                               {tl: done for tl in valid_traffic_lights})
                        
                        losses = manager.update_agents()
                        
                        # Update metrics
                        for tl_id in valid_traffic_lights:
                            # Track rewards
                            episode_rewards[tl_id] += rewards.get(tl_id, 0)
                            
                            # Track losses
                            if losses and tl_id in losses:
                                if losses[tl_id] is not None:
                                    episode_losses[tl_id].append(float(losses[tl_id]))
                            
                            # Track traffic metrics
                            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                            metrics = {
                                'queue_length': sum(traci.lane.getLastStepHaltingNumber(lane) 
                                                  for lane in controlled_lanes),
                                'waiting_time': sum(traci.lane.getWaitingTime(lane) 
                                                  for lane in controlled_lanes),
                                'throughput': sum(traci.lane.getLastStepVehicleNumber(lane)
                                                for lane in controlled_lanes)
                            }
                            for key, value in metrics.items():
                                episode_metrics[key].append(value)
                        
                        states = next_states
                        step += 1
                        
                        # Log progress periodically
                        if step % 100 == 0:
                            avg_reward = np.mean([r for r in rewards.values()])
                            avg_queue = np.mean(episode_metrics['queue_length'][-1:])
                            avg_wait = np.mean(episode_metrics['waiting_time'][-1:])
                            
                            logger.info(f"\nStep {step} of Episode {episode+1}")
                            logger.info(f"Average reward: {avg_reward:.3f}")
                            logger.info(f"Average queue length: {avg_queue:.2f}")
                            logger.info(f"Average waiting time: {avg_wait:.2f}")
                    
                    except Exception as e:
                        logger.error(f"Error during step {step}: {str(e)}")
                        logger.exception("Step error details:")
                        break
                
                # Calculate episode metrics
                if episode_rewards:
                    metrics['mean_rewards'].append(np.mean([r for r in episode_rewards.values()]))
                    metrics['global_rewards'].append(sum(episode_rewards.values()))
                    metrics['mean_waiting_times'].append(np.mean(episode_metrics['waiting_time']))
                    metrics['mean_queue_lengths'].append(np.mean(episode_metrics['queue_length']))
                    metrics['throughput'].append(np.mean(episode_metrics['throughput']))
                    
                    for tl_id in valid_traffic_lights:
                        metrics['agent_rewards'][tl_id].append(episode_rewards[tl_id])
                        valid_losses = [loss for loss in episode_losses[tl_id] if loss is not None]
                        if valid_losses:
                            metrics['losses'][tl_id].append(float(np.mean(valid_losses)))
                
                # Log episode summary
                logger.info(f"\nEpisode {episode+1} Summary:")
                logger.info(f"Steps completed: {step}")
                logger.info(f"Mean reward: {metrics['mean_rewards'][-1]:.2f}")
                logger.info(f"Global reward: {metrics['global_rewards'][-1]:.2f}")
                logger.info(f"Mean waiting time: {metrics['mean_waiting_times'][-1]:.2f}")
                logger.info(f"Mean queue length: {metrics['mean_queue_lengths'][-1]:.2f}")
                
                # Save checkpoint
                if (episode + 1) % 2 == 0:
                    save_dir = model_dir / f"checkpoint_{episode+1}"
                    save_dir.mkdir(exist_ok=True)
                    
                    manager.save_agents(save_dir)
                    plot_metrics(metrics, save_dir / "metrics.png")
                    
                    with open(save_dir / "metrics.json", 'w') as f:
                        json.dump(metrics, f, indent=4, default=json_serialize)
                
                # Check early stopping
                if episode >= 4:
                    recent_rewards = metrics['mean_rewards'][-5:]
                    recent_waiting = metrics['mean_waiting_times'][-5:]
                    recent_queues = metrics['mean_queue_lengths'][-5:]
                    
                    reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
                    waiting_trend = np.polyfit(range(len(recent_waiting)), recent_waiting, 1)[0]
                    queue_trend = np.polyfit(range(len(recent_queues)), recent_queues, 1)[0]
                    
                    if ((reward_trend > 0.005 or waiting_trend < -0.2) and
                        np.mean(recent_waiting) < 220 and
                        np.mean(recent_queues) < 9 and
                        episode > num_episodes // 3):
                        logger.info("Performance targets achieved, stopping early")
                        break
            
            except Exception as e:
                logger.error(f"Error during episode {episode + 1}: {str(e)}")
                logger.exception("Episode error details:")
                continue
    
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Critical training error: {str(e)}")
        logger.exception("Training error details:")
    finally:
        try:
            # Save final state
            final_dir = model_dir / "final"
            final_dir.mkdir(exist_ok=True)
            
            manager.save_agents(final_dir)
            plot_metrics(metrics, log_dir / "final_metrics.png")
            
            with open(final_dir / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=4, default=json_serialize)
            
            logger.info("\nTraining Summary:")
            logger.info(f"Episodes completed: {len(metrics['mean_rewards'])}")
            logger.info(f"Best reward: {max(metrics['mean_rewards']):.2f}")
            logger.info(f"Best waiting time: {min(metrics['mean_waiting_times']):.2f}")
        except Exception as e:
            logger.error(f"Error saving final state: {str(e)}")
        finally:
            env.close()
    
    return metrics

def main():
    """Main function to run independent agent training"""
    print("=" * 50)
    print("Starting main function")
    print("=" * 50)
    
    try:
        print("\nSetting up argument parser...")
        parser = argparse.ArgumentParser(description='Run independent agent training')
        parser.add_argument('--net-file', default='Version1/2024-11-05-18-42-37/osm.net.xml.gz',
                          help='Path to SUMO network file')
        parser.add_argument('--route-file', default='Version1/2024-11-05-18-42-37/osm.passenger.trips.xml',
                          help='Path to SUMO route file')
        parser.add_argument('--config', default='config/independent_agents.yaml',
                          help='Path to configuration file')
        parser.add_argument('--output-dir', default='experiments/outputs/independent',
                          help='Directory for outputs')
        parser.add_argument('--gui', action='store_true',
                          help='Enable SUMO GUI')
        parser.add_argument('--seed', type=int, default=42,
                          help='Random seed')
        
        print("\nParsing arguments...")
        args = parser.parse_args()
        
        print("\nParsed arguments:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")

        print("\nChecking SUMO environment...")
        if 'SUMO_HOME' not in os.environ:
            raise EnvironmentError("Please set SUMO_HOME environment variable")
        print(f"SUMO_HOME is set to: {os.environ['SUMO_HOME']}")

        # Check input files
        print("\nChecking input files...")
        net_path = Path(args.net_file).absolute()
        route_path = Path(args.route_file).absolute()
        config_path = Path(args.config).absolute()
        
        print(f"Network file path: {net_path}")
        print(f"Route file path: {route_path}")
        print(f"Config file path: {config_path}")
        
        if not net_path.exists():
            raise FileNotFoundError(f"Network file not found: {net_path}")
        if not route_path.exists():
            raise FileNotFoundError(f"Route file not found: {route_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        print("All input files found successfully")
        
        # Load configuration
        print("\nLoading configuration...")
        config = load_config(config_path)

        # Update environment configuration
        env_config = config.get('environment', {})
        env_config.update({
            'net_file': str(net_path.absolute()),
            'route_file': str(route_path.absolute()),
            'use_gui': args.gui,
            'out_csv_name': str(Path(args.output_dir) / 'independent_results.csv')
        })
        
        print("\nStarting independent traffic control training...")
        print("\nEnvironment Configuration:")
        for key, value in env_config.items():
            if key not in ['net_file', 'route_file', 'out_csv_name']:
                print(f"- {key}: {value}")
        
        print("\nAgent Configuration:")
        for key, value in config['agent'].items():
            print(f"- {key}: {value}")
        
        print("\nSetting up environment...")
        # Create output directories
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")
        
        # Run training
        print("\nStarting training process...")
        metrics = train_independent_agents(
            env_config,
            config['agent'],
            num_episodes=15  # Default number of episodes
        )
        
        print("\nTraining completed!")
        if metrics['mean_rewards']:
            print("\nFinal Performance Summary:")
            print(f"Average reward: {np.mean(metrics['mean_rewards']):.4f}")
            print(f"Best episode reward: {max(metrics['mean_rewards']):.4f}")
            print(f"Final waiting time: {metrics['waiting_times'][-1]:.2f}")
            print(f"Best waiting time: {min(metrics['waiting_times']):.2f}")
            print(f"Average throughput: {np.mean(metrics['throughput']):.2f}")
            
            # Calculate improvements
            waiting_improvement = ((metrics['waiting_times'][0] - metrics['waiting_times'][-1]) / 
                                metrics['waiting_times'][0] * 100)
            queue_improvement = ((metrics['queue_lengths'][0] - metrics['queue_lengths'][-1]) / 
                               metrics['queue_lengths'][0] * 100)
            
            print("\nImprovements:")
            print(f"Waiting time reduction: {waiting_improvement:.1f}%")
            print(f"Queue length reduction: {queue_improvement:.1f}%")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except FileNotFoundError as e:
        print(f"\nFile not found error: {e}")
        print("Please check if the network and route files exist in the specified locations.")
    except EnvironmentError as e:
        print(f"\nEnvironment error: {e}")
    except Exception as e:
        print(f"\nUnexpected error occurred: {str(e)}")
        print("Detailed error information:")
        import traceback
        traceback.print_exc()
    finally:
        print("\nProgram finished")

if __name__ == "__main__":
    main()