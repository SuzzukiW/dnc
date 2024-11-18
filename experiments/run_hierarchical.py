# experiments/run_hierarchical.py
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
from experiments.scenarios.communication.hierarchical import HierarchicalManager
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

def calculate_hierarchical_rewards(env, traffic_lights, region_manager):
    """Calculate rewards with hierarchical considerations"""
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
    
    # Calculate per-region metrics first
    region_metrics = defaultdict(lambda: defaultdict(float))
    for region_id, coordinator in region_manager.coordinators.items():
        region_vehicles = 0
        region_waiting = 0
        region_queue = 0
        
        for tl_id in coordinator.member_agents:
            if tl_id not in traffic_lights:
                continue
                
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            for lane in controlled_lanes:
                region_vehicles += traci.lane.getLastStepVehicleNumber(lane)
                region_waiting += traci.lane.getWaitingTime(lane)
                region_queue += traci.lane.getLastStepHaltingNumber(lane)
        
        if coordinator.member_agents:
            region_metrics[region_id].update({
                'vehicles': region_vehicles,
                'waiting': region_waiting,
                'queue': region_queue,
                'density': region_vehicles / len(coordinator.member_agents)
            })
    
    # Calculate individual rewards with regional context
    for tl_id in traffic_lights:
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        num_lanes = len(controlled_lanes)
        if num_lanes == 0:
            rewards[tl_id] = 0.0
            continue

# Part B of experiments/run_hierarchical.py

def calculate_metrics(env, traffic_lights, manager, step):
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
    
    # Calculate regional metrics
    region_metrics = {}
    for region_id, coordinator in manager.coordinators.items():
        region_vehicles = sum(
            traci.lane.getLastStepVehicleNumber(lane)
            for tl_id in coordinator.member_agents
            for lane in traci.trafficlight.getControlledLanes(tl_id)
        )
        region_metrics[region_id] = {
            'vehicles': region_vehicles,
            'density': region_vehicles / max(len(coordinator.member_agents), 1)
        }
    
    # Normalize metrics
    num_lanes = sum(len(traci.trafficlight.getControlledLanes(tl_id)) 
                   for tl_id in traffic_lights)
    metrics.update({
        'waiting_time': total_waiting / max(num_lanes, 1),
        'queue_length': total_queue / max(num_lanes, 1),
        'throughput': total_vehicles / max(num_lanes, 1),
        'avg_speed': total_speed / max(num_lanes, 1),
        'num_regions': len(manager.coordinators),
        'avg_region_size': np.mean([len(c.member_agents) for c in manager.coordinators.values()]),
        'simulation_step': step
    })
    
    return metrics

def plot_training_progress(metrics, save_path):
    """Plot training metrics with hierarchical information"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    axes[0, 0].plot(metrics['mean_rewards'], label='Mean')
    axes[0, 0].plot(metrics['regional_rewards'], label='Regional')
    axes[0, 0].set_title('Rewards per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    
    # Plot traffic metrics
    axes[0, 1].plot(metrics['waiting_times'], label='Waiting Time')
    axes[0, 1].plot(metrics['queue_lengths'], label='Queue Length')
    axes[0, 1].set_title('Traffic Metrics per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].legend()
    
    # Plot hierarchical metrics
    axes[1, 0].plot(metrics['num_regions'], label='Number of Regions')
    axes[1, 0].plot(metrics['avg_region_size'], label='Avg Region Size')
    axes[1, 0].set_title('Hierarchical Structure')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].legend()
    
    # Plot throughput
    axes[1, 1].plot(metrics['throughput'], label='Network Throughput')
    axes[1, 1].plot(metrics['coordination_rate'], label='Coordination Rate')
    axes[1, 1].set_title('Network Performance')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_hierarchical(env_config, agent_config, hierarchy_config, num_episodes=15):
    """Train hierarchical multi-agent system"""
    
    # Ensure GUI is disabled for training
    env_config['use_gui'] = False
    
    # Initialize environment
    env = MultiAgentSumoEnvironment(**env_config)
    
    # Get valid traffic lights
    valid_traffic_lights = filter_valid_traffic_lights(env)
    print(f"\nProceeding with training using {len(valid_traffic_lights)} valid traffic lights")
    
    # Update environment's traffic lights
    env.traffic_lights = valid_traffic_lights
    
    # Initialize hierarchical manager
    state_size = env.observation_spaces[valid_traffic_lights[0]].shape[0]
    action_size = env.action_spaces[valid_traffic_lights[0]].n
    
    manager = HierarchicalManager(
        state_size=state_size,
        action_size=action_size,
        net_file=env_config['net_file'],
        config={**agent_config, **hierarchy_config}
    )
    
    # Add agents to manager
    for tl_id in valid_traffic_lights:
        manager.add_agent(tl_id)
    
    # Setup logging and directories
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path("experiments/logs") / f"hierarchical_{timestamp}"
    model_dir = Path("experiments/models") / f"hierarchical_{timestamp}"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("hierarchical_training", log_dir / "training.log")
    
    # Initialize metrics
    metrics = {
        'mean_rewards': [],
        'regional_rewards': [],
        'waiting_times': [],
        'queue_lengths': [],
        'throughput': [],
        'num_regions': [],
        'avg_region_size': [],
        'coordination_rate': [],
        'agent_rewards': defaultdict(list),
        'region_rewards': defaultdict(list),
        'losses': defaultdict(list)
    }

# Part C

    try:
        logger.info("Starting hierarchical training...")
        steps_per_episode = env_config['num_seconds'] // env_config['delta_time']
        
        for episode in range(num_episodes):
            try:
                # Reset environment and get initial states
                states, _ = env.reset()
                states = {tl: state for tl, state in states.items() 
                         if tl in valid_traffic_lights}

                episode_metrics = defaultdict(list)
                episode_rewards = defaultdict(float)
                episode_losses = defaultdict(list)
                episode_coordination_rates = []
                done = False
                step = 0
                
                while not done and step < steps_per_episode:
                    try:
                        # Update regions based on current positions
                        positions = env.get_agent_positions()
                        manager.update_regions(positions)

                        # Process communications and get actions
                        manager.process_communications()
                        actions = manager.step(states)

                        # Execute actions
                        next_states, rewards, done, _, info = env.step(actions)

                        # Update agents and get losses
                        losses = manager.update(states, actions, rewards, next_states,
                                             {tl: done for tl in valid_traffic_lights})
                        for agent_id, loss in losses.items():
                            if loss is not None:
                                episode_losses[agent_id].append(loss)

                        # Track metrics
                        current_metrics = calculate_metrics(env, valid_traffic_lights, manager, step)
                        for key, value in current_metrics.items():
                            episode_metrics[key].append(value)

                        for tl_id, reward in rewards.items():
                            episode_rewards[tl_id] += reward

                        episode_coordination_rates.append(
                            sum(len(c.messages) for c in manager.coordinators.values()) / max(step, 1)
                        )

                        states = next_states
                        step += 1

                        # Log progress
                        if step % 100 == 0:
                            avg_reward = np.mean(list(rewards.values()))
                            avg_wait = np.mean(episode_metrics['waiting_time'][-100:])
                            avg_queue = np.mean(episode_metrics['queue_length'][-100:])
                            avg_coordination = np.mean(episode_coordination_rates[-100:])

                            logger.info(f"\nStep {step}/{steps_per_episode} of Episode {episode+1}")
                            logger.info(f"Average reward: {avg_reward:.3f}")
                            logger.info(f"Average waiting time: {avg_wait:.2f}")
                            logger.info(f"Average queue length: {avg_queue:.2f}")
                            logger.info(f"Average coordination rate: {avg_coordination:.2f}")
                            logger.info(f"Number of regions: {current_metrics['num_regions']}")
                            logger.info(f"Simulation time: {step * env_config['delta_time']} seconds")

                    except Exception as e:
                        logger.error(f"Error during step {step}: {str(e)}")
                        logger.exception("Step error details:")
                        break

                # Calculate episode summary metrics
                if episode_metrics:
                    metrics['mean_rewards'].append(np.mean(list(episode_rewards.values())))
                    metrics['regional_rewards'].append(
                        np.mean([r.get_performance_score() for r in manager.coordinators.values()])
                    )
                    metrics['waiting_times'].append(np.mean(episode_metrics['waiting_time']))
                    metrics['queue_lengths'].append(np.mean(episode_metrics['queue_length']))
                    metrics['throughput'].append(np.mean(episode_metrics['throughput']))
                    metrics['num_regions'].append(np.mean(episode_metrics['num_regions']))
                    metrics['avg_region_size'].append(np.mean(episode_metrics['avg_region_size']))
                    metrics['coordination_rate'].append(np.mean(episode_coordination_rates))

                # Log episode summary with safety checks
                logger.info(f"\nEpisode {episode+1} Summary:")
                logger.info(f"Steps completed: {step}/{steps_per_episode}")

                if metrics['mean_rewards']:
                    logger.info(f"Mean reward: {metrics['mean_rewards'][-1]:.2f}")
                    logger.info(f"Regional reward: {metrics['regional_rewards'][-1]:.2f}")
                    logger.info(f"Mean waiting time: {metrics['waiting_times'][-1]:.2f}")
                    logger.info(f"Mean queue length: {metrics['queue_lengths'][-1]:.2f}")
                    logger.info(f"Average throughput: {metrics['throughput'][-1]:.2f}")
                    logger.info(f"Number of regions: {metrics['num_regions'][-1]:.1f}")
                    logger.info(f"Average coordination rate: {metrics['coordination_rate'][-1]:.2f}")
                else:
                    logger.info("No metrics available for this episode")

                logger.info("-" * 50)

                # Save checkpoint
                if (episode + 1) % 2 == 0:
                    try:
                        save_dir = model_dir / f"checkpoint_{episode+1}"
                        save_dir.mkdir(exist_ok=True)

                        manager.save(save_dir)
                        plot_training_progress(metrics, save_dir / "progress.png")

                        with open(save_dir / "metrics.json", 'w') as f:
                            json.dump(metrics, f, indent=4, default=json_serialize)

                        logger.info(f"\nSaved checkpoint to {save_dir}")
                    except Exception as e:
                        logger.error(f"Error saving checkpoint: {str(e)}")

                # Check early stopping
                if episode >= 4:
                    recent_rewards = metrics['mean_rewards'][-5:]
                    recent_waiting = metrics['waiting_times'][-5:]
                    recent_queues = metrics['queue_lengths'][-5:]

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
    finally:
        try:
            final_dir = model_dir / "final"
            final_dir.mkdir(exist_ok=True)

            manager.save(final_dir)
            plot_training_progress(metrics, log_dir / "final_progress.png")

            with open(final_dir / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=4, default=json_serialize)

            logger.info("\nTraining Summary:")
            logger.info(f"Episodes completed: {len(metrics['mean_rewards'])}")
            logger.info(f"Best reward: {max(metrics['mean_rewards']):.2f}")
            logger.info(f"Best waiting time: {min(metrics['waiting_times']):.2f}")
        except Exception as e:
            logger.error(f"Error saving final state: {str(e)}")
        finally:
            env.close()

    return metrics

def main():
    """Main function for hierarchical training"""
    parser = argparse.ArgumentParser(description='Run hierarchical traffic control training')
    
    parser.add_argument('--net-file', required=True,
                      help='Path to SUMO network file')
    parser.add_argument('--route-file', required=True,
                      help='Path to SUMO route file')
    parser.add_argument('--config', default='config/hierarchical_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--output-dir', default='experiments/outputs/hierarchical',
                      help='Directory for outputs')
    parser.add_argument('--gui', action='store_true',
                      help='Enable SUMO GUI')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update environment configuration
    env_config = config['environment']
    env_config.update({
        'net_file': args.net_file,
        'route_file': args.route_file,
        'use_gui': args.gui,
        'out_csv_name': str(Path(args.output_dir) / 'hierarchical_results.csv')
    })
    
    print("\nStarting hierarchical traffic control training...")
    print("\nEnvironment Configuration:")
    for key, value in env_config.items():
        if key not in ['net_file', 'route_file', 'out_csv_name']:
            print(f"- {key}: {value}")
    
    print("\nAgent Configuration:")
    for key, value in config['agent'].items():
        print(f"- {key}: {value}")
    
    print("\nHierarchy Configuration:")
    for key, value in config['hierarchy'].items():
        print(f"- {key}: {value}")
        
    try:
        metrics = train_hierarchical(
            env_config,
            config['agent'],
            config['hierarchy'],
            num_episodes=config['training']['num_episodes']
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
            print(f"Average number of regions: {np.mean(metrics['num_regions']):.1f}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        print("\nTraining session completed")

if __name__ == "__main__":
    main()