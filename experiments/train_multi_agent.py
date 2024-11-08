# experiments/train_multi_agent.py

import os
import sys
from pathlib import Path
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

from src.agents import MultiAgentDQN
from src.environment import MultiAgentSumoEnvironment
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

def calculate_rewards(env, traffic_lights):
    """Balanced reward calculation with stability focus"""
    rewards = {}
    MAX_WAITING_TIME = 150.0  # Reduced back
    MIN_GREEN_UTIL = 0.35
    MAX_QUEUE_LENGTH = 10     # Back to original
    
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

def save_metrics(metrics, path):
    """Save metrics with proper JSON serialization"""
    serializable_metrics = {
        'mean_rewards': [float(x) for x in metrics['mean_rewards']],
        'global_rewards': [float(x) for x in metrics['global_rewards']],
        'mean_waiting_times': [float(x) for x in metrics['mean_waiting_times']],
        'mean_queue_lengths': [float(x) for x in metrics['mean_queue_lengths']],
        'agent_rewards': {k: [float(x) for x in v] 
                        for k, v in metrics['agent_rewards'].items()},
        'losses': {k: [float(x) for x in v] 
                  for k, v in metrics['losses'].items()}
    }
    
    with open(path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)

def train_multi_agent(env_config, agent_config, num_episodes=10):
    """Train multiple cooperative DQN agents with enhanced traffic management"""
    
    # Ensure GUI is disabled for automated training
    env_config['use_gui'] = False
    
    # Initialize environment
    env = MultiAgentSumoEnvironment(**env_config)
    
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
    
    # Calculate expected steps per episode
    steps_per_episode = env_config['num_seconds'] // env_config['delta_time']
    print(f"\nTraining Configuration:")
    print(f"Steps per episode: {steps_per_episode}")
    print(f"Episode duration: {env_config['num_seconds']} seconds")
    print(f"Action interval: {env_config['delta_time']} seconds")
    
    # Initialize traffic pressure tracking
    if not hasattr(env, 'traffic_pressure'):
        env.traffic_pressure = defaultdict(float)
    
    # Initialize multi-agent system
    state_size = env.observation_spaces[valid_traffic_lights[0]].shape[0]
    action_size = env.action_spaces[valid_traffic_lights[0]].n
    
    print("\nAgent Configuration:")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    
    multi_agent_system = MultiAgentDQN(
        state_size=state_size,
        action_size=action_size,
        agent_ids=valid_traffic_lights,
        neighbor_map=neighbor_map,
        config=agent_config
    )
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path("experiments/logs") / f"multi_agent_{timestamp}"
    model_dir = Path("experiments/models") / f"multi_agent_{timestamp}"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "final_models").mkdir(exist_ok=True)
    
    logger = setup_logger("multi_agent_training", log_dir / "training.log")
    
    # Initialize metrics
    metrics = {
        'mean_rewards': [],
        'global_rewards': [],
        'mean_waiting_times': [],
        'mean_queue_lengths': [],
        'throughput': [],
        'traffic_pressure': [],
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
                episode_queues = defaultdict(list)
                episode_waiting_times = defaultdict(list)
                episode_throughput = []
                episode_pressure = defaultdict(list)
                done = False
                step = 0
                
                while not done and step < steps_per_episode:
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
                        
                        # Execute actions
                        next_states, raw_rewards, done, _, info = env.step(actions)
                        
                        # Calculate custom rewards with pressure consideration
                        rewards = calculate_rewards(env, valid_traffic_lights)
                        
                        # Update agents
                        losses = multi_agent_system.step(
                            states, actions, rewards, next_states,
                            {tl: done for tl in valid_traffic_lights},
                            global_reward=info.get('global_reward', 0)
                        )
                        
                        # Update metrics
                        for tl_id in valid_traffic_lights:
                            # Add rewards
                            episode_rewards[tl_id] += rewards.get(tl_id, 0)
                            
                            # Add losses if available
                            if losses and tl_id in losses:
                                valid_loss = losses[tl_id]
                                if valid_loss is not None:
                                    episode_losses[tl_id].append(float(valid_loss))
                            
                            # Calculate traffic metrics
                            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                            total_queue = sum(traci.lane.getLastStepHaltingNumber(lane) 
                                            for lane in controlled_lanes)
                            total_waiting = sum(traci.lane.getWaitingTime(lane) 
                                              for lane in controlled_lanes)
                            
                            episode_queues[tl_id].append(total_queue)
                            episode_waiting_times[tl_id].append(total_waiting)
                        
                        # Track throughput and update states
                        total_vehicles = sum(traci.lane.getLastStepVehicleNumber(lane)
                                           for tl_id in valid_traffic_lights
                                           for lane in traci.trafficlight.getControlledLanes(tl_id))
                        episode_throughput.append(total_vehicles)
                        
                        states = {tl: state for tl, state in next_states.items() 
                                if tl in valid_traffic_lights}
                        step += 1
                        
                        # Log progress
                        if step % 100 == 0:
                            avg_reward = np.mean([r for r in rewards.values()])
                            avg_queue = np.mean([q[-1] for q in episode_queues.values()])
                            avg_pressure = np.mean([env.traffic_pressure[tl] for tl in valid_traffic_lights])
                            
                            logger.info(f"\nStep {step}/{steps_per_episode} of Episode {episode+1}")
                            logger.info(f"Average reward: {avg_reward:.3f}")
                            logger.info(f"Average queue length: {avg_queue:.2f}")
                            logger.info(f"Traffic pressure: {avg_pressure:.3f}")
                            logger.info(f"Vehicles in network: {total_vehicles}")
                            logger.info(f"Simulation time: {step * env_config['delta_time']} seconds")
                    
                    except Exception as e:
                        logger.error(f"Error during step {step}: {str(e)}")
                        logger.exception("Step error details:")
                        break
                
                # Calculate episode metrics
                if episode_rewards:
                    metrics['mean_rewards'].append(np.mean([r for r in episode_rewards.values()]))
                    metrics['global_rewards'].append(sum(episode_rewards.values()))
                    metrics['mean_waiting_times'].append(
                        np.mean([np.mean(w) for w in episode_waiting_times.values()])
                    )
                    metrics['mean_queue_lengths'].append(
                        np.mean([np.mean(q) for q in episode_queues.values()])
                    )
                    metrics['throughput'].append(np.mean(episode_throughput))
                    metrics['traffic_pressure'].append(
                        np.mean([np.mean(p) for p in episode_pressure.values()])
                    )
                    
                    # Update per-agent metrics safely
                    for tl_id in valid_traffic_lights:
                        metrics['agent_rewards'][tl_id].append(episode_rewards[tl_id])
                        
                        valid_losses = [loss for loss in episode_losses[tl_id] if loss is not None]
                        if valid_losses:
                            metrics['losses'][tl_id].append(float(np.mean(valid_losses)))
                        else:
                            metrics['losses'][tl_id].append(0.0)
                
                # Log episode summary
                logger.info(f"\nEpisode {episode+1} Summary:")
                logger.info(f"Steps completed: {step}/{steps_per_episode}")
                logger.info(f"Mean reward: {metrics['mean_rewards'][-1]:.2f}")
                logger.info(f"Global reward: {metrics['global_rewards'][-1]:.2f}")
                logger.info(f"Mean waiting time: {metrics['mean_waiting_times'][-1]:.2f}")
                logger.info(f"Mean queue length: {metrics['mean_queue_lengths'][-1]:.2f}")
                logger.info(f"Average throughput: {metrics['throughput'][-1]:.2f}")
                logger.info(f"Average traffic pressure: {metrics['traffic_pressure'][-1]:.3f}")
                logger.info("-" * 50)
                
                # Save checkpoint every 2 episodes
                if (episode + 1) % 2 == 0:
                    try:
                        save_dir = model_dir / f"checkpoint_{episode+1}"
                        save_dir.mkdir(exist_ok=True)
                        
                        multi_agent_system.save_agents(save_dir)
                        plot_metrics(metrics, save_dir / "metrics.png")
                        
                        with open(save_dir / "metrics.json", 'w') as f:
                            json.dump(metrics, f, indent=4, default=json_serialize)
                        
                        logger.info(f"\nCheckpoint {episode + 1} Summary:")
                        logger.info(f"Mean reward: {metrics['mean_rewards'][-1]:.2f}")
                        logger.info(f"Mean waiting time: {metrics['mean_waiting_times'][-1]:.2f}")
                        logger.info(f"Mean queue length: {metrics['mean_queue_lengths'][-1]:.2f}")
                        
                    except Exception as e:
                        logger.error(f"Error saving checkpoint: {str(e)}")
                        logger.exception("Checkpoint error details:")
                
                # Check early stopping criteria
                if episode >= 4:
                    recent_rewards = metrics['mean_rewards'][-5:]
                    recent_waiting = metrics['mean_waiting_times'][-5:]
                    recent_queues = metrics['mean_queue_lengths'][-5:]
                    
                    reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
                    waiting_trend = np.polyfit(range(len(recent_waiting)), recent_waiting, 1)[0]
                    queue_trend = np.polyfit(range(len(recent_queues)), recent_queues, 1)[0]
                    
                    avg_waiting = np.mean(recent_waiting)
                    avg_queue = np.mean(recent_queues)
                    
                    logger.info(f"\nPerformance Trend Analysis:")
                    logger.info(f"Reward trend: {reward_trend:.4f}")
                    logger.info(f"Waiting time trend: {waiting_trend:.4f}")
                    logger.info(f"Queue trend: {queue_trend:.4f}")
                    logger.info(f"Average waiting time: {avg_waiting:.2f}")
                    logger.info(f"Average queue length: {avg_queue:.2f}")
                    
                    # More balanced stopping criteria
                    if ((reward_trend > 0.005 or waiting_trend < -0.2) and  # Less aggressive
                        avg_waiting < 220 and                               # More realistic
                        avg_queue < 9 and
                        episode > num_episodes // 3):
                        logger.info("Performance targets achieved, stopping early")
                        break
            
            except Exception as e:
                logger.error(f"Error during episode {episode + 1}: {str(e)}")
                logger.exception("Episode error details:")
                continue
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Critical training error: {str(e)}")
        logger.exception("Training error details:")
    finally:
        try:
            if metrics['mean_rewards']:
                final_dir = model_dir / "final"
                final_dir.mkdir(exist_ok=True)
                
                multi_agent_system.save_agents(final_dir)
                plot_metrics(metrics, log_dir / "final_metrics.png")
                
                with open(final_dir / "metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=4, default=json_serialize)
                
                logger.info("Final state saved successfully")
                logger.info("\nTraining Summary:")
                logger.info(f"Episodes completed: {len(metrics['mean_rewards'])}")
                logger.info(f"Best reward: {max(metrics['mean_rewards']):.2f}")
                logger.info(f"Best waiting time: {min(metrics['mean_waiting_times']):.2f}")
        except Exception as e:
            logger.error(f"Error saving final state: {str(e)}")
            logger.exception("Final save error details:")
        finally:
            env.close()
    
    return metrics

def main():
    """Main training function with fixed environment configuration"""
    
    # Load base configurations
    env_config = load_config(project_root / 'config/env_config.yaml')
    agent_config = load_config(project_root / 'config/agent_config.yaml')
    
    # Update environment configuration with only supported parameters
    env_config.update({
        'neighbor_distance': 120,      # Back to more focused coordination
        'yellow_time': 3,             # Back to safer timing
        'min_green': 12,              # More balanced timing
        'max_green': 45,              # More balanced
        'num_seconds': 1800,          # Back to moderate episode length
        'delta_time': 8,              # Less frequent but more stable decisions
        'use_gui': False,
        'max_depart_delay': 200,      # More moderate
        'time_to_teleport': 200       # More moderate
    })

    # Update agent configuration (keeping the same as before)
    agent_config.update({
        # Memory and batch settings
        'memory_size': 150000,        # More moderate memory
        'batch_size': 192,            # More moderate batch size
        'prioritized_replay': True,
        'priority_alpha': 0.65,
        'priority_beta': 0.45,
        
        # Learning parameters
        'learning_rate': 0.0004,      # Moderate learning rate
        'learning_rate_decay': 0.998,
        'min_learning_rate': 0.0001,
        'epsilon_start': 1.0,
        'epsilon_min': 0.03,          # More exploration
        'epsilon_decay': 0.93,        # Moderate decay
        'gamma': 0.96,                # Balanced future discount
        
        # Network architecture
        'hidden_size': 384,           # More moderate size
        'num_hidden_layers': 3,       # Back to original depth
        'activation': 'relu',
        'dropout_rate': 0.12,
        
        # Training stability
        'target_update_frequency': 75,
        'gradient_clip': 0.8,
        'double_dqn': True,
        'dueling_dqn': True,
        
        # Multi-agent specific
        'communication_mode': 'shared_experience',
        'reward_type': 'hybrid',
        'shared_memory_fraction': 0.35, # More moderate sharing
        'neighbor_update_freq': 4       # More moderate updates
    })

    print("\nStarting traffic light control training...")
    print("\nEnvironment Configuration:")
    for key, value in env_config.items():
        if key not in ['net_file', 'route_file', 'out_csv_name']:
            print(f"- {key}: {value}")
    
    print("\nAgent Configuration:")
    for key, value in agent_config.items():
        print(f"- {key}: {value}")

    # Set up training parameters
    training_params = {
        'num_episodes': 15,           # Back to moderate
        'eval_frequency': 2,
        'checkpoint_frequency': 2,
        'early_stopping_patience': 4,
        'min_improvement': 0.02,
        'max_no_improvement': 5,
        'reward_threshold': -0.4,
        'waiting_time_threshold': 200,  # More realistic
        'queue_length_threshold': 8     # More realistic
    }

    print("\nTraining Parameters:")
    for key, value in training_params.items():
        print(f"- {key}: {value}")

    try:
        # Run training
        print("\nStarting training...")
        metrics = train_multi_agent(env_config, agent_config, num_episodes=training_params['num_episodes'])

        # Print final results
        print("\nTraining completed!")
        if metrics['mean_rewards']:
            print("\nFinal Metrics Summary:")
            print(f"Average reward per agent: {np.mean(metrics['mean_rewards']):.4f}")
            print(f"Best episode reward: {max(metrics['mean_rewards']):.4f}")
            print(f"Final average waiting time: {metrics['mean_waiting_times'][-1]:.2f} seconds")
            print(f"Best waiting time: {min(metrics['mean_waiting_times']):.2f} seconds")
            print(f"Average queue length: {np.mean(metrics['mean_queue_lengths']):.2f} vehicles")
            if 'traffic_pressure' in metrics:
                print(f"Average traffic pressure: {np.mean(metrics['traffic_pressure']):.3f}")
            if 'throughput' in metrics:
                print(f"Average throughput: {np.mean(metrics['throughput']):.2f} vehicles")

            # Calculate improvement percentages
            waiting_time_improvement = ((metrics['mean_waiting_times'][0] - metrics['mean_waiting_times'][-1]) / 
                                     metrics['mean_waiting_times'][0] * 100)
            queue_improvement = ((metrics['mean_queue_lengths'][0] - metrics['mean_queue_lengths'][-1]) / 
                               metrics['mean_queue_lengths'][0] * 100)

            print("\nPerformance Improvements:")
            print(f"Waiting time reduction: {waiting_time_improvement:.1f}%")
            print(f"Queue length reduction: {queue_improvement:.1f}%")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        print("\nTraining session completed")

if __name__ == "__main__":
    main()