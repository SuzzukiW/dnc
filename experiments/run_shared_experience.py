#!/usr/bin/env python3
# experiments/run_shared_experience.py

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

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scenarios.communication.shared_experience import (
    SharedMemory, 
    ExperienceCoordinator, 
    SharedExperienceNetwork
)

def setup_logger(log_file=None, log_level=logging.INFO):
    """
    Setup logger with file and console output
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level
    
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('SharedExperienceLogger')
    logger.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

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

def calculate_metrics(shared_memory, experience_coordinator, episode):
    """
    Calculate performance metrics for shared experience learning
    
    Args:
        shared_memory: SharedMemory instance
        experience_coordinator: ExperienceCoordinator instance
        episode: Current episode number
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'episode': episode,
        'total_experiences': len(shared_memory.experiences),
        'shared_experiences': len(shared_memory.shared_experiences),
        'agent_contributions': dict(shared_memory.agent_contributions),
        'sharing_successes': experience_coordinator.metrics['sharing_successes'],
        'similarity_scores': experience_coordinator.metrics['similarity_scores']
    }
    return metrics

def plot_training_progress(metrics_history, save_path):
    """
    Plot training progress metrics
    
    Args:
        metrics_history: List of metric dictionaries
        save_path: Path to save plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot total and shared experiences
    plt.subplot(2, 2, 1)
    plt.plot([m['episode'] for m in metrics_history], 
             [m['total_experiences'] for m in metrics_history], 
             label='Total Experiences')
    plt.plot([m['episode'] for m in metrics_history], 
             [m['shared_experiences'] for m in metrics_history], 
             label='Shared Experiences')
    plt.title('Experience Accumulation')
    plt.xlabel('Episode')
    plt.ylabel('Number of Experiences')
    plt.legend()
    
    # Plot agent contributions
    plt.subplot(2, 2, 2)
    agent_contributions = defaultdict(list)
    for m in metrics_history:
        for agent, contrib in m['agent_contributions'].items():
            agent_contributions[agent].append(contrib)
    
    for agent, contribs in agent_contributions.items():
        plt.plot(range(len(contribs)), contribs, label=f'{agent} Contributions')
    plt.title('Agent Experience Contributions')
    plt.xlabel('Episode')
    plt.ylabel('Contributions')
    plt.legend()
    
    # Plot similarity scores
    plt.subplot(2, 2, 3)
    plt.plot(range(len(metrics_history)), 
             [np.mean(m['similarity_scores']) if m['similarity_scores'] else 0 
              for m in metrics_history], 
             label='Mean Similarity')
    plt.title('Experience Similarity')
    plt.xlabel('Episode')
    plt.ylabel('Mean Similarity Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_shared_experience(
    num_agents=4, 
    state_size=10, 
    action_size=5, 
    shared_size=20, 
    num_episodes=10,
    log_dir=None
):
    """
    Train multi-agent system with shared experience learning
    
    Args:
        num_agents: Number of agents to train
        state_size: Dimensionality of state space
        action_size: Number of possible actions
        shared_size: Size of shared experience embedding
        num_episodes: Number of training episodes
        log_dir: Directory to save logs and plots
    """
    # Setup logging
    if log_dir is None:
        log_dir = Path(__file__).parent / 'logs' / f'shared_exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(log_dir / 'training.log')
    logger.info(f"Training Shared Experience Model with {num_agents} agents")
    
    # Initialize shared components
    shared_memory = SharedMemory(capacity=10000)
    experience_coordinator = ExperienceCoordinator()
    
    # Initialize agent networks
    agent_networks = [
        SharedExperienceNetwork(
            state_size=state_size, 
            action_size=action_size, 
            shared_size=shared_size
        ) for _ in range(num_agents)
    ]
    
    # Optimizers
    optimizers = [torch.optim.Adam(net.parameters(), lr=0.001) for net in agent_networks]
    
    # Loss function
    loss_fn = torch.nn.MSELoss()
    
    # Training loop
    metrics_history = []
    
    for episode in range(num_episodes):
        # Initialize episode states for each agent
        agent_states = [np.random.rand(state_size) for _ in range(num_agents)]
        
        for step in range(100):  # Max steps per episode
            agent_actions = []
            agent_rewards = []
            agent_next_states = []
            
            # Agent interactions
            for agent_id in range(num_agents):
                # Select action
                state_tensor = torch.FloatTensor(agent_states[agent_id]).unsqueeze(0)
                with torch.no_grad():
                    q_values = agent_networks[agent_id](state_tensor, None)
                action = torch.argmax(q_values).item()
                agent_actions.append(action)
                
                # Simulate reward
                reward = np.random.rand() * (action + 1)
                agent_rewards.append(reward)
                
                # Generate next state
                next_state = np.random.rand(state_size)
                agent_next_states.append(next_state)
                
                # Add experience to shared memory
                exp_id = shared_memory.add_experience(
                    state=agent_states[agent_id],
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=(step == 99),
                    agent_id=f"agent_{agent_id}"
                )
                
                # Register experience with coordinator
                experience_coordinator.register_experience(
                    exp_id=exp_id,
                    state=agent_states[agent_id],
                    agent_id=f"agent_{agent_id}"
                )
            
            # Experience sharing and learning
            for agent_id in range(num_agents):
                # Find similar experiences
                similar_exps = experience_coordinator.find_similar_experiences(
                    state=agent_states[agent_id],
                    agent_id=f"agent_{agent_id}"
                )
                
                # Get shared batch
                shared_batch = shared_memory.get_shared_batch(
                    batch_size=min(len(similar_exps), 16),
                    requesting_agent=f"agent_{agent_id}"
                )
                
                # Train with shared experiences
                if shared_batch:
                    shared_states = torch.FloatTensor([exp[1] for exp in shared_batch])
                    shared_rewards = torch.FloatTensor([exp[3] for exp in shared_batch])
                    
                    current_q_values = agent_networks[agent_id](
                        torch.FloatTensor(agent_states[agent_id]).unsqueeze(0),
                        shared_states
                    )
                    
                    loss = loss_fn(current_q_values.squeeze(), shared_rewards)
                    
                    optimizers[agent_id].zero_grad()
                    loss.backward()
                    optimizers[agent_id].step()
                
                # Update agent states
                agent_states[agent_id] = agent_next_states[agent_id]
        
        # Calculate and log metrics
        episode_metrics = calculate_metrics(shared_memory, experience_coordinator, episode)
        metrics_history.append(episode_metrics)
        
        logger.info(f"Episode {episode}: {json.dumps(episode_metrics, default=json_serialize)}")
    
    # Plot and save training progress
    plot_save_path = log_dir / 'training_progress.png'
    plot_training_progress(metrics_history, plot_save_path)
    
    logger.info(f"Training complete. Metrics saved to {log_dir}")
    
    return metrics_history

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Shared Experience Multi-Agent Training')
    parser.add_argument('--num_agents', type=int, default=4, help='Number of agents')
    parser.add_argument('--state_size', type=int, default=10, help='State space dimensionality')
    parser.add_argument('--action_size', type=int, default=5, help='Action space size')
    parser.add_argument('--shared_size', type=int, default=20, help='Shared experience embedding size')
    parser.add_argument('--episodes', type=int, default=10, help='Number of training episodes')
    parser.add_argument('--log_dir', type=str, default=None, help='Logging directory')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run training
    train_shared_experience(
        num_agents=args.num_agents,
        state_size=args.state_size,
        action_size=args.action_size,
        shared_size=args.shared_size,
        num_episodes=args.episodes,
        log_dir=args.log_dir
    )

if __name__ == "__main__":
    main()
