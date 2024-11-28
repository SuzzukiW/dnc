# experiments/train/train_shared_experience.py

import os
import sys
import yaml
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime
from typing import Dict, List
import logging
import traci
import random
import time
import traceback
from evaluation_sets.metrics import (
    average_waiting_time, 
    total_throughput, 
    average_speed, 
    max_waiting_time
)

import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.agents.dqn_agent import DQNAgent
from src.environment.multi_agent_sumo_env_shared_experience import MultiAgentSumoEnvSharedExperience
from experiments.scenarios.communication.shared_experience import SharedExperienceScenario

class SharedExperienceTrainer:
    def __init__(self, config_path: str):
        """Initialize the shared experience trainer."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set up logging
        self._setup_logging()

        # Initialize scenario
        self.scenario = SharedExperienceScenario(config_path)
        
        # Initialize environment
        self.env = MultiAgentSumoEnvSharedExperience(
            net_file=self.config['simulation']['net_file'],
            route_file=self.config['simulation']['route_file'],
            scenario=self.scenario,
            use_gui=self.config['simulation']['gui']
        )
        
        # Get state and action sizes from environment
        self.state_size = self.env.observation_space_size
        self.action_size = self.env.action_space_size
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Shared experience buffer
        self.shared_buffer = defaultdict(list)
        
        # Training metrics
        self.episode_rewards = defaultdict(list)
        self.losses = defaultdict(list)

    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = self.config['training']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(
            log_dir, 
            f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_agents(self) -> Dict[str, DQNAgent]:
        """Initialize DQN agents for each intersection."""
        # Start simulation temporarily to get traffic lights
        if not traci.isLoaded():
            self.env.start_simulation()
            
        agents = {}
        intersection_ids = self.env.intersection_ids
        
        print(f"\nInitializing agents for {len(intersection_ids)} intersections...")
        
        for intersection_id in intersection_ids:
            print(f"Creating agent for intersection: {intersection_id}")
            agents[intersection_id] = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config=self.config['agent']
            )
            
        print("Agents initialized successfully!")
        return agents

    def _share_experiences(self, step):
        """Share experiences between agents and return the total number of experiences shared."""
        total_shared = 0
        if not self.scenario.should_share_experience(step):
            return total_shared

        sharing_config = self.scenario.get_sharing_config()
        
        # Get all experiences from memory
        all_experiences = []
        for intersection_id, agent in self.agents.items():
            if agent.memory:  # Check if agent has any experiences in memory
                # Take the last N experiences from the agent's memory
                num_experiences = min(10, len(agent.memory))  # Share last 10 experiences
                experiences = list(agent.memory)[-num_experiences:]
                all_experiences.extend(experiences)
        
        # Share experiences with all agents
        if all_experiences:
            for agent in self.agents.values():
                # Add experiences directly to agent's memory
                for exp in all_experiences:
                    if len(agent.memory) < agent.memory.maxlen:  # Check if memory isn't full
                        agent.memory.append(exp)
                        total_shared += 1
        
        return total_shared

    def _calculate_metrics(self, episode_rewards, episode_start_time, episode_losses):
        """Calculate metrics in baseline simulation format."""
        # Get current vehicles
        current_vehicles = set(traci.vehicle.getIDList())
        
        # Prepare vehicle data for metrics calculation
        vehicle_data = []
        for vid in current_vehicles:
            vehicle_data.append({
                'waiting_time': traci.vehicle.getWaitingTime(vid),
                'speed': traci.vehicle.getSpeed(vid)
            })
        
        # Prepare metrics dictionary using evaluation_sets metrics
        metrics = {
            'mean_waiting_time (s)': average_waiting_time(vehicle_data),
            'total_throughput (vehicles)': total_throughput(vehicle_data),
            'mean_speed (km/h)': average_speed(vehicle_data),
            'max_waiting_time (s)': max_waiting_time(vehicle_data)
        }
        
        return metrics

    def _log_episode_metrics(self, episode, metrics, epsilon):
        """Log metrics in a format similar to baseline simulation."""
        print("\nEpisode Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.2f}")
        
        # Optional: log epsilon for exploration
        print(f"Exploration Rate (Epsilon): {epsilon:.2f}")

    def train(self):
        """Train all agents."""
        print("\nStarting training...")
        
        # Initialize metrics storage
        all_episodes_metrics = {
            'mean_waiting_time (s)': [],
            'total_throughput (vehicles)': [],
            'mean_speed (km/h)': [],
            'max_waiting_time (s)': []
        }
        
        try:
            total_episodes = self.config['training'].get('total_episodes', 100)
            max_steps = self.config['training'].get('max_steps_per_episode', 1000)
            log_interval = self.config['training'].get('log_interval', 1)
            save_interval = self.config['training'].get('save_interval', 10)
            share_log_interval = 50  # Log shared experiences every 50 steps
            
            for episode in range(total_episodes):
                episode_start_time = time.time()
                states = self.env.reset()
                
                if not states or not self.agents:
                    print("No states or agents available. Exiting training.")
                    break
                
                # Collect episode-specific metrics
                episode_rewards = defaultdict(float)
                episode_losses = defaultdict(list)
                
                # Training loop for the episode
                for step in range(max_steps):
                    # Perform actions and collect rewards
                    actions = {}
                    for agent_id, agent in self.agents.items():
                        actions[agent_id] = agent.act(states[agent_id])
                    
                    next_states, rewards, dones, _ = self.env.step(actions)
                    
                    # Update episode rewards and losses
                    for agent_id, agent in self.agents.items():
                        episode_rewards[agent_id] += rewards[agent_id]
                        
                        # Store experience in memory
                        agent.remember(
                            states[agent_id],
                            actions[agent_id],
                            rewards[agent_id],
                            next_states[agent_id],
                            dones[agent_id]
                        )
                        
                        # Perform replay (training)
                        loss = agent.replay()
                        
                        if loss is not None:
                            episode_losses[agent_id].append(loss)
                    
                    # Optionally share experiences
                    if step % share_log_interval == 0:
                        shared_count = self._share_experiences(step)
                        if shared_count > 0:
                            print(f"Shared {shared_count} experiences at step {step}")
                    
                    # Update states
                    states = next_states
                    
                    # Check if episode is done
                    if all(dones.values()):
                        break
                
                # Calculate and log metrics
                metrics = self._calculate_metrics(episode_rewards, episode_start_time, episode_losses)
                
                # Store metrics for this episode
                for key in all_episodes_metrics.keys():
                    all_episodes_metrics[key].append(metrics[key])
                
                # Log episode metrics periodically
                if (episode + 1) % log_interval == 0:
                    print(f"\nEpisode {episode + 1} Metrics:")
                    for key, value in metrics.items():
                        print(f"{key}: {value:.2f}")
                
                # Save models periodically
                if (episode + 1) % save_interval == 0:
                    self._save_models(episode + 1)
            
            # Final summary of metrics
            print("\nTraining Complete. Final Metrics Summary:")
            for key, values in all_episodes_metrics.items():
                print(f"{key}: {np.mean(values):.2f} ± {np.std(values):.2f}")
        
        except Exception as e:
            print(f"Training interrupted: {e}")
            traceback.print_exc()

    def _evaluate(self):
        """Evaluate current policy without exploration."""
        eval_rewards = defaultdict(float)
        states = self.env.reset()
        
        for step in range(self.config['training']['max_steps_per_episode']):
            actions = {}
            for intersection_id, state in states.items():
                actions[intersection_id] = self.agents[intersection_id].act(state, training=False)
            
            next_states, rewards, dones, _ = self.env.step(actions)
            
            for intersection_id, reward in rewards.items():
                eval_rewards[intersection_id] += reward
            
            if all(dones.values()):
                break
                
            states = next_states
        
        avg_eval_reward = np.mean(list(eval_rewards.values()))
        self.logger.info(f"Evaluation - Average Reward: {avg_eval_reward:.2f}")

    def _save_models(self, episode: int, final: bool = False):
        """Save agent models."""
        save_dir = self.config['training']['checkpoint_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        for intersection_id, agent in self.agents.items():
            save_path = os.path.join(
                save_dir,
                f"agent_{intersection_id}_episode_{episode}.pth"
            )
            agent.save(save_path)
        
        self.logger.info(f"Saved agent models at episode {episode}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train shared experience multi-agent reinforcement learning scenario')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Use the config path from arguments
    trainer = SharedExperienceTrainer(args.config)
    trainer.train()

if __name__ == "__main__":
    main()