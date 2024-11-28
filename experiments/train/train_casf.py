# experiments/train/train_casf.py

import os
import yaml
import numpy as np
import tensorflow as tf
import logging
from datetime import datetime
from tqdm import trange
from src.environment.multi_agent_sumo_env_casf import MACSFEnvironment
from src.agents.casf import CASFAgent
from src.utils.logger import get_logger
import traci
from evaluation_sets.metrics import (
    average_waiting_time,
    total_throughput,
    average_speed,
    max_waiting_time
)

class MACSFTrainer:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Extract configuration sections
        self.training_config = config.get('training', {})
        self.model_config = config.get('model', {})
        self.env_config = config.get('environment', {})
        self.memory_config = config.get('memory', {})
        self.logging_config = config.get('logging', {})
        
        # Extract reward_weights from environment configuration with default values
        self.reward_weights = self.env_config.get('reward_weights', {
            'waiting_time': 1.0,
            'queue_length': 1.0,
            'throughput': 1.0
        })
        
        # Initialize logger
        self.logger = get_logger('CASF_Trainer', level=logging.INFO)
        self.logger.info("Initializing MACSFTrainer")
        
        # Log the loaded configurations for debugging
        self.logger.debug(f"Training Configuration: {self.training_config}")
        self.logger.debug(f"Model Configuration: {self.model_config}")
        self.logger.debug(f"Environment Configuration: {self.env_config}")
        self.logger.debug(f"Memory Configuration: {self.memory_config}")
        self.logger.debug(f"Logging Configuration: {self.logging_config}")
        self.logger.debug(f"Reward Weights: {self.reward_weights}")

        # Initialize environment
        self.env = MACSFEnvironment(config=self.env_config)
        
        # Store metrics for evaluation
        self.metrics_history = {
            'average_waiting_time': [],
            'total_throughput': [],
            'average_speed': [],
            'max_waiting_time': []
        }
        
        # Initialize agents
        self.agents = []
        traffic_light_ids = traci.trafficlight.getIDList()
        self.logger.info(f"Found {len(traffic_light_ids)} traffic lights in the simulation.")
        
        for agent_id in traffic_light_ids:
            agent = CASFAgent(
                agent_id=agent_id,
                state_dim=self.env.state_size,
                action_dim=self.env_config.get('action_dim', 1),
                network_config=self.model_config,
                memory_config=self.memory_config
            )
            self.agents.append(agent)
        self.logger.info(f"Initialized {len(self.agents)} agents")
        
        # Set up TensorBoard logging
        self.setup_tensorboard()

    def setup_tensorboard(self):
        """
        Sets up TensorBoard logging directories.
        """
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        train_log_dir = f'logs/casf/{current_time}/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.logger.info(f"TensorBoard logging set up at {train_log_dir}")

    def compute_traffic_metrics(self, info):
        """
        Extracts and computes traffic metrics from the environment's info dictionary.
        """
        avg_waiting_time = info.get('avg_waiting_time', 0)
        avg_queue_length = info.get('avg_queue_length', 0)
        throughput = info.get('throughput', 0)
        return {
            'avg_waiting_time': avg_waiting_time,
            'avg_queue_length': avg_queue_length,
            'throughput': throughput
        }

    def train(self):
        """
        Main training loop for the MACSFTrainer.
        """
        step = 0  # Global step counter
        
        self.logger.info("Starting training")

        all_metrics = {
            'average_waiting_time': [],
            'total_throughput': [],
            'average_speed': [],
            'max_waiting_time': []
        }

        for episode in trange(1, self.training_config.get('num_episodes', 10) + 1, desc="Episodes"):
            self.logger.info(f"\nStarting Episode {episode}/{self.training_config.get('num_episodes', 10)}")
            
            # Reset metrics history for this episode
            self.metrics_history = {k: [] for k in self.metrics_history.keys()}
            
            # Reset environment and get initial states
            states, neighbor_states = self.env.reset()
            
            episode_rewards = []  # To track rewards per step in the episode
            
            # Create a progress bar for steps within the episode
            for step_in_episode in trange(1, self.training_config.get('steps_per_episode', 600) + 1, desc=f"Episode {episode} Steps", leave=False):
                # Get actions from all agents
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.act(states[i], neighbor_states[i])
                    actions.append(action)
                
                # Environment step
                next_states, neighbor_next_states, rewards, dones, info = self.env.step(actions)
                step += 1  # Increment global step
                
                # Get vehicle data and update metrics
                vehicle_data = self.env.get_vehicle_data()
                self.metrics_history['average_waiting_time'].append(average_waiting_time(vehicle_data))
                self.metrics_history['total_throughput'].append(total_throughput(vehicle_data))
                self.metrics_history['average_speed'].append(average_speed(vehicle_data))
                self.metrics_history['max_waiting_time'].append(max_waiting_time(vehicle_data))
                
                # Define neighbor_next_actions as zeros (baseline behavior)
                neighbor_next_actions = [
                    np.zeros((agent.config.get('max_neighbors', 4), agent.action_dim))
                    for agent in self.agents
                ]
                
                # Store experiences including neighbor_next_actions
                for i, agent in enumerate(self.agents):
                    agent.store_experience(
                        state=states[i], 
                        action=actions[i],
                        reward=rewards[i], 
                        next_state=next_states[i],
                        neighbor_next_state=neighbor_next_states[i],
                        neighbor_next_action=neighbor_next_actions[i],
                        done=dones[i]
                    )
                
                # Aggregate rewards for the episode
                episode_rewards.append(np.mean(rewards))
                
                # Compute and log traffic metrics
                metrics = self.compute_traffic_metrics(info)
                with self.train_summary_writer.as_default():
                    for metric_name, metric_value in metrics.items():
                        tf.summary.scalar(f'traffic_metrics/{metric_name}', metric_value, step=step)
                self.train_summary_writer.flush()
                
                # Log metrics to console
                self.logger.info(f"Episode {episode} Step {step_in_episode} - Metrics: {metrics}")
                
                # Update networks at specified intervals
                if step % self.training_config.get('update_every', 100) == 0:
                    losses = self.update_networks(self.training_config.get('batch_size', 128), step)
                    self.log_training_stats(step, losses)
                
                # Update current states
                states = next_states
                neighbor_states = neighbor_next_states
                
                # Check if the episode is done
                if all(dones):
                    self.logger.info(f"Episode {episode} finished after {step_in_episode} steps.")
                    break  # Proceed to the next episode
            
            # Store final metrics for this episode
            for metric in all_metrics:
                if self.metrics_history[metric]:  # Check if we have any metrics
                    all_metrics[metric].append(self.metrics_history[metric][-1])
                else:
                    all_metrics[metric].append(0)  # Fallback if no metrics were recorded

            # Log episode statistics
            mean_episode_reward = np.mean(episode_rewards)
            self.log_episode_stats(episode, mean_episode_reward)
            
            # Save checkpoints at specified intervals
            if episode % self.training_config.get('checkpoint_interval', 1) == 0:
                self.save_checkpoints(episode)
        
        # Calculate and print average metrics across all episodes
        print("\nCASF Training Results - Averaged over {} episodes:".format(self.training_config['num_episodes']))
        print(f"Average Waiting Time: {np.mean(all_metrics['average_waiting_time']):.2f} ± {np.std(all_metrics['average_waiting_time']):.2f} seconds")
        print(f"Average Total Throughput: {np.mean(all_metrics['total_throughput']):.2f} ± {np.std(all_metrics['total_throughput']):.2f} vehicles")
        print(f"Average Speed: {np.mean(all_metrics['average_speed']):.2f} ± {np.std(all_metrics['average_speed']):.2f} km/h")
        print(f"Average Max Waiting Time: {np.mean(all_metrics['max_waiting_time']):.2f} ± {np.std(all_metrics['max_waiting_time']):.2f} seconds")
        
        # Print final training summary
        print("\n" + "="*50)
        print("Final Training Summary")
        print("="*50)
        print(f"Total Episodes: {self.training_config['num_episodes']}")
        print(f"Steps per Episode: {self.training_config.get('steps_per_episode', 200)}")
        print(f"Total Training Steps: {self.training_config['num_episodes'] * self.training_config.get('steps_per_episode', 200)}")
        
        # Calculate improvement percentages
        print("\nPerformance Improvements:")
        print("(Comparing average of last 5 episodes vs first 5 episodes)")
        metrics_improvement = {}
        for metric in all_metrics:
            if len(all_metrics[metric]) >= 10:  # Need at least 10 episodes for meaningful comparison
                # Get averages of first 5 and last 5 episodes
                first_5_avg = np.mean(all_metrics[metric][:5])
                last_5_avg = np.mean(all_metrics[metric][-5:])
                
                if first_5_avg != 0:  # Avoid division by zero
                    if metric in ['average_waiting_time', 'max_waiting_time']:
                        # For waiting times, decrease is improvement
                        improvement = ((first_5_avg - last_5_avg) / first_5_avg) * 100
                        print(f"{metric.replace('_', ' ').title()}:")
                        print(f"  First 5 episodes avg: {first_5_avg:.2f}")
                        print(f"  Last 5 episodes avg: {last_5_avg:.2f}")
                        print(f"  Change: {improvement:+.2f}% {'(improved)' if improvement > 0 else '(worse)'}")
                    else:
                        # For throughput and speed, increase is improvement
                        improvement = ((last_5_avg - first_5_avg) / first_5_avg) * 100
                        print(f"{metric.replace('_', ' ').title()}:")
                        print(f"  First 5 episodes avg: {first_5_avg:.2f}")
                        print(f"  Last 5 episodes avg: {last_5_avg:.2f}")
                        print(f"  Change: {improvement:+.2f}% {'(improved)' if improvement > 0 else '(worse)'}")
                    metrics_improvement[metric] = improvement
            else:
                print(f"{metric.replace('_', ' ').title()}: Not enough episodes for comparison")
        
        # Print best and worst episodes
        print("\nBest/Worst Episodes:")
        for metric in all_metrics:
            values = np.array(all_metrics[metric])
            if len(values) > 0:
                if metric in ['average_waiting_time', 'max_waiting_time']:
                    best_ep = np.argmin(values) + 1
                    worst_ep = np.argmax(values) + 1
                else:
                    best_ep = np.argmax(values) + 1
                    worst_ep = np.argmin(values) + 1
                print(f"{metric.replace('_', ' ').title()}:")
                print(f"  Best: Episode {best_ep} ({values[best_ep-1]:.2f})")
                print(f"  Worst: Episode {worst_ep} ({values[worst_ep-1]:.2f})")
        
        print("="*50)
        
        self.logger.info("Training completed")

    def update_networks(self, batch_size, step):
        """
        Updates the actor and critic networks for all agents.
        """
        losses = {
            'actor_loss': [],
            'critic_loss': [],
            'td_error': []
        }

        # Update each agent
        for agent in self.agents:
            agent_losses = agent.update(batch_size)
            if agent_losses:
                for k, v in agent_losses.items():
                    losses[k].append(v)

        # Calculate average losses across all agents
        averaged_losses = {k: np.mean(v) for k, v in losses.items() if v}
        self.logger.info(f"Step {step}: Actor Loss={averaged_losses.get('actor_loss', 0):.4f}, "
                         f"Critic Loss={averaged_losses.get('critic_loss', 0):.4f}, "
                         f"TD Error={averaged_losses.get('td_error', 0):.4f}")
        return averaged_losses

    def log_training_stats(self, step, losses):
        """
        Logs the training statistics to TensorBoard and console.
        """
        if losses:
            with self.train_summary_writer.as_default():
                for name, value in losses.items():
                    tf.summary.scalar(f'training/{name}', value, step=step)
            self.train_summary_writer.flush()
        
            # Additionally, log to console or file
            self.logger.info(f"Step {step}: Actor Loss={losses.get('actor_loss', 0):.4f}, "
                             f"Critic Loss={losses.get('critic_loss', 0):.4f}, "
                             f"TD Error={losses.get('td_error', 0):.4f}")

    def log_episode_stats(self, episode, reward):
        """
        Logs the episode statistics to TensorBoard and console.
        """
        with self.train_summary_writer.as_default():
            tf.summary.scalar('episode/mean_reward', reward, step=episode)
        
        self.logger.info(f'Episode {episode}: Mean Reward = {reward:.3f}')

    def save_checkpoints(self, episode):
        """
        Saves the checkpoints for all agents.
        """
        checkpoint_dir = f'checkpoints/casf/episode_{episode}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger.info(f"Saving checkpoints to {checkpoint_dir}")

        for i, agent in enumerate(self.agents):
            agent.save(os.path.join(checkpoint_dir, f'agent_{i}'))
        self.logger.info(f"Checkpoints saved for Episode {episode}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train CASF Multi-Agent System')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    trainer = MACSFTrainer(args.config)
    trainer.train()
