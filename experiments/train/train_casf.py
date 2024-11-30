# experiments/train/train_casf.py

import argparse
import yaml
import logging
import numpy as np
from src.environment.multi_agent_sumo_env_casf import MACSFEnvironment
from src.agents.casf import CASFAgent
import os
import tensorflow as tf
from datetime import datetime
import traci

from evaluation_sets.metrics import average_waiting_time, total_throughput, average_speed, max_waiting_time  # Import metrics
from tqdm import tqdm  # Import tqdm for progress bars

class MACSFTrainer:
    def __init__(self, config_path):
        # Initialize logger
        self.logger = self.setup_logger()
        self.logger.info("Initializing MACSFTrainer")

        # Load configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Extract configuration sections
        self.training_config = config.get('training', {})
        self.model_config = config.get('model', {})
        self.env_config = config.get('environment', {})
        self.memory_config = config.get('memory', {})
        self.logging_config = config.get('logging', {})

        # Initialize environment
        self.env = MACSFEnvironment(config=self.env_config)

        # Ensure action_dim and state_dim are in model_config
        self.model_config['action_dim'] = self.env.action_space.shape[0]
        self.model_config['state_dim'] = self.env.observation_space.shape[0]

        # Initialize agents
        self.agents = {}
        agent_ids = self.env.agent_ids  # Use agent IDs from the environment
        for agent_id in agent_ids:
            agent = CASFAgent(
                agent_id=agent_id,
                state_dim=self.model_config['state_dim'],
                action_dim=self.model_config['action_dim'],
                network_config=self.model_config,
                memory_config=self.memory_config,
                max_neighbors=self.env_config.get('max_neighbors', 4),
                logger=self.logger
            )
            self.agents[agent_id] = agent
            self.logger.info(f"Initialized CASFAgent {agent_id}")

        # Setup TensorBoard
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join('logs', 'casf', current_time, 'train')
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.logger.info(f"TensorBoard logging set up at {log_dir}")

        # Initialize other training parameters
        self.num_episodes = self.training_config.get('num_episodes', 1000)
        self.steps_per_episode = self.training_config.get('steps_per_episode', 200)
        self.batch_size = self.training_config.get('batch_size', 128)
        self.gamma = self.training_config.get('gamma', 0.95)
        self.tau = self.training_config.get('tau', 0.01)
        self.update_every = self.training_config.get('update_every', 10)
        self.checkpoint_interval = self.training_config.get('checkpoint_interval', 50)

        # Initialize checkpoint manager
        self.checkpoint_dir = os.path.join('checkpoints', 'casf', current_time)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint = tf.train.Checkpoint(agents=self.agents)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=5)

        # Initialize metrics storage
        self.metrics_history = []

    def setup_logger(self):
        logger = logging.getLogger('MACSFTrainer')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler('train_casf.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def train(self):
        self.logger.info("Starting training")
        for episode in tqdm(range(1, self.num_episodes + 1), desc="Episodes"):
            states, neighbor_states = self.env.reset()
            total_rewards = {agent_id: 0 for agent_id in self.agents}
            episode_metrics = {}
            for step in tqdm(range(1, self.steps_per_episode + 1), desc="Steps", leave=False):
                actions = []
                agent_ids = list(self.agents.keys())
                for idx, agent_id in enumerate(agent_ids):
                    agent = self.agents[agent_id]
                    action = agent.act(states[idx], neighbor_states[idx], noise=0.1)
                    actions.append(action)

                next_states, next_neighbor_states, rewards, dones, info = self.env.step(actions)

                for idx, agent_id in enumerate(agent_ids):
                    agent = self.agents[agent_id]
                    # Ensure consistent shapes for states and neighbor states
                    state = np.array(states[idx]).reshape(self.model_config['state_dim'])
                    next_state = np.array(next_states[idx]).reshape(self.model_config['state_dim'])
                    neighbor_next_state = np.array(next_neighbor_states[idx]).reshape(self.env_config.get('max_neighbors', 4), self.model_config['state_dim'])

                    # Placeholder for neighbor_next_action
                    neighbor_next_action = np.zeros((self.env_config.get('max_neighbors', 4), self.model_config['action_dim']), dtype=np.float32)

                    agent.store_experience(
                        state=state,
                        action=actions[idx],
                        reward=rewards[idx],
                        next_state=next_state,
                        neighbor_next_state=neighbor_next_state,
                        neighbor_next_action=neighbor_next_action,  # Placeholder
                        done=dones[idx]
                    )
                    total_rewards[agent_id] += rewards[idx]

                states = next_states
                neighbor_states = next_neighbor_states

                # Learn every update_every steps
                if step % self.update_every == 0:
                    for agent in self.agents.values():
                        agent.learn()

            # Collect vehicle data and compute metrics
            vehicle_data = self.env.get_vehicle_data()
            avg_wait_time = average_waiting_time(vehicle_data)
            total_thruput = total_throughput(vehicle_data)
            avg_speed = average_speed(vehicle_data)
            max_wait_time = max_waiting_time(vehicle_data)

            # Store metrics for final summary
            episode_metrics['average_waiting_time'] = avg_wait_time
            episode_metrics['total_throughput'] = total_thruput
            episode_metrics['average_speed'] = avg_speed
            episode_metrics['max_waiting_time'] = max_wait_time
            self.metrics_history.append(episode_metrics)

            # Log episode metrics
            with self.summary_writer.as_default():
                for agent_id, reward in total_rewards.items():
                    tf.summary.scalar(f'reward/{agent_id}', reward, step=episode)
                tf.summary.scalar('average_waiting_time', avg_wait_time, step=episode)
                tf.summary.scalar('total_throughput', total_thruput, step=episode)
                tf.summary.scalar('average_speed', avg_speed, step=episode)
                tf.summary.scalar('max_waiting_time', max_wait_time, step=episode)

            self.logger.info(f"Episode {episode}/{self.num_episodes} - Total Rewards: {total_rewards}")
            self.logger.info(f"Episode {episode} Metrics: Average Waiting Time: {avg_wait_time}, Total Throughput: {total_thruput}, Average Speed: {avg_speed}, Max Waiting Time: {max_wait_time}")

            # Save checkpoints
            if episode % self.checkpoint_interval == 0:
                saved_checkpoint = self.checkpoint_manager.save()
                self.logger.info(f"Saved checkpoint: {saved_checkpoint}")

        # Final summary
        total_episodes = len(self.metrics_history)
        avg_wait_times = [m['average_waiting_time'] for m in self.metrics_history]
        total_thruputs = [m['total_throughput'] for m in self.metrics_history]
        avg_speeds = [m['average_speed'] for m in self.metrics_history]
        max_wait_times = [m['max_waiting_time'] for m in self.metrics_history]

        final_metrics = {
            'average_waiting_time': sum(avg_wait_times) / total_episodes,
            'total_throughput': sum(total_thruputs),
            'average_speed': sum(avg_speeds) / total_episodes,
            'max_waiting_time': max(max_wait_times)
        }

        self.logger.info("Training completed")
        self.logger.info(f"Final Metrics over {total_episodes} episodes:")
        self.logger.info(f"Average Waiting Time: {final_metrics['average_waiting_time']}")
        self.logger.info(f"Total Throughput: {final_metrics['total_throughput']}")
        self.logger.info(f"Average Speed: {final_metrics['average_speed']}")
        self.logger.info(f"Max Waiting Time: {final_metrics['max_waiting_time']}")

        self.env.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CASF Agents")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    args = parser.parse_args()

    trainer = MACSFTrainer(config_path=args.config)
    trainer.train()