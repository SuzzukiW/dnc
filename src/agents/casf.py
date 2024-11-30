# src/agents/casf.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from src.models.casf_network import CASFNetwork
from src.utils.prioritized_replay_buffer_casf import PrioritizedReplayBuffer
import logging
from src.utils.logger import get_logger


class CASFAgent:
    def __init__(self, agent_id, state_dim, action_dim, network_config, memory_config, max_neighbors, logger=None):
        """
        Initializes the CASF Agent.

        Args:
            agent_id (str): Unique identifier for the agent.
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            network_config (dict): Configuration for the neural network.
            memory_config (dict): Configuration for the replay buffer.
            max_neighbors (int): Maximum number of neighbors.
            logger (logging.Logger, optional): Logger instance. Defaults to None.
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_neighbors = max_neighbors

        # Initialize logger
        self.logger = logger if logger else get_logger(f'CASFAgent_{agent_id}', level=logging.INFO)
        self.logger.info(f"Initializing CASFAgent {agent_id}")

        # Update network_config with action_dim and state_dim
        network_config['action_dim'] = self.action_dim
        network_config['state_dim'] = self.state_dim

        # Initialize Actor-Critic Network
        self.network = CASFNetwork(config=network_config)
        self.target_network = CASFNetwork(config=network_config)
        self.update_target_network(tau=1.0)  # Initialize target network weights

        # Initialize Optimizers
        self.actor_optimizer = Adam(learning_rate=network_config.get('learning_rate', 0.001))
        self.critic_optimizer = Adam(learning_rate=network_config.get('learning_rate', 0.001))

        # Initialize Replay Buffer
        self.memory = PrioritizedReplayBuffer(
            size=memory_config['capacity'],
            alpha=memory_config['alpha'],
            state_dim=state_dim,
            action_dim=action_dim,
            max_neighbors=max_neighbors,
            beta_start=memory_config.get('beta_start', 0.4),
            beta_end=memory_config.get('beta_end', 1.0),
            beta_steps=memory_config.get('beta_steps', 100000)
        )

        self.gamma = memory_config.get('gamma', 0.95)
        self.tau = memory_config.get('tau', 0.01)
        self.batch_size = memory_config.get('batch_size', 128)
        self.grad_clip = network_config.get('grad_clip', 0.5)

        self.logger.info(f"CASFAgent {agent_id} initialized successfully")

    def update_target_network(self, tau=None):
        """
        Updates the target network parameters.

        Args:
            tau (float, optional): Soft update parameter. Defaults to self.tau.
        """
        tau = self.tau if tau is None else tau
        target_weights = self.target_network.get_weights()
        weights = self.network.get_weights()
        new_weights = []
        for target, main in zip(target_weights, weights):
            new_weights.append(tau * main + (1 - tau) * target)
        self.target_network.set_weights(new_weights)
        self.logger.debug(f"Target network for agent {self.agent_id} updated with tau={tau}")

    def act(self, state, neighbor_states, noise=0.0):
        """
        Selects an action based on the current state and neighbor states.

        Args:
            state (np.ndarray): Current state.
            neighbor_states (np.ndarray): Neighbor states.
            noise (float, optional): Noise parameter for exploration. Defaults to 0.0.

        Returns:
            np.ndarray: Selected action.
        """
        state = np.expand_dims(state, axis=0)  # Shape: (1, state_dim)
        neighbor_states = np.expand_dims(neighbor_states, axis=0)  # Shape: (1, max_neighbors, state_dim)

        action, _ = self.network([state, neighbor_states])
        action = tf.squeeze(action, axis=0).numpy()

        if noise > 0.0:
            action += np.random.normal(0, noise, size=self.action_dim)

        action = np.clip(action, -1.0, 1.0)
        self.logger.debug(f"Agent {self.agent_id} acting with action: {action}")
        return action

    def store_experience(self, state, action, reward, next_state, neighbor_next_state, neighbor_next_action, done, td_error=None):
        """
        Stores an experience in the replay buffer.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            neighbor_next_state (np.ndarray): Next states of neighbors.
            neighbor_next_action (np.ndarray): Next actions of neighbors.
            done (bool): Whether the episode is done.
            td_error (float, optional): TD-error for priority update. If None, uses max priority.
        """
        # Ensure all states are numpy arrays with consistent shapes
        state = np.array(state, dtype=np.float32).reshape(self.state_dim)
        next_state = np.array(next_state, dtype=np.float32).reshape(self.state_dim)
        neighbor_next_state = np.array(neighbor_next_state, dtype=np.float32).reshape(self.max_neighbors, self.state_dim)
        neighbor_next_action = np.array(neighbor_next_action, dtype=np.float32).reshape(self.max_neighbors, self.action_dim)
        action = np.array(action, dtype=np.float32).reshape(self.action_dim)

        if td_error is None:
            td_error = self.memory.max_priority

        self.memory.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            neighbor_next_state=neighbor_next_state,
            neighbor_next_action=neighbor_next_action,
            done=done,
            td_error=td_error
        )
        self.logger.debug(f"Agent {self.agent_id} stored experience with TD-error: {td_error}")

    def learn(self):
        """
        Learns from a batch of experiences sampled from the replay buffer.
        """
        if len(self.memory) < self.batch_size:
            self.logger.info(f"Agent {self.agent_id}: Not enough samples to learn. Required: {self.batch_size}, Available: {len(self.memory)}")
            return

        (
            states,
            actions,
            rewards,
            next_states,
            neighbor_next_states,
            neighbor_next_actions,
            dones,
            indices,
            weights
        ) = self.memory.sample(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        neighbor_next_states = tf.convert_to_tensor(neighbor_next_states, dtype=tf.float32)
        neighbor_next_actions = tf.convert_to_tensor(neighbor_next_actions, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)

        # Critic update
        with tf.GradientTape() as tape:
            # Current Q-values
            _, critic_value = self.network([states, neighbor_next_states, actions])
            critic_value = tf.squeeze(critic_value, axis=1)

            # Target actions and Q-values
            target_actions, _ = self.target_network([next_states, neighbor_next_states])
            _, target_critic_value = self.target_network([next_states, neighbor_next_states, target_actions])
            target_critic_value = tf.squeeze(target_critic_value, axis=1)

            # Compute target Q-values
            target = rewards + (1 - dones) * self.gamma * target_critic_value
            target = tf.stop_gradient(target)

            # Compute TD-errors
            td_errors = target - critic_value

            # Compute loss
            critic_loss = tf.reduce_mean(weights * tf.square(td_errors))

        # Compute gradients and update critic network
        critic_grad = tape.gradient(critic_loss, self.network.critic_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.network.critic_variables))

        # Actor update
        with tf.GradientTape() as tape:
            # Compute actor loss
            actions_pred, _ = self.network([states, neighbor_next_states])
            _, critic_value = self.network([states, neighbor_next_states, actions_pred])
            actor_loss = -tf.reduce_mean(critic_value)

        # Compute gradients and update actor network
        actor_grad = tape.gradient(actor_loss, self.network.actor_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.network.actor_variables))

        # Update target networks
        self.update_target_network()

        # Update priorities in replay buffer
        td_errors_np = td_errors.numpy()
        self.memory.update_priorities(indices, td_errors_np)

        self.logger.debug(f"Agent {self.agent_id} updated networks. Critic loss: {critic_loss.numpy()}, Actor loss: {actor_loss.numpy()}")

    def save_model(self, filepath):
        """
        Saves the actor and critic models.

        Args:
            filepath (str): Directory path to save the models.
        """
        self.network.save(filepath)
        self.logger.info(f"Agent {self.agent_id} saved models to {filepath}")

    def load_model(self, filepath):
        """
        Loads the actor and critic models.

        Args:
            filepath (str): Directory path from where to load the models.
        """
        self.network = tf.keras.models.load_model(filepath)
        self.target_network = tf.keras.models.load_model(filepath)
        self.logger.info(f"Agent {self.agent_id} loaded models from {filepath}")