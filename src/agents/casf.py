# src/agents/casf.py

import numpy as np
import tensorflow as tf
from src.utils.logger import get_logger
from src.utils.replay_buffer import PrioritizedReplayBuffer  # Ensure this import exists
import logging


class CASFAgent:
    def __init__(self, agent_id, state_dim, action_dim, network_config, memory_config):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize logger with INFO level
        self.logger = get_logger(f'CASFAgent_{self.agent_id}', level=logging.INFO)
        self.logger.info(f"Initializing CASFAgent {self.agent_id}")

        # Initialize network configurations
        self.n_heads = network_config['num_attention_heads']
        self.key_dim = network_config['key_dim']
        self.value_dim = network_config['value_dim']

        # Assign network_config to self.config
        self.config = network_config
        self.logger.info("Assigned network_config to self.config")

        # Initialize the MultiHeadAttention layer
        self.attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=self.key_dim,
            value_dim=self.value_dim
        )
        self.logger.info("Initialized MultiHeadAttention layer")

        # Build actor and critic networks
        self.actor_local = self._build_actor(network_config)
        self.actor_target = self._build_actor(network_config)
        self.critic_local = self._build_critic(network_config)
        self.critic_target = self._build_critic(network_config)

        # Copy weights from local to target networks
        self.actor_target.set_weights(self.actor_local.get_weights())
        self.critic_target.set_weights(self.critic_local.get_weights())

        # Initialize memory with increased capacity
        self.memory = PrioritizedReplayBuffer(
            capacity=memory_config['capacity'],  # Set to 200000 in config
            alpha=memory_config['alpha'],
            beta=memory_config['beta'],
            epsilon=memory_config['epsilon']
        )

        # Training parameters
        self.gamma = network_config.get('gamma', 0.95)
        self.tau = network_config.get('tau', 0.01)

        # Optimizers with gradient clipping
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=network_config['learning_rate'],
            clipvalue=network_config['grad_clip']
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=network_config['learning_rate'],
            clipvalue=network_config['grad_clip']
        )

        self.logger.info(f"CASFAgent {self.agent_id} initialized successfully")

        # Initialize variables for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.epsilon = 1e-8  # To prevent division by zero

    def _build_actor(self, config):
        """
        Constructs the actor network with enhanced architecture.
        """
        state_input = tf.keras.layers.Input(shape=(self.state_dim,), name='state_input')
        neighbor_states = tf.keras.layers.Input(shape=(config['max_neighbors'], self.state_dim), name='neighbor_states')

        # State encoder with increased depth and dropout
        x = self._build_state_encoder(state_input, config)
        x = tf.keras.layers.Dropout(0.3)(x)  # Added Dropout

        # Encode neighbor states
        neighbor_encodings = self._build_state_encoder(neighbor_states, config)
        neighbor_encodings = tf.keras.layers.Dropout(0.3)(neighbor_encodings)  # Added Dropout

        # Multi-head attention fusion with layer normalization
        fused_state = self._build_attention_fusion(x, neighbor_encodings, config)
        fused_state = tf.keras.layers.LayerNormalization()(fused_state)  # Added LayerNormalization

        # Policy head with additional layers
        x = tf.keras.layers.Dense(config['actor_hidden_units'][0], activation='leaky_relu')(fused_state)  # Changed to LeakyReLU
        x = tf.keras.layers.Dropout(0.3)(x)  # Added Dropout
        x = tf.keras.layers.Dense(config['actor_hidden_units'][1], activation='leaky_relu')(x)  # Changed to LeakyReLU
        x = tf.keras.layers.Dropout(0.3)(x)  # Added Dropout
        actions = tf.keras.layers.Dense(self.action_dim, activation='tanh', name='actions')(x)  # action_dim=1

        return tf.keras.Model(inputs=[state_input, neighbor_states], outputs=actions, name=f'Actor_{self.agent_id}')

    def _build_critic(self, config):
        """
        Constructs the critic network with enhanced architecture.
        """
        state_input = tf.keras.layers.Input(shape=(self.state_dim,), name='state_input')
        action_input = tf.keras.layers.Input(shape=(self.action_dim,), name='action_input')
        neighbor_states = tf.keras.layers.Input(shape=(config['max_neighbors'], self.state_dim), name='neighbor_states')
        neighbor_actions = tf.keras.layers.Input(shape=(config['max_neighbors'], self.action_dim), name='neighbor_actions')

        # Encode states and actions
        state_encoding = self._build_state_encoder(state_input, config)
        state_encoding = tf.keras.layers.Dropout(0.3)(state_encoding)  # Added Dropout
        neighbor_encodings = self._build_state_encoder(neighbor_states, config)
        neighbor_encodings = tf.keras.layers.Dropout(0.3)(neighbor_encodings)  # Added Dropout

        # Fuse states with attention and layer normalization
        fused_state = self._build_attention_fusion(state_encoding, neighbor_encodings, config)
        fused_state = tf.keras.layers.LayerNormalization()(fused_state)  # Added LayerNormalization

        # Combine with actions
        x = tf.keras.layers.Concatenate(name='concat')([fused_state, action_input])

        # Value head with increased depth and dropout
        x = tf.keras.layers.Dense(config['critic_hidden_units'][0], activation='leaky_relu')(x)  # Changed to LeakyReLU
        x = tf.keras.layers.Dropout(0.3)(x)  # Added Dropout
        x = tf.keras.layers.Dense(config['critic_hidden_units'][1], activation='leaky_relu')(x)  # Changed to LeakyReLU
        x = tf.keras.layers.Dropout(0.3)(x)  # Added Dropout
        q_value = tf.keras.layers.Dense(1, name='q_value')(x)

        return tf.keras.Model(
            inputs=[state_input, action_input, neighbor_states, neighbor_actions],
            outputs=q_value,
            name=f'Critic_{self.agent_id}'
        )

    def _build_state_encoder(self, inputs, config):
        """
        Builds the state encoder part of the network with increased depth and dropout.
        """
        x = tf.keras.layers.Dense(config['state_encoder_units'][0], activation='leaky_relu')(inputs)  # Changed to LeakyReLU
        x = tf.keras.layers.Dropout(0.3)(x)  # Added Dropout
        x = tf.keras.layers.Dense(config['state_encoder_units'][1], activation='leaky_relu')(x)  # Changed to LeakyReLU
        x = tf.keras.layers.Dropout(0.3)(x)  # Added Dropout
        return tf.keras.layers.Dense(config['state_embedding_dim'], activation='relu')(x)

    def _build_attention_fusion(self, query, keys, config):
        """
        Applies multi-head attention to fuse query and keys with improved stability.
        """
        # Expand query to have temporal dimension
        query_expanded = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1), name='query_expand')(query)

        # Apply MultiHeadAttention
        attn_output = self.attention_layer(query=query_expanded, key=keys, value=keys)

        # Squeeze the temporal dimension
        attn_output_squeezed = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1), name='attn_squeeze')(attn_output)

        return attn_output_squeezed

    def act(self, state, neighbor_states, training=False):
        """
        Selects an action based on the current state and neighbor states.
        """
        if neighbor_states is None:
            # Create dummy neighbor states if not provided
            neighbor_states = np.zeros((self.config['max_neighbors'], self.state_dim))  # Adjust as per max_neighbors

        state = tf.convert_to_tensor([state], dtype=tf.float32)
        neighbor_states = tf.convert_to_tensor([neighbor_states], dtype=tf.float32)

        actions = self.actor_local([state, neighbor_states])
        # Extract the scalar from the action array
        action = np.clip(actions[0][0], -1, 1)

        return action

    def store_experience(self, state, action, reward, next_state, neighbor_next_state, neighbor_next_action, done):
        """
        Stores experience in the replay buffer.
        """
        self.memory.add(state, action, reward, next_state, neighbor_next_state, neighbor_next_action, done)

    def update(self, batch_size):
        """
        Updates the actor and critic networks based on sampled experiences.
        Returns the losses for logging.
        """
        if len(self.memory) < batch_size:
            self.logger.info(f"Agent {self.agent_id}: Not enough samples to update. Required: {batch_size}, Available: {len(self.memory)}")
            return None

        # Sample batch
        sample = self.memory.sample(batch_size)
        if sample is None:
            self.logger.info(f"Agent {self.agent_id}: Memory sample returned None")
            return None
        states, actions, rewards, next_states, neighbor_next_states, neighbor_next_actions, dones, weights, indices = sample

        # Normalize rewards
        rewards = self._normalize_rewards(rewards)

        # Convert all inputs to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        neighbor_next_states = tf.convert_to_tensor(neighbor_next_states, dtype=tf.float32)
        neighbor_next_actions = tf.convert_to_tensor(neighbor_next_actions, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # Predict next actions using actor_target with both next_states and neighbor_next_states
            next_actions = self.actor_target([next_states, neighbor_next_states])

            # Get Q values for next states
            q_targets_next = self.critic_target([next_states, next_actions, neighbor_next_states, neighbor_next_actions])

            # Compute Q targets
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

            # Get expected Q values
            q_expected = self.critic_local([states, actions, neighbor_next_states, neighbor_next_actions])

            # Compute critic loss using Huber loss
            critic_loss = tf.reduce_mean(weights * tf.keras.losses.Huber()(q_targets, q_expected))

        # Update critic
        critic_grads = tape.gradient(critic_loss, self.critic_local.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_local.trainable_variables))

        with tf.GradientTape() as tape:
            # Predict actions using actor_local with both states and neighbor_states
            actions_pred = self.actor_local([states, neighbor_next_states])
            # Compute actor loss
            actor_loss = -tf.reduce_mean(self.critic_local([states, actions_pred, neighbor_next_states, neighbor_next_actions]))

        # Update actor
        actor_grads = tape.gradient(actor_loss, self.actor_local.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_local.trainable_variables))

        # Update target networks
        self._update_target_network(self.actor_target, self.actor_local)
        self._update_target_network(self.critic_target, self.critic_local)

        # Update priorities in replay buffer
        td_errors = q_targets - q_expected
        # Flatten td_errors to shape (batch_size,)
        td_errors = tf.reshape(td_errors, (-1,)).numpy()
        self.memory.update_priorities(indices, td_errors)

        self.logger.info(f"Agent {self.agent_id} - Updated networks. Actor Loss: {actor_loss.numpy():.4f}, "
                         f"Critic Loss: {critic_loss.numpy():.4f}, TD Error: {np.mean(np.abs(td_errors)):.4f}")

        return {
            'actor_loss': actor_loss.numpy(),
            'critic_loss': critic_loss.numpy(),
            'td_error': np.mean(np.abs(td_errors))
        }

    def _normalize_rewards(self, rewards):
        """
        Normalizes rewards to have zero mean and unit variance.
        """
        mean = np.mean(rewards)
        std = np.std(rewards)
        if std < self.epsilon:
            std = self.epsilon
        normalized_rewards = (rewards - mean) / (std + self.epsilon)
        return normalized_rewards

    def _update_target_network(self, target, source):
        """
        Performs a soft update of target network parameters.
        """
        for target_var, source_var in zip(target.trainable_variables, source.trainable_variables):
            target_var.assign(self.tau * source_var + (1 - self.tau) * target_var)

    def save(self, path):
        """
        Saves the weights of the actor and critic networks.
        The filenames must end with '.weights.h5' as per Keras requirements.
        """
        actor_path = f'{path}_actor.weights.h5'
        critic_path = f'{path}_critic.weights.h5'

        self.actor_local.save_weights(actor_path)
        self.critic_local.save_weights(critic_path)
        self.logger.info(f"Agent {self.agent_id} - Saved actor weights to {actor_path}")
        self.logger.info(f"Agent {self.agent_id} - Saved critic weights to {critic_path}")

    def load(self, path):
        """
        Loads the weights of the actor and critic networks.
        The filenames must end with '.weights.h5' as per Keras requirements.
        """
        actor_path = f'{path}_actor.weights.h5'
        critic_path = f'{path}_critic.weights.h5'

        self.actor_local.load_weights(actor_path)
        self.critic_local.load_weights(critic_path)
        self.actor_target.set_weights(self.actor_local.get_weights())
        self.critic_target.set_weights(self.critic_local.get_weights())
        self.logger.info(f"Agent {self.agent_id} - Loaded actor weights from {actor_path}")
        self.logger.info(f"Agent {self.agent_id} - Loaded critic weights from {critic_path}")
