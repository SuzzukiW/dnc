# src/agents/maddpg_agent.py

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import os
import torch.nn.functional as F

from src.models.maddpg_network import Actor, Critic
from src.utils.prioritized_replay_buffer_maddpg import PrioritizedReplayBuffer
from src.utils.noise import OUNoise
from src.utils.state_preprocessor import StatePreprocessor
from src.utils.state_dimensionality_reducer import StateDimensionalityReducer

class MADDPGAgent:
    """MADDPG Agent with dynamic action size handling."""
    
    def __init__(self, config_path: str, agent_id: str, observation_space, action_space, max_action_size: int):
        """Initialize a MADDPG agent with flexible state and action handling."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.agent_id = agent_id
        
        # Flexible state and action size handling
        self.original_state_size = observation_space.shape[0]
        self.action_size = action_space.shape[0]
        self.max_action_size = max_action_size
        
        # Adaptive state preprocessing
        self.state_preprocessor = self._create_state_preprocessor(
            input_size=self.original_state_size, 
            config=config
        )
        
        # Compute preprocessed state size
        sample_state = torch.randn(1, self.original_state_size)
        self.preprocessed_state_size = self.state_preprocessor(sample_state).size(1)
        
        # Number of agents from config
        self.num_agents = config['environment']['num_agents']
        
        # Neural Networks with flexible input sizes
        self.actor = Actor(
            state_size=self.preprocessed_state_size, 
            action_size=self.action_size, 
            hidden_layers=config['network']['actor']['hidden_layers']
        )
        self.actor_target = Actor(
            state_size=self.preprocessed_state_size, 
            action_size=self.action_size, 
            hidden_layers=config['network']['actor']['hidden_layers']
        )
        
        # Critic takes full state and action from all agents
        critic_state_size = self.preprocessed_state_size * self.num_agents
        critic_action_size = self.max_action_size * self.num_agents
        
        self.critic = Critic(
            state_size=critic_state_size, 
            action_size=critic_action_size, 
            hidden_layers=config['network']['critic']['hidden_layers']
        )
        self.critic_target = Critic(
            state_size=critic_state_size, 
            action_size=critic_action_size, 
            hidden_layers=config['network']['critic']['hidden_layers']
        )
        
        # Copy weights
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['training']['learning_rate_actor'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['training']['learning_rate_critic'])
        
        # Move networks to device
        self.to(self.device)
        
        # Replay memory
        memory_config = config['memory']
        self.memory = PrioritizedReplayBuffer(
            memory_config['capacity'], 
            memory_config.get('alpha', 0.6),
            self.preprocessed_state_size * self.num_agents, 
            self.max_action_size * self.num_agents,
            memory_config['max_neighbors']
        )
        
        # Noise process
        self.noise = OUNoise(
            size=self.action_size, 
            mu=config['noise'].get('mu', 0.0), 
            theta=config['noise'].get('theta', 0.15), 
            sigma=config['noise'].get('sigma', 0.2)
        )
        
        # Hyperparameters
        self.gamma = config['training']['gamma']
        self.tau = config['training']['tau']
        self.batch_size = config['training']['batch_size']
        self.update_every = config['training'].get('update_every', 1)
        self.step_count = 0
    
    def _create_state_preprocessor(self, input_size: int, config: Dict):
        """
        Create a flexible state preprocessor that can handle varying input sizes.
        
        Args:
            input_size: Original state vector size
            config: Configuration dictionary
        
        Returns:
            A neural network module for state preprocessing
        """
        # Use a more robust state preprocessor with batch normalization handling
        preprocessor = StateDimensionalityReducer(
            input_size=input_size, 
            output_size=config.get('state_preprocessor', {}).get('output_size', 32),
            reduction_method=config.get('state_preprocessor', {}).get('method', 'adaptive')
        )
        
        # Ensure the preprocessor can handle single sample scenarios
        preprocessor.eval()  # Set to evaluation mode to handle single sample
        return preprocessor

    def preprocess_state(self, state: torch.Tensor):
        # Ensure input is a tensor
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        # Add batch dimension if missing
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Flatten if more than 2 dimensions
        if state.dim() > 2:
            state = state.view(state.size(0), -1)
        
        # Ensure tensor is float
        state = state.float()
        
        # Truncate or pad to match input size
        if state.size(1) > self.original_state_size:
            state = state[:, :self.original_state_size]
        elif state.size(1) < self.original_state_size:
            padding = torch.zeros(
                state.size(0), 
                self.original_state_size - state.size(1), 
                device=state.device,
                dtype=state.dtype
            )
            state = torch.cat([state, padding], dim=1)
        
        # Preprocess with dimensionality reducer
        # Ensure at least 2 samples for BatchNorm
        if state.size(0) == 1:
            # Duplicate the single sample
            state = state.repeat(2, 1)
            preprocessed_state = self.state_preprocessor(state)
            preprocessed_state = preprocessed_state[:1]  # Return only the first tensor
        else:
            preprocessed_state = self.state_preprocessor(state)
        
        return preprocessed_state
    
    def to(self, device):
        """Move agent's networks to specified device."""
        self.state_preprocessor = self.state_preprocessor.to(device)
        self.actor = self.actor.to(device)
        self.actor_target = self.actor_target.to(device)
        self.critic = self.critic.to(device)
        self.critic_target = self.critic_target.to(device)
        return self
    
    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Get actions from actor network."""
        # Convert state to tensor and preprocess
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        preprocessed_state = self.preprocess_state(state_tensor)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(preprocessed_state.to(self.device)).cpu().data.numpy()
        self.actor.train()
        
        if add_noise:
            action += self.noise.sample()
            action = np.clip(action, -1, 1)
        
        return action.squeeze()
    
    def step(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool, other_agents_states: List[np.ndarray],
            other_agents_actions: List[np.ndarray], other_agents_next_states: List[np.ndarray]):
        """
        Store experience and potentially learn from it.
        
        Handles state and action preprocessing for experiences with varying dimensions.
        """
        # Process states and store multi-agent information
        preprocessed_state = self.preprocess_state(torch.FloatTensor(state).unsqueeze(0))  # Shape: [1, preprocessed_size]
        preprocessed_next_state = self.preprocess_state(torch.FloatTensor(next_state).unsqueeze(0))  # Shape: [1, preprocessed_size]
        
        # Process other agents' states
        processed_other_states = [
            self.preprocess_state(torch.FloatTensor(other_state).unsqueeze(0)).detach().cpu().numpy() 
            for other_state in other_agents_states
        ]
        processed_other_next_states = [
            self.preprocess_state(torch.FloatTensor(other_next_state).unsqueeze(0)).detach().cpu().numpy() 
            for other_next_state in other_agents_next_states
        ]
        
        # Ensure all states have consistent shapes before concatenation
        preprocessed_state_np = preprocessed_state.detach().cpu().numpy()
        preprocessed_next_state_np = preprocessed_next_state.detach().cpu().numpy()
        
        # Reshape all states to have the same dimensions
        all_states = [preprocessed_state_np] + processed_other_states
        all_next_states = [preprocessed_next_state_np] + processed_other_next_states
        
        # Ensure all states have the same shape by padding if necessary
        max_state_dim = max(state.shape[-1] for state in all_states)
        padded_states = []
        padded_next_states = []
        
        for state in all_states:
            if state.shape[-1] < max_state_dim:
                padded_state = np.pad(
                    state,
                    ((0, 0), (0, max_state_dim - state.shape[-1])),
                    mode='constant'
                )
                padded_states.append(padded_state)
            else:
                padded_states.append(state)
                
        for state in all_next_states:
            if state.shape[-1] < max_state_dim:
                padded_state = np.pad(
                    state,
                    ((0, 0), (0, max_state_dim - state.shape[-1])),
                    mode='constant'
                )
                padded_next_states.append(padded_state)
            else:
                padded_next_states.append(state)
        
        # Concatenate all states for full state representation
        full_state = np.concatenate(padded_states, axis=1)  # Shape: [1, total_state_size]
        full_next_state = np.concatenate(padded_next_states, axis=1)  # Shape: [1, total_state_size]
        
        # Pad and concatenate actions
        padded_action = np.pad(action, (0, self.max_action_size - len(action)), mode='constant')
        padded_other_actions = [
            np.pad(a, (0, self.max_action_size - len(a)), mode='constant')
            for a in other_agents_actions
        ]
        full_action = np.concatenate([padded_action] + padded_other_actions)
        
        # Calculate TD error for prioritized replay
        with torch.no_grad():
            # Convert to tensors with correct dimensions
            state_tensor = torch.FloatTensor(full_state).to(self.device)  # Shape: [1, state_size]
            next_state_tensor = torch.FloatTensor(full_next_state).to(self.device)  # Shape: [1, state_size]
            action_tensor = torch.FloatTensor(full_action).to(self.device).unsqueeze(0)  # Shape: [1, action_size]
            
            current_Q = self.critic(state_tensor, action_tensor)
            next_actions = []
            for i, agent_next_state in enumerate([next_state] + other_agents_next_states):
                if i == 0:
                    # Preprocess the state before passing to actor
                    processed_next_state = self.preprocess_state(torch.FloatTensor(agent_next_state).unsqueeze(0))
                    next_action = self.actor_target(processed_next_state.to(self.device))
                else:
                    # For simplicity, use current agent's target network for other agents
                    processed_next_state = self.preprocess_state(torch.FloatTensor(agent_next_state).unsqueeze(0))
                    next_action = self.actor_target(processed_next_state.to(self.device))
                padded_next_action = torch.nn.functional.pad(
                    next_action, 
                    (0, self.max_action_size - next_action.size(-1)), 
                    mode='constant'
                )
                next_actions.append(padded_next_action)
            
            # Concatenate and ensure correct dimensions
            next_actions = torch.cat(next_actions, dim=1)  # Shape: [1, total_action_size]
            target_Q = reward + (self.gamma * self.critic_target(next_state_tensor, next_actions) * (1 - done))
            td_error = abs(float(target_Q - current_Q))
            
            # Apply very strict TD error control
            td_error = min(td_error, 1.0)  # Clip to [0, 1]
            td_error = 0.1 + 0.9 * td_error  # Ensure minimum priority of 0.1

        # Add a check to ensure the replay buffer has enough samples
        if len(self.memory) >= self.batch_size:
            # Clip the TD error to prevent extreme priority values
            clipped_td_error = np.clip(np.abs(td_error), 1e-6, 100.0)
            
            self.memory.add(
                full_state,
                full_action,
                reward,
                full_next_state,
                float(done),
                1.0,  # Default priority
                None,  # No attention states needed
                clipped_td_error  # Add controlled TD error for prioritized replay
            )

            # Perform learning step only when buffer has enough samples
            if self.step_count % self.update_every == 0:
                experiences = self.memory.sample(self.batch_size)
                if experiences is not None:
                    self.learn(experiences)

        self.step_count += 1
    
    def learn(self, experiences: Tuple):
        """Update policy and value parameters using given batch of experience tuples."""
        # Skip if no experiences provided
        if experiences is None:
            return
            
        # Handle different experience tuple formats
        if len(experiences) == 5:  # Basic format: (states, actions, rewards, next_states, dones)
            states, actions, rewards, next_states, dones = experiences
            indices = None
            weights = torch.ones(states.shape[0], 1).to(self.device)  # Default weights
        elif len(experiences) == 7:  # Format with priorities: (states, actions, rewards, next_states, dones, indices, weights)
            states, actions, rewards, next_states, dones, indices, weights = experiences
        elif len(experiences) == 9:  # Extended format: (states, actions, rewards, next_states, dones, indices, weights, attention_states, td_errors)
            states, actions, rewards, next_states, dones, indices, weights, _, _ = experiences
        else:
            raise ValueError(f"Unexpected number of values in experiences tuple: {len(experiences)}")
        
        # Convert numpy arrays to tensors and move to device
        states = torch.FloatTensor(np.array(states)).to(self.device)  # Shape: [batch_size, num_agents * state_size]
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)  # Shape: [batch_size, num_agents * state_size]
        actions = torch.FloatTensor(np.array(actions)).to(self.device)  # Shape: [batch_size, num_agents * action_size]
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(-1).to(self.device)  # Shape: [batch_size, 1]
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(-1).to(self.device)  # Shape: [batch_size, 1]
        
        if weights is not None:
            # Convert weights to float32 array first
            weights = np.array(weights, dtype=np.float32)
            weights = torch.FloatTensor(weights).unsqueeze(-1).to(self.device)  # Shape: [batch_size, 1]
        else:
            weights = torch.ones_like(rewards).to(self.device)
        
        # Get dimensions
        batch_size = states.size(0)
        num_agents = self.num_agents
        state_size = states.size(-1) // num_agents
        action_size = actions.size(-1) // num_agents

        # Reshape states and actions to have consistent dimensions
        states = states.view(batch_size, num_agents, state_size)
        next_states = next_states.view(batch_size, num_agents, state_size)
        actions = actions.view(batch_size, num_agents, action_size)

        # Get next actions for each agent using their respective target networks
        next_actions = []
        for i in range(num_agents):
            agent_next_state = next_states[:, i, :]
            next_action = self.actor_target(agent_next_state)
            # Pad if necessary
            if next_action.size(-1) < self.max_action_size:
                next_action = torch.nn.functional.pad(
                    next_action,
                    (0, self.max_action_size - next_action.size(-1)),
                    mode='constant'
                )
            next_actions.append(next_action)
        next_actions = torch.cat(next_actions, dim=1)

        # Flatten states and actions for critic input
        states_flat = states.view(batch_size, -1)  # Shape: (batch_size, num_agents * state_size)
        next_states_flat = next_states.view(batch_size, -1)  # Shape: (batch_size, num_agents * state_size)
        actions_flat = actions.view(batch_size, -1)  # Shape: (batch_size, num_agents * action_size)

        # Compute target Q value
        target_q_next = self.critic_target(next_states_flat, next_actions)
        target_rewards = rewards + (self.gamma * target_q_next * (1 - dones))

        # Compute current Q value
        current_q_values = self.critic(states_flat, actions_flat)
        
        # Compute TD errors for updating priorities
        td_errors = torch.abs(target_rewards - current_q_values).mean(dim=1).detach().cpu().numpy()
        
        # Update priorities in replay buffer if indices are provided
        if indices is not None:
            # Apply very strict TD error control
            td_errors = np.clip(td_errors, 0, 1.0)  # Clip to [0, 1]
            td_errors = 0.1 + 0.9 * td_errors  # Ensure minimum priority of 0.1
            
            # Convert indices and td_errors to numpy arrays with correct types
            indices = np.asarray(indices, dtype=np.int32)
            td_errors = np.asarray(td_errors, dtype=np.float32)
            self.memory.update_priorities(indices, td_errors)
        
        # Compute critic loss with importance sampling weights
        critic_loss = F.mse_loss(current_q_values, target_rewards, reduction='none')
        critic_loss = (critic_loss * weights).mean()
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Compute actor loss
        # Get current agent's actions
        current_actions = self.actor(states[:, 0, :])
        
        # Pad actions if necessary
        if current_actions.size(-1) < self.max_action_size:
            current_actions = torch.nn.functional.pad(
                current_actions,
                (0, self.max_action_size - current_actions.size(-1)),
                mode='constant'
            )
        
        # Create full actions tensor with padded actions for all agents
        actions_for_critic = []
        for i in range(self.num_agents):
            if i == 0:
                actions_for_critic.append(current_actions)
            else:
                agent_action = actions[:, i, :]
                if agent_action.size(-1) < self.max_action_size:
                    agent_action = torch.nn.functional.pad(
                        agent_action,
                        (0, self.max_action_size - agent_action.size(-1)),
                        mode='constant'
                    )
                actions_for_critic.append(agent_action)
        actions_for_critic = torch.cat(actions_for_critic, dim=1)
        
        # Compute actor loss using properly formatted actions
        actor_loss = -self.critic(states_flat, actions_for_critic).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)
    
    def soft_update(self, local_model: torch.nn.Module, target_model: torch.nn.Module):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def reset(self):
        """Reset the noise process."""
        self.noise.reset()
        
    def save(self, path: str):
        """Save the agent's networks to files.
        
        Args:
            path: Path to save the model files
        """
        os.makedirs(path, exist_ok=True)
        torch.save(
            self.actor.state_dict(),
            os.path.join(path, f'actor_{self.agent_id}.pth')
        )
        torch.save(
            self.actor_target.state_dict(),
            os.path.join(path, f'actor_target_{self.agent_id}.pth')
        )
        torch.save(
            self.critic.state_dict(),
            os.path.join(path, f'critic_{self.agent_id}.pth')
        )
        torch.save(
            self.critic_target.state_dict(),
            os.path.join(path, f'critic_target_{self.agent_id}.pth')
        )
    
    def load(self, path: str):
        """Load the agent's networks from files.
        
        Args:
            path: Path to the saved model files
        """
        self.actor.load_state_dict(
            torch.load(os.path.join(path, f'actor_{self.agent_id}.pth'))
        )
        self.actor_target.load_state_dict(
            torch.load(os.path.join(path, f'actor_target_{self.agent_id}.pth'))
        )
        self.critic.load_state_dict(
            torch.load(os.path.join(path, f'critic_{self.agent_id}.pth'))
        )
        self.critic_target.load_state_dict(
            torch.load(os.path.join(path, f'critic_target_{self.agent_id}.pth'))
        )