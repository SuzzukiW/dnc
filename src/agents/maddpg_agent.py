import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.models.maddpg_network import MADDPGNetwork
from src.utils.ou_noise import OUNoise

class MADDPGAgent:
    """Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Agent"""
    
    def __init__(self, state_size, action_size, num_agents, random_seed, 
                 hidden_size=256, lr_actor=1e-4, lr_critic=1e-3, 
                 weight_decay=0, gamma=0.99, tau=1e-3):
        """Initialize a MADDPG Agent
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents in the environment
            random_seed (int): random seed
            hidden_size (int): number of nodes in hidden layers
            lr_actor (float): learning rate for actor
            lr_critic (float): learning rate for critic
            weight_decay (float): L2 weight decay
            gamma (float): discount factor
            tau (float): soft update of target parameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = torch.manual_seed(random_seed)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        
        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        
        # Actor Network (w/ Target Network)
        self.actor_local = MADDPGNetwork(state_size, action_size, hidden_size, num_agents)
        self.actor_target = MADDPGNetwork(state_size, action_size, hidden_size, num_agents)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Critic Network (w/ Target Network)
        self.critic_local = MADDPGNetwork(state_size * num_agents, action_size * num_agents, 
                                          hidden_size, num_agents, critic=True)
        self.critic_target = MADDPGNetwork(state_size * num_agents, action_size * num_agents, 
                                           hidden_size, num_agents, critic=True)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), 
                                           lr=lr_critic, weight_decay=weight_decay)
        
        # Hard copy weights from local to target networks
        self.soft_update(self.actor_local, self.actor_target, 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)
    
    def act(self, states, add_noise=True):
        """Returns actions for given states as per current policy."""
        states = torch.from_numpy(states).float()
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            noise = self.noise.sample()
            actions = np.clip(actions + noise, -1, 1)
        
        return actions
    
    def reset(self):
        """Reset the noise process"""
        self.noise.reset()
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def learn(self, experiences, agent_number):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            agent_number (int): index of agent being updated
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Convert to torch tensors if not already
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Compute actions for all agents using target actor networks
        next_actions = torch.zeros_like(actions)
        for i in range(self.num_agents):
            next_actions[:, i, :] = self.actor_target(next_states[:, i, :])
        
        # Flatten states and actions for critic
        critic_states = next_states.view(next_states.size(0), -1)
        critic_actions = next_actions.view(next_actions.size(0), -1)
        
        # Get predicted next-state Q values from target models
        Q_targets_next = self.critic_target(torch.cat((critic_states, critic_actions), dim=1))
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards[:, agent_number].unsqueeze(1) + \
                    self.gamma * Q_targets_next * (1 - dones[:, agent_number].unsqueeze(1))
        
        # Compute critic loss
        critic_states_current = states.view(states.size(0), -1)
        critic_actions_current = actions.view(actions.size(0), -1)
        Q_expected = self.critic_local(torch.cat((critic_states_current, critic_actions_current), dim=1))
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        pred_actions = torch.zeros_like(actions)
        for i in range(self.num_agents):
            pred_actions[:, i, :] = self.actor_local(states[:, i, :])
        
        pred_actions_flat = pred_actions.view(pred_actions.size(0), -1)
        actor_loss = -self.critic_local(torch.cat((critic_states_current, pred_actions_flat), dim=1)).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)
