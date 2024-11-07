# experiments/scenarios/communication/shared_experience.py

import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
import random

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.agents.dqn_agent import DQNAgent, DQNNetwork
from src.environment.multi_agent_sumo_env import MultiAgentSumoEnvironment
from src.utils.logger import setup_logger

class SharedReplayBuffer:
    """Shared experience replay buffer for all agents"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done, agent_id):
        """Save experience to shared memory"""
        self.memory.append((state, action, reward, next_state, done, agent_id))
    
    def sample(self, batch_size):
        """Sample random batch from shared memory"""
        return random.sample(list(self.memory), batch_size)
    
    def __len__(self):
        return len(self.memory)

class SharedExperienceAgent(DQNAgent):
    """DQN agent with shared experience replay"""
    def __init__(self, state_size, action_size, config, shared_memory, agent_id):
        super().__init__(state_size, action_size, config)
        self.shared_memory = shared_memory
        self.agent_id = agent_id
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in shared memory"""
        self.shared_memory.push(state, action, reward, next_state, done, self.agent_id)
    
    def replay(self):
        """Train on batch from shared memory"""
        if len(self.shared_memory) < self.batch_size:
            return
        
        batch = self.shared_memory.sample(self.batch_size)
        states = torch.FloatTensor([x[0] for x in batch]).to(self.device)
        actions = torch.LongTensor([x[1] for x in batch]).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        next_states = torch.FloatTensor([x[3] for x in batch]).to(self.device)
        dones = torch.FloatTensor([x[4] for x in batch]).to(self.device)
        
        # Compute Q(s_t, a)
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = self.target_net(next_states).max(1)[0].detach()
        
        # Compute expected Q values
        expected_q = rewards + self.gamma * next_state_values * (1 - dones)
        
        # Compute loss
        loss = torch.nn.MSELoss()(current_q, expected_q.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

class SharedExperienceScenario:
    """
    Shared Experience scenario - agents learn from shared experiences
    All agents contribute to and learn from a common experience pool
    """
    def __init__(self, env_config, agent_config):
        self.env = MultiAgentSumoEnvironment(**env_config)
        self.traffic_lights = self.env.traffic_lights
        
        # Create shared replay buffer
        self.shared_memory = SharedReplayBuffer(agent_config.get('memory_size', 100000))
        
        # Create agents with shared memory
        self.agents = {}
        for tl_id in self.traffic_lights:
            state_size = self.env.observation_spaces[tl_id].shape[0]
            action_size = self.env.action_spaces[tl_id].n
            self.agents[tl_id] = SharedExperienceAgent(
                state_size=state_size,
                action_size=action_size,
                config=agent_config,
                shared_memory=self.shared_memory,
                agent_id=tl_id
            )
    
    def train(self, num_episodes, log_dir):
        """Train agents with shared experience"""
        logger = setup_logger("shared_experience_training", log_dir / "training.log")
        metrics = defaultdict(list)
        
        for episode in range(num_episodes):
            states, _ = self.env.reset()
            episode_rewards = defaultdict(float)
            done = False
            
            while not done:
                # Each agent selects action
                actions = {}
                for tl_id in self.traffic_lights:
                    actions[tl_id] = self.agents[tl_id].act(states[tl_id])
                
                # Environment step
                next_states, rewards, done, _, info = self.env.step(actions)
                
                # Store experiences in shared memory
                for tl_id in self.traffic_lights:
                    self.agents[tl_id].remember(
                        states[tl_id],
                        actions[tl_id],
                        rewards[tl_id],
                        next_states[tl_id],
                        done
                    )
                    
                    # Learning step
                    if len(self.shared_memory) > self.agents[tl_id].batch_size:
                        loss = self.agents[tl_id].replay()
                        metrics[f'loss_{tl_id}'].append(loss)
                    
                    episode_rewards[tl_id] += rewards[tl_id]
                
                states = next_states
            
            # Update target networks periodically
            if episode % 10 == 0:
                for agent in self.agents.values():
                    agent.update_target_network()
            
            # Log episode results
            mean_reward = np.mean([r for r in episode_rewards.values()])
            metrics['mean_rewards'].append(mean_reward)
            metrics['global_rewards'].append(info['global_reward'])
            metrics['shared_memory_size'].append(len(self.shared_memory))
            
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            logger.info(f"Mean reward: {mean_reward:.2f}")
            logger.info(f"Global reward: {info['global_reward']:.2f}")
            logger.info(f"Shared memory size: {len(self.shared_memory)}")
            
            # Save models periodically
            if (episode + 1) % 100 == 0:
                self.save_agents(log_dir / f"models_episode_{episode + 1}")
        
        return metrics
    
    def save_agents(self, save_dir):
        """Save all agents"""
        save_dir.mkdir(parents=True, exist_ok=True)
        for tl_id, agent in self.agents.items():
            agent.save(save_dir / f"agent_{tl_id}.pt")
    
    def load_agents(self, load_dir):
        """Load all agents"""
        for tl_id, agent in self.agents.items():
            agent.load(load_dir / f"agent_{tl_id}.pt")
    
    def close(self):
        """Clean up environment"""
        self.env.close()

def run_shared_experience_scenario(env_config, agent_config, num_episodes=1000):
    """Run shared experience scenario"""
    # Setup logging directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path("experiments/logs/shared_experience") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configurations
    with open(log_dir / "env_config.yaml", 'w') as f:
        yaml.dump(env_config, f)
    with open(log_dir / "agent_config.yaml", 'w') as f:
        yaml.dump(agent_config, f)
    
    # Run training
    scenario = SharedExperienceScenario(env_config, agent_config)
    try:
        metrics = scenario.train(num_episodes, log_dir)
        # Save final models and metrics
        scenario.save_agents(log_dir / "final_models")
        return metrics
    finally:
        scenario.close()

if __name__ == "__main__":
    # Load configurations
    with open("config/env_config.yaml", 'r') as f:
        env_config = yaml.safe_load(f)
    with open("config/agent_config.yaml", 'r') as f:
        agent_config = yaml.safe_load(f)
    
    # Run scenario
    metrics = run_shared_experience_scenario(env_config, agent_config)
    print("Shared experience training completed!")