# experiments/train_offline.py

import os
import sys
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.agents.dqn_agent import DQNAgent
from src.utils.logger import Logger
from src.utils.sumo_data_parser import SUMODataParser

class OfflineTrainer:
    def __init__(self, data_dir: str, log_dir: str):
        self.data_parser = SUMODataParser(data_dir)
        self.logger = Logger(log_dir)
        
        # Get state and action space sizes
        self.state_size = self.data_parser.get_state_space_size()
        self.action_size = self.data_parser.get_action_space_size()
        
        # Create agents dictionary
        self.agents = {}
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.gamma = 0.99
        
    def create_agents(self, states_dict):
        """Create DQN agents for each traffic light."""
        for tls_id in states_dict.keys():
            self.agents[tls_id] = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                epsilon=0.0,  # No exploration needed for offline training
                epsilon_min=0.0,
                epsilon_decay=1.0,
                memory_size=10000,
                batch_size=self.batch_size
            )
    
    def prepare_training_data(self):
        """Prepare training data from historical SUMO data."""
        print("Parsing SUMO data...")
        states, rewards = self.data_parser.create_training_dataset()
        return states, rewards
    
    def train_epoch(self, states_dict, rewards_dict, epoch):
        """Train all agents for one epoch."""
        epoch_losses = []
        
        for tls_id, agent in self.agents.items():
            states = states_dict[tls_id]
            rewards = rewards_dict[tls_id]
            
            # Skip if not enough data
            if len(states) < 2:
                continue
            
            # Prepare training data
            for i in range(len(states) - 1):
                state = states[i]
                next_state = states[i + 1]
                reward = rewards[i]
                
                # Get action (we'll use the actual action taken as target)
                action = np.argmax(agent.model(
                    torch.FloatTensor(state).unsqueeze(0)
                ).detach().numpy())
                
                # Store transition
                agent.store_transition(state, action, reward, next_state, False)
            
            # Train if enough samples
            if len(agent.memory) > agent.batch_size:
                loss = agent.train()
                epoch_losses.append(loss)
        
        # Log metrics
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        self.logger.log_episode({
            'epoch': epoch,
            'average_loss': avg_loss,
        })
        
        return avg_loss
    
    def train(self, num_epochs=100):
        """Train agents using historical data."""
        print("Starting offline training...")
        
        # Prepare training data
        states_dict, rewards_dict = self.prepare_training_data()
        
        # Create agents
        self.create_agents(states_dict)
        
        # Training loop
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(states_dict, rewards_dict, epoch)
            
            # Save models periodically
            if epoch % 10 == 0:
                self.save_models(epoch)
                
            print(f"Epoch {epoch}/{num_epochs} - Average Loss: {avg_loss:.4f}")
    
    def save_models(self, epoch):
        """Save all agent models."""
        save_dir = Path("models") / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for tls_id, agent in self.agents.items():
            save_path = save_dir / f"agent_{tls_id}.pth"
            agent.save_model(str(save_path))

def main():
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/offline_training_{timestamp}"
    data_dir = "TEST_CASE"
    
    # Create trainer and start training
    trainer = OfflineTrainer(data_dir, log_dir)
    trainer.train(num_epochs=100)

if __name__ == "__main__":
    main()