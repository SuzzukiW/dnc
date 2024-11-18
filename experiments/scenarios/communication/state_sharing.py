import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any

class StateSharedAgent:
    """
    Multi-agent reinforcement learning agent with state sharing capabilities
    for traffic light control in a grid network.
    """
    def __init__(self, 
                 state_dim: int = 10, 
                 action_dim: int = 4, 
                 learning_rate: float = 0.001,
                 communication_type: str = 'full'):
        """
        Initialize the state-shared agent with different communication strategies.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.communication_type = communication_type
        
        # Neural network for Q-learning
        self.q_network = self._build_neural_network()
        self.target_network = self._build_neural_network()
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Shared memory for state communication
        self.shared_memory = {}
        
        # Performance tracking
        self.total_reward = 0
        self.episodes_completed = 0

    def _build_neural_network(self) -> nn.Module:
        """
        Build a neural network for Q-learning with state representation.
        """
        return nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """
        Select an action using epsilon-greedy strategy.
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update(self, 
               state: np.ndarray, 
               action: int, 
               reward: float, 
               next_state: np.ndarray,
               done: bool):
        """
        Update the Q-network using experience replay.
        """
        # Track total reward
        self.total_reward += reward
        
        if done:
            self.episodes_completed += 1
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        
        # Current Q-value
        current_q = self.q_network(state_tensor)[action]
        
        # Compute target Q-value
        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensor)
            max_next_q = next_q_values.max()
            target_q = reward + (1 - done) * 0.99 * max_next_q
        
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def simulate_traffic_scenario(agents: List[StateSharedAgent], 
                               num_episodes: int = 100, 
                               max_steps: int = 1000) -> Dict[str, float]:
    """
    Simulate a traffic scenario with multiple agents.
    
    Args:
        agents (List[StateSharedAgent]): List of agents to simulate
        num_episodes (int): Number of episodes to run
        max_steps (int): Maximum steps per episode
    
    Returns:
        Dict[str, float]: Performance metrics
    """
    for episode in range(num_episodes):
        # Initialize episode states
        states = [np.random.rand(10) for _ in agents]
        
        for step in range(max_steps):
            # Select actions for each agent
            actions = [agent.select_action(state) for agent, state in zip(agents, states)]
            
            # Simulate rewards and next states (simplified)
            rewards = [
                np.random.normal(loc=0, scale=1) * (1 - abs(action - 2)/2) 
                for action in actions
            ]
            
            # Simulate next states (random walk)
            next_states = [
                state + np.random.normal(loc=0, scale=0.1, size=state.shape) 
                for state in states
            ]
            
            # Update agents
            for i, (agent, state, action, reward, next_state) in enumerate(
                zip(agents, states, actions, rewards, next_states)
            ):
                done = step == max_steps - 1
                agent.update(state, action, reward, next_state, done)
                
                # Optional: Share states based on communication type
                if agent.communication_type != 'none':
                    agent.shared_memory[f'agent_{i}'] = state
            
            states = next_states
    
    # Compute performance metrics
    metrics = {
        'avg_reward': np.mean([agent.total_reward / max(1, agent.episodes_completed) for agent in agents]),
        'total_throughput': num_episodes * max_steps,
        'avg_wait_time': np.mean([1 / max(0.1, agent.total_reward) for agent in agents])
    }
    
    return metrics

def run_state_sharing_experiment(
    num_agents: int = 4, 
    communication_types: List[str] = ['none', 'local', 'full'],
    num_episodes: int = 100,
    max_steps: int = 1000
) -> Dict[str, Any]:
    """
    Run experiments with different state sharing strategies.
    """
    results = {}
    
    for comm_type in communication_types:
        # Initialize agents with specific communication strategy
        agents = [
            StateSharedAgent(
                state_dim=10,  # Example state dimension 
                action_dim=4,  # Example action dimension
                communication_type=comm_type
            ) for _ in range(num_agents)
        ]
        
        # Run traffic simulation
        performance_metrics = simulate_traffic_scenario(
            agents, 
            num_episodes=num_episodes, 
            max_steps=max_steps
        )
        
        results[comm_type] = performance_metrics
    
    return results

if __name__ == "__main__":
    # Run the state sharing experiment
    experiment_results = run_state_sharing_experiment()
    print("State Sharing Experiment Results:")
    for comm_type, metrics in experiment_results.items():
        print(f"\nCommunication Type: {comm_type}")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
