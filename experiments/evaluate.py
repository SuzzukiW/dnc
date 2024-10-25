# experiments/evaluate.py

import os
import yaml
import torch
import numpy as np
from pathlib import Path
from src.environment.sumo_env import SumoEnvironment
from src.agents.dqn_agent import DQNAgent
from src.utils.visualization import plot_metrics

def evaluate_agent(env, agents, num_episodes=5):
    """Evaluate trained agents"""
    metrics = {
        'rewards': [],
        'waiting_times': [],
        'queue_lengths': [],
        'emissions': []
    }
    
    for episode in range(num_episodes):
        states = env.reset()
        episode_rewards = {tl_id: 0 for tl_id in env.traffic_lights}
        episode_metrics = []
        
        done = False
        while not done:
            # Select actions (no exploration during evaluation)
            actions = {}
            for tl_id in env.traffic_lights:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(states[tl_id]).unsqueeze(0)
                    q_values = agents[tl_id].policy_net(state_tensor)
                    actions[tl_id] = q_values.argmax().item()
            
            # Execute actions
            next_states, rewards, dones, _ = env.step(actions)
            
            # Update episode rewards
            for tl_id in env.traffic_lights:
                episode_rewards[tl_id] += rewards[tl_id]
            
            # Collect metrics
            current_metrics = env.data_collector.collected_data['traffic_metrics'][-1]
            episode_metrics.append(current_metrics)
            
            # Update states and check if done
            states = next_states
            done = any(dones.values())
        
        # Store episode metrics
        metrics['rewards'].append(sum(episode_rewards.values()))
        metrics['waiting_times'].append(np.mean([m['total_waiting'] for m in episode_metrics]))
        metrics['queue_lengths'].append(np.mean([m['vehicle_count'] for m in episode_metrics]))
        metrics['emissions'].append(np.mean([m.get('total_co2', 0) for m in episode_metrics]))
        
        print(f"Episode {episode + 1}/{num_episodes}:")
        print(f"Total Reward: {metrics['rewards'][-1]:.2f}")
        print(f"Average Waiting Time: {metrics['waiting_times'][-1]:.2f}")
        print(f"Average Queue Length: {metrics['queue_lengths'][-1]:.2f}")
        print(f"Average CO2 Emissions: {metrics['emissions'][-1]:.2f}")
        print("---")
    
    return metrics

def main():
    # Load configurations
    with open('config/env_config.yaml', 'r') as f:
        env_config = yaml.safe_load(f)
    with open('config/agent_config.yaml', 'r') as f:
        agent_config = yaml.safe_load(f)
    
    # Create environment
    env = SumoEnvironment(
        config_file=env_config['sumo']['config_file'],
        use_gui=True,  # Always use GUI for evaluation
        num_seconds=env_config['sumo']['simulation_steps']
    )
    
    # Load trained agents
    agents = {}
    checkpoint_dir = Path(agent_config['checkpoint']['save_dir'])
    
    for tl_id in env.traffic_lights:
        state_dim = env.observation_spaces[tl_id].shape[0]
        action_dim = env.action_spaces[tl_id].n
        
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **agent_config['dqn']
        )
        
        # Load the latest checkpoint for this agent
        checkpoints = list(checkpoint_dir.glob(f'dqn_agent_{tl_id}_episode_*.pt'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(str(x).split('_')[-1].split('.')[0]))
            agent.load(str(latest_checkpoint))
            print(f"Loaded checkpoint for agent {tl_id}: {latest_checkpoint}")
        
        agents[tl_id] = agent
    
    # Evaluate agents
    print("Starting evaluation...")
    metrics = evaluate_agent(env, agents)
    
    # Plot and save results
    plot_metrics(metrics, save_path='results/evaluation_metrics.png')
    
    env.close()

if __name__ == '__main__':
    main()