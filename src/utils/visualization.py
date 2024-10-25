# src/utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_metrics(metrics, save_path=None):
    """Plot evaluation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Traffic Control Evaluation Metrics')
    
    # Plot rewards
    axes[0,0].plot(metrics['rewards'])
    axes[0,0].set_title('Total Rewards per Episode')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Reward')
    
    # Plot waiting times
    axes[0,1].plot(metrics['waiting_times'])
    axes[0,1].set_title('Average Waiting Time per Episode')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Waiting Time (s)')
    
    # Plot queue lengths
    axes[1,0].plot(metrics['queue_lengths'])
    axes[1,0].set_title('Average Queue Length per Episode')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Queue Length')
    
    # Plot emissions
    axes[1,1].plot(metrics['emissions'])
    axes[1,1].set_title('Average CO2 Emissions per Episode')
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('CO2 Emissions (mg/s)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved metrics plot to {save_path}")
    
    plt.show()

def plot_training_progress(logger, save_path=None):
    """Plot training progress metrics"""
    metrics = logger.get_metrics()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress')
    
    # Plot average reward
    axes[0,0].plot(metrics['episode'], metrics['avg_reward'])
    axes[0,0].set_title('Average Reward per Episode')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Reward')
    
    # Plot epsilon
    axes[0,1].plot(metrics['episode'], metrics['epsilon'])
    axes[0,1].set_title('Exploration Rate (Epsilon)')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Epsilon')
    
    # Plot loss
    axes[1,0].plot(metrics['episode'], metrics['loss'])
    axes[1,0].set_title('Training Loss')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Loss')
    
    # Plot average waiting time
    axes[1,1].plot(metrics['episode'], metrics['avg_waiting_time'])
    axes[1,1].set_title('Average Waiting Time')
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Time (s)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved training progress plot to {save_path}")
    
    plt.show()