# experiments/scenarios/communication/shared_experience.py

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, deque
import random
import logging
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class SharedMemory:
    """Shared experience memory for cooperative learning"""
    
    def __init__(self,
                capacity: int,
                sharing_threshold: float = 0.5,
                importance_alpha: float = 0.6):
        """
        Initialize shared memory
        
        Args:
            capacity: Maximum number of transitions to store
            sharing_threshold: Minimum importance for experience sharing
            importance_alpha: Importance sampling exponent
        """
        self.capacity = capacity
        self.sharing_threshold = sharing_threshold
        self.importance_alpha = importance_alpha
        
        # Main memory buffers
        self.experiences = deque(maxlen=capacity)
        self.priorities = defaultdict(float)  # experience_id -> priority
        self.importances = defaultdict(float)  # experience_id -> importance
        
        # Agent specific buffers
        self.agent_memories = defaultdict(list)  # agent_id -> experiences
        self.agent_contributions = defaultdict(int)  # agent_id -> count
        
        # Shared experience tracking
        self.shared_experiences = set()  # experiences marked for sharing
        self.sharing_history = defaultdict(list)  # agent_id -> shared experiences
        
        # Performance metrics
        self.metrics = {
            'shared_ratio': [],  # Ratio of shared experiences
            'importance_dist': [],  # Distribution of importance values
            'agent_usage': defaultdict(list),  # Usage of experiences by agents
            'sharing_benefits': defaultdict(list)  # Benefits from sharing
        }
    
    def add_experience(self,
                      state: np.ndarray,
                      action: int,
                      reward: float,
                      next_state: np.ndarray,
                      done: bool,
                      agent_id: str,
                      importance: Optional[float] = None) -> str:
        """
        Add experience to memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
            agent_id: ID of contributing agent
            importance: Optional importance value
            
        Returns:
            Experience ID
        """
        # Generate unique experience ID
        exp_id = f"{agent_id}_{len(self.experiences)}"
        
        # Create experience tuple
        experience = (exp_id, state, action, reward, next_state, done, agent_id)
        
        # Add to main memory
        self.experiences.append(experience)
        
        # Add to agent's memory
        self.agent_memories[agent_id].append(experience)
        self.agent_contributions[agent_id] += 1
        
        # Calculate importance if not provided
        if importance is None:
            importance = self._calculate_importance(experience)
        
        self.importances[exp_id] = importance
        self.priorities[exp_id] = max(importance, 1e-6)  # Ensure non-zero priority
        
        # Mark for sharing if important enough
        if importance >= self.sharing_threshold:
            self.shared_experiences.add(exp_id)
        
        # Update metrics
        self._update_metrics()
        
        return exp_id
    
    def _calculate_importance(self, experience: Tuple) -> float:
        """
        Calculate importance value for experience
        
        Args:
            experience: Experience tuple
            
        Returns:
            Importance value (0-1)
        """
        _, state, action, reward, next_state, done, _ = experience
        
        # Factors that contribute to importance:
        
        # 1. Reward magnitude
        reward_importance = min(abs(reward) / 10.0, 1.0)  # Normalize reward
        
        # 2. State change magnitude
        state_diff = np.linalg.norm(next_state - state)
        state_importance = min(state_diff / 10.0, 1.0)  # Normalize state change
        
        # 3. Terminal state importance
        terminal_importance = 1.0 if done else 0.0
        
        # Combine factors
        importance = 0.4 * reward_importance + \
                    0.4 * state_importance + \
                    0.2 * terminal_importance
        
        return importance
    
    def get_shared_batch(self,
                        batch_size: int,
                        requesting_agent: str) -> List[Tuple]:
        """
        Get batch of shared experiences
        
        Args:
            batch_size: Number of experiences to retrieve
            requesting_agent: ID of requesting agent
            
        Returns:
            List of experience tuples
        """
        if not self.shared_experiences:
            return []
        
        # Get available shared experiences
        available_exps = [exp for exp in self.experiences 
                         if exp[0] in self.shared_experiences]
        
        if not available_exps:
            return []
        
        # Calculate sampling probabilities
        probs = np.array([self.priorities[exp[0]] ** self.importance_alpha 
                         for exp in available_exps])
        probs = probs / np.sum(probs)
        
        # Sample experiences
        batch_size = min(batch_size, len(available_exps))
        selected_indices = np.random.choice(
            len(available_exps), 
            size=batch_size,
            p=probs,
            replace=False
        )
        
        batch = [available_exps[i] for i in selected_indices]
        
        # Track usage
        self.metrics['agent_usage'][requesting_agent].extend(
            [exp[0] for exp in batch]
        )
        
        return batch
    
    def update_importance(self,
                        exp_id: str,
                        new_importance: float,
                        agent_id: str):
        """
        Update importance value for experience
        
        Args:
            exp_id: Experience ID
            new_importance: New importance value
            agent_id: Agent providing update
        """
        if exp_id in self.importances:
            # Update importance using exponential moving average
            alpha = 0.3  # Weight for new importance
            old_importance = self.importances[exp_id]
            updated_importance = (1 - alpha) * old_importance + alpha * new_importance
            
            self.importances[exp_id] = updated_importance
            self.priorities[exp_id] = max(updated_importance, 1e-6)
            
            # Update sharing status
            if updated_importance >= self.sharing_threshold:
                self.shared_experiences.add(exp_id)
            elif exp_id in self.shared_experiences:
                self.shared_experiences.remove(exp_id)
            
            # Track sharing benefit
            if agent_id in self.agent_contributions:
                self.metrics['sharing_benefits'][agent_id].append(
                    new_importance - old_importance
                )
    
    def _update_metrics(self):
        """Update performance metrics"""
        # Calculate shared ratio
        if self.experiences:
            shared_ratio = len(self.shared_experiences) / len(self.experiences)
            self.metrics['shared_ratio'].append(shared_ratio)
        
        # Update importance distribution
        if self.importances:
            self.metrics['importance_dist'].append(
                list(self.importances.values())
            )
    
    def get_statistics(self) -> Dict[str, Union[float, Dict]]:
        """Get memory statistics"""
        stats = {
            'total_experiences': len(self.experiences),
            'shared_experiences': len(self.shared_experiences),
            'avg_shared_ratio': np.mean(self.metrics['shared_ratio']) 
                              if self.metrics['shared_ratio'] else 0,
            'avg_importance': np.mean(list(self.importances.values()))
                           if self.importances else 0,
            'agent_contributions': dict(self.agent_contributions)
        }
        
        # Calculate sharing benefits per agent
        sharing_benefits = {}
        for agent_id, benefits in self.metrics['sharing_benefits'].items():
            if benefits:
                sharing_benefits[agent_id] = {
                    'mean': np.mean(benefits),
                    'total': np.sum(benefits),
                    'count': len(benefits)
                }
        
        stats['sharing_benefits'] = sharing_benefits
        
        return stats
    
    def get_agent_metrics(self, agent_id: str) -> Dict[str, float]:
        """Get metrics for specific agent"""
        if agent_id not in self.agent_contributions:
            return {}
        
        metrics = {
            'contributions': self.agent_contributions[agent_id],
            'memory_size': len(self.agent_memories[agent_id])
        }
        
        # Calculate sharing metrics
        usage = self.metrics['agent_usage'].get(agent_id, [])
        if usage:
            metrics.update({
                'shared_usage': len(usage),
                'unique_shared': len(set(usage))
            })
        
        # Calculate benefits
        benefits = self.metrics['sharing_benefits'].get(agent_id, [])
        if benefits:
            metrics.update({
                'avg_benefit': np.mean(benefits),
                'total_benefit': np.sum(benefits)
            })
        
        return metrics

# Part B

class ExperienceCoordinator:
    """Coordinate experience sharing between agents"""
    
    def __init__(self,
                similarity_threshold: float = 0.6,
                max_shared_batch: int = 32):
        """
        Initialize experience coordinator
        
        Args:
            similarity_threshold: Minimum similarity for experience sharing
            max_shared_batch: Maximum batch size for shared experiences
        """
        self.similarity_threshold = similarity_threshold
        self.max_shared_batch = max_shared_batch
        
        # Experience tracking
        self.agent_experiences = defaultdict(dict)  # agent_id -> {exp_id -> state}
        self.success_tracking = defaultdict(list)  # agent_id -> [(exp_id, success)]
        
        # State representations
        self.state_clusters = defaultdict(list)  # cluster_id -> [(exp_id, state)]
        self.next_cluster_id = 0
        
        # Metrics
        self.metrics = {
            'sharing_successes': defaultdict(list),  # agent_id -> success_rates
            'similarity_scores': [],
            'cluster_sizes': [],
            'transfer_stats': defaultdict(list)  # agent_id -> transfer_metrics
        }
    
    def register_experience(self,
                         exp_id: str,
                         state: np.ndarray,
                         agent_id: str):
        """
        Register new experience for potential sharing
        
        Args:
            exp_id: Experience ID
            state: State from experience
            agent_id: Contributing agent ID
        """
        # Store experience
        self.agent_experiences[agent_id][exp_id] = state
        
        # Find similar states and assign to cluster
        max_similarity = 0
        best_cluster = None
        
        for cluster_id, cluster_states in self.state_clusters.items():
            for _, cluster_state in cluster_states:
                similarity = self._calculate_similarity(state, cluster_state)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_cluster = cluster_id
        
        # Create new cluster if no similar states found
        if max_similarity < self.similarity_threshold:
            best_cluster = self.next_cluster_id
            self.next_cluster_id += 1
        
        # Add to cluster
        self.state_clusters[best_cluster].append((exp_id, state))
        
        # Update metrics
        self.metrics['similarity_scores'].append(max_similarity)
        self.metrics['cluster_sizes'].append(len(self.state_clusters[best_cluster]))
    
    def find_similar_experiences(self,
                              state: np.ndarray,
                              agent_id: str,
                              k: int = 10) -> List[str]:
        """
        Find similar experiences for an agent
        
        Args:
            state: Current state
            agent_id: Requesting agent ID
            k: Number of experiences to return
            
        Returns:
            List of experience IDs
        """
        similarities = []
        
        # Calculate similarities with all experiences except own
        for other_id, experiences in self.agent_experiences.items():
            if other_id != agent_id:
                for exp_id, exp_state in experiences.items():
                    similarity = self._calculate_similarity(state, exp_state)
                    similarities.append((exp_id, similarity))
        
        # Sort by similarity and filter
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_exps = [
            exp_id for exp_id, sim in similarities
            if sim >= self.similarity_threshold
        ][:k]
        
        return similar_exps
    
    def update_sharing_success(self,
                            exp_id: str,
                            agent_id: str,
                            success: float):
        """
        Update sharing success metrics
        
        Args:
            exp_id: Experience ID
            agent_id: Agent ID
            success: Success value (0-1)
        """
        self.success_tracking[agent_id].append((exp_id, success))
        self.metrics['sharing_successes'][agent_id].append(success)
        
        # Calculate transfer statistics
        if len(self.success_tracking[agent_id]) > 1:
            # Calculate improvement over time
            recent_successes = [s for _, s in self.success_tracking[agent_id][-10:]]
            self.metrics['transfer_stats'][agent_id].append(np.mean(recent_successes))
    
    def _calculate_similarity(self,
                           state1: np.ndarray,
                           state2: np.ndarray) -> float:
        """
        Calculate similarity between states
        
        Args:
            state1: First state
            state2: Second state
            
        Returns:
            Similarity score (0-1)
        """
        if state1.shape != state2.shape:
            return 0.0
        
        # Normalize states
        state1_norm = (state1 - np.mean(state1)) / (np.std(state1) + 1e-8)
        state2_norm = (state2 - np.mean(state2)) / (np.std(state2) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(state1_norm, state2_norm) / (
            np.linalg.norm(state1_norm) * np.linalg.norm(state2_norm) + 1e-8
        )
        
        return max(0.0, similarity)
    
    def get_statistics(self) -> Dict[str, Union[float, Dict]]:
        """Get coordinator statistics"""
        stats = {
            'num_clusters': len(self.state_clusters),
            'avg_cluster_size': np.mean(self.metrics['cluster_sizes'])
                             if self.metrics['cluster_sizes'] else 0,
            'avg_similarity': np.mean(self.metrics['similarity_scores'])
                           if self.metrics['similarity_scores'] else 0
        }
        
        # Calculate per-agent statistics
        agent_stats = {}
        for agent_id in self.success_tracking:
            successes = self.metrics['sharing_successes'][agent_id]
            transfer_stats = self.metrics['transfer_stats'][agent_id]
            
            if successes:
                agent_stats[agent_id] = {
                    'avg_success': np.mean(successes),
                    'num_shared': len(successes),
                    'improvement': np.mean(transfer_stats) if transfer_stats else 0
                }
        
        stats['agent_stats'] = agent_stats
        return stats

class SharedExperienceNetwork(nn.Module):
    """Neural network with shared experience handling"""
    
    def __init__(self,
                state_size: int,
                action_size: int,
                shared_size: int,
                hidden_size: int = 128):
        """
        Initialize network
        
        Args:
            state_size: Size of agent's state
            action_size: Size of action space
            shared_size: Size of shared experience embedding
            hidden_size: Size of hidden layers
        """
        super(SharedExperienceNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.shared_size = shared_size
        self.hidden_size = hidden_size
        
        # State processing
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Shared experience processing
        self.shared_encoder = nn.Sequential(
            nn.Linear(shared_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Attention mechanism for shared experiences
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Combined processing
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        # Training metrics
        self.updates = 0
        self.attention_weights = []
        self.losses = []
    
    def forward(self,
               state: torch.Tensor,
               shared_exp: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional shared experience
        
        Args:
            state: Agent state
            shared_exp: Optional tensor of shared experiences
            
        Returns:
            Q-values for each action
        """
        # Encode own state
        state_features = self.state_encoder(state)
        
        if shared_exp is None or shared_exp.numel() == 0:
            # No shared experience available or empty tensor
            combined = torch.cat([state_features, torch.zeros_like(state_features)], dim=-1)
        else:
            # Process shared experiences
            shared_features = self.shared_encoder(shared_exp)
            
            # Calculate attention weights
            combined_features = torch.cat([
                state_features.unsqueeze(1).repeat(1, shared_features.size(1), 1),
                shared_features
            ], dim=-1)
            
            attention_weights = self.attention(combined_features)
            
            # Store attention weights for analysis
            self.attention_weights.append(attention_weights.detach().mean().item())
            
            # Apply attention
            context = torch.sum(attention_weights * shared_features, dim=1)
            
            # Combine with state features
            combined = torch.cat([state_features, context], dim=-1)
        
        # Final processing
        q_values = self.combined_net(combined)
        
        return q_values
    
    def get_statistics(self) -> Dict[str, float]:
        """Get network statistics"""
        stats = {
            'num_updates': self.updates,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_attention': np.mean(self.attention_weights[-100:])
                          if self.attention_weights else 0
        }
        
        return stats