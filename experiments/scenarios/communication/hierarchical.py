# experiments/scenarios/communication/hierarchical.py

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import random
import logging
import networkx as nx
import sumolib
import json

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise ValueError("Please declare environment variable 'SUMO_HOME'")

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.utils import setup_logger

class Message:
    """Represents a communication message between agents"""
    
    TYPES = {
        'STATE_UPDATE': 1,     # Share state information
        'ACTION_PLAN': 2,      # Share planned actions
        'COORDINATION': 3,     # Coordination requests/responses
        'RECOMMENDATION': 4,   # Regional recommendations
        'ALERT': 5            # Critical situation alerts
    }
    
    def __init__(self,
                msg_type: int,
                sender: str,
                content: dict,
                priority: float = 1.0,
                timestamp: Optional[float] = None):
        """
        Initialize message
        
        Args:
            msg_type: Type of message (from TYPES)
            sender: ID of sending agent
            content: Message content dictionary
            priority: Message priority (0-1)
            timestamp: Message creation time
        """
        self.type = msg_type
        self.sender = sender
        self.content = content
        self.priority = priority
        self.timestamp = timestamp or traci.simulation.getTime()
        
        # Track message handling
        self.processed = False
        self.response_to = None
        self.responses = []

class RegionalCoordinator:
    """Manages coordination between traffic lights in a region"""
    
    def __init__(self,
                region_id: str,
                member_agents: List[str],
                state_size: int,
                action_size: int,
                config: dict):
        """
        Initialize regional coordinator
        
        Args:
            region_id: Unique identifier for the region
            member_agents: List of agent IDs in this region
            state_size: Size of individual agent state
            action_size: Size of agent action space
            config: Configuration dictionary
        """
        self.region_id = region_id
        self.member_agents = set(member_agents)
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Coordination parameters
        self.influence_radius = config.get('influence_radius', 200.0)
        self.coordination_interval = config.get('coordination_interval', 5)
        self.max_agents = config.get('max_region_size', 10)
        
        # State tracking
        self.agent_states = {}
        self.agent_positions = {}
        self.last_actions = defaultdict(int)
        
        # Regional metrics
        self.metrics = {
            'congestion_levels': defaultdict(list),
            'waiting_times': defaultdict(list),
            'coordination_scores': defaultdict(list),
            'throughput': defaultdict(list)
        }
        
        # Communication tracking
        self.messages = defaultdict(list)
        self.message_history = defaultdict(list)
        
        # Setup logging
        self.logger = logging.getLogger(f'Region_{region_id}')
        self.logger.setLevel(logging.INFO)
    
    def process_messages(self):
        """Process messages in queue and update regional state"""
        # Process each type of message
        for msg_type in Message.TYPES.values():
            # Sort messages of this type by priority and timestamp
            self.messages[msg_type] = sorted(
                self.messages[msg_type],
                key=lambda x: (-x.priority, x.timestamp)
            )
            
            # Process each message of this type
            for message in self.messages[msg_type]:
                if message.processed:
                    continue
                    
                # Process based on message type
                if msg_type == Message.TYPES['STATE_UPDATE']:
                    self._process_state_update(message)
                elif msg_type == Message.TYPES['ACTION_PLAN']:
                    self._process_action_plan(message)
                elif msg_type == Message.TYPES['COORDINATION']:
                    self._process_coordination(message)
                
                # Mark as processed and store in history
                message.processed = True
                self.message_history[msg_type].append(message)
        
        # Clear processed messages
        for msg_type in Message.TYPES.values():
            self.messages[msg_type] = [
                m for m in self.messages[msg_type] 
                if not m.processed
            ]
    
    def _process_state_update(self, message: Message):
        """Process state update message"""
        if not message.content:
            return
            
        agent_id = message.sender
        self.agent_states[agent_id] = message.content
        
        # Update metrics
        if 'metrics' in message.content:
            for metric, value in message.content['metrics'].items():
                self.metrics[metric][agent_id].append(value)
    
    def _process_action_plan(self, message: Message):
        """Process action plan message"""
        if not message.content:
            return
            
        agent_id = message.sender
        self.last_actions[agent_id] = message.content.get('action')
    
    def _process_coordination(self, message: Message):
        """Process coordination request/response"""
        if not message.content:
            return
            
        # For coordination requests
        if message.response_to is None:
            # Generate and send response if appropriate
            response = self._generate_coordination_response(message)
            if response:
                self.messages[Message.TYPES['COORDINATION']].append(response)
        
        # For coordination responses
        else:
            # Update coordination scores
            self.metrics['coordination_scores'][message.sender].append(
                message.content.get('value', 0.0)
            )
    
    def get_performance_score(self) -> float:
        """Calculate performance score for the region"""
        if not self.metrics:
            return 0.0
            
        scores = []
        
        # Calculate throughput score
        if self.metrics['throughput']:
            recent_throughput = self.metrics['throughput'][-5:]
            if recent_throughput:
                throughput_score = np.mean(recent_throughput)
                scores.append(min(1.0, throughput_score / 10.0))
        
        # Calculate waiting time score
        if self.metrics['waiting_times']:
            recent_waiting = self.metrics['waiting_times'][-5:]
            if recent_waiting:
                waiting_score = 1.0 - min(1.0, np.mean(recent_waiting) / 180.0)
                scores.append(waiting_score)
        
        # Calculate congestion score
        if self.metrics['congestion_levels']:
            recent_congestion = self.metrics['congestion_levels'][-5:]
            if recent_congestion:
                congestion_score = 1.0 - min(1.0, np.mean(recent_congestion))
                scores.append(congestion_score)
        
        # Calculate coordination score
        if self.metrics['coordination_scores']:
            recent_coordination = self.metrics['coordination_scores'][-5:]
            if recent_coordination:
                coord_score = min(1.0, np.mean(recent_coordination))
                scores.append(coord_score)
        
        # Return average score
        return np.mean(scores) if scores else 0.0
    
    def _generate_coordination_response(self, request: Message) -> Optional[Message]:
        """Generate response to coordination request"""
        if request.sender not in self.member_agents:
            return None
            
        return Message(
            msg_type=Message.TYPES['COORDINATION'],
            sender=self.region_id,
            content={
                'region_state': self.agent_states,
                'recommendations': self._generate_recommendations(),
                'timestamp': request.timestamp
            },
            priority=0.8,
            response_to=request.sender
        )
    
    def _generate_recommendations(self) -> dict:
        """Generate action recommendations based on regional state"""
        recommendations = {}
        
        # Simple recommendation based on average metrics
        for metric in self.metrics:
            if metric in ['waiting_times', 'congestion_levels']:
                for agent_id in self.member_agents:
                    if agent_id in self.metrics[metric]:
                        recent_values = self.metrics[metric][agent_id][-5:]
                        if recent_values:
                            recommendations[f'{metric}_{agent_id}'] = np.mean(recent_values)
        
        return recommendations

class RegionManager:
    """Manages division of network into regions and regional coordinators"""
    
    def __init__(self,
                net_file: str,
                config: dict,  # Added config parameter
                max_region_size: int = 10,
                min_region_size: int = 3,
                overlap_threshold: float = 0.2):
        """
        Initialize region manager
        
        Args:
            net_file: Path to SUMO network file
            config: Configuration dictionary
            max_region_size: Maximum agents per region
            min_region_size: Minimum agents per region
            overlap_threshold: Allowed region overlap ratio
        """
        self.net_file = net_file
        self.max_region_size = max_region_size
        self.min_region_size = min_region_size
        self.overlap_threshold = overlap_threshold
        
        # Get influence radius from config
        self.influence_radius = config.get('influence_radius', 120.0)  # Default to 120.0
        
        # Network representation
        self.network_graph = nx.Graph()
        self.regions = {}
        self.agent_region_map = defaultdict(set)
        
        # Region metrics
        self.region_metrics = defaultdict(dict)
        self.cross_region_edges = set()
        
        # Performance tracking
        self.metrics = {
            'num_regions': [],
            'avg_region_size': [],
            'region_overlap': [],
            'cross_region_flow': []
        }
        
        # Build initial network representation
        self._build_network_graph()
    
    def _build_network_graph(self):
        """Build graph representation of the traffic network"""
        # Read SUMO network
        net = sumolib.net.readNet(self.net_file)
        
        # Add nodes (junctions)
        for junction in net.getNodes():
            self.network_graph.add_node(
                junction.getID(),
                pos=junction.getCoord(),
                type=junction.getType()
            )
        
        # Add edges
        for edge in net.getEdges():
            from_node = edge.getFromNode().getID()
            to_node = edge.getToNode().getID()
            
            # Add edge with attributes
            self.network_graph.add_edge(
                from_node,
                to_node,
                id=edge.getID(),
                length=edge.getLength(),
                speed=edge.getSpeed()
            )
    
    def _identify_regions(self, agent_positions: Dict[str, Tuple[float, float]]):
        """
        Identify regions based on agent positions and network structure
        
        Args:
            agent_positions: Dictionary mapping agent IDs to their positions
        """
        # Reset current regions
        self.regions.clear()
        self.agent_region_map.clear()
        
        # Create graph of agent proximities
        proximity_graph = nx.Graph()
        for agent_id, pos in agent_positions.items():
            proximity_graph.add_node(agent_id, pos=pos)
        
        # Add edges between nearby agents
        for agent1 in agent_positions:
            pos1 = np.array(agent_positions[agent1])
            for agent2 in agent_positions:
                if agent1 != agent2:
                    pos2 = np.array(agent_positions[agent2])
                    distance = np.linalg.norm(pos1 - pos2)
                    
                    # Connect agents within influence radius
                    if distance <= self.influence_radius:
                        proximity_graph.add_edge(agent1, agent2, weight=distance)
        
        # Find communities in proximity graph
        communities = self._find_communities(proximity_graph)
        
        # Create regions from communities
        for i, community in enumerate(communities):
            region_id = f"region_{i}"
            self.regions[region_id] = list(community)
            
            # Update agent-region mapping
            for agent_id in community:
                self.agent_region_map[agent_id].add(region_id)
        
        # Update metrics
        self._update_region_metrics()
    
    def _find_communities(self, graph: nx.Graph) -> List[Set[str]]:
        """Find communities in agent proximity graph using Louvain method"""
        try:
            import community
            
            # First pass: detect base communities
            partition = community.best_partition(graph)
            communities = defaultdict(set)
            
            for node, comm_id in partition.items():
                communities[comm_id].add(node)
            
            # Adjust community sizes
            final_communities = []
            for comm in communities.values():
                if len(comm) > self.max_region_size:
                    # Split large communities
                    sub_comms = self._split_community(graph.subgraph(comm))
                    final_communities.extend(sub_comms)
                elif len(comm) < self.min_region_size:
                    # Try to merge small communities
                    self._merge_small_community(comm, final_communities)
                else:
                    final_communities.append(comm)
            
            return final_communities
            
        except ImportError:
            # Fallback to simple distance-based clustering
            return self._distance_based_clustering(graph)
    
    def _split_community(self, community_graph: nx.Graph) -> List[Set[str]]:
        """Split large community into smaller sub-communities"""
        sub_communities = []
        nodes = list(community_graph.nodes())
        
        while nodes:
            # Start new sub-community
            sub_comm = {nodes[0]}
            frontier = {nodes[0]}
            
            # Grow sub-community until size limit
            while len(sub_comm) < self.max_region_size and frontier:
                current = frontier.pop()
                neighbors = set(community_graph.neighbors(current)) - sub_comm
                
                # Add closest neighbors
                sorted_neighbors = sorted(
                    neighbors,
                    key=lambda x: community_graph[current][x]['weight']
                )
                
                for neighbor in sorted_neighbors:
                    if len(sub_comm) < self.max_region_size:
                        sub_comm.add(neighbor)
                        frontier.add(neighbor)
            
            # Add sub-community and remove processed nodes
            sub_communities.append(sub_comm)
            nodes = [n for n in nodes if n not in sub_comm]
        
        return sub_communities
    
    def _merge_small_community(self, 
                             community: Set[str],
                             existing_communities: List[Set[str]]):
        """Try to merge small community with existing ones"""
        if not existing_communities:
            existing_communities.append(community)
            return
        
        # Find best community to merge with
        best_merge = None
        min_size = float('inf')
        
        for i, existing in enumerate(existing_communities):
            merged_size = len(existing | community)
            if merged_size <= self.max_region_size and merged_size < min_size:
                min_size = merged_size
                best_merge = i
        
        if best_merge is not None:
            # Merge communities
            existing_communities[best_merge] |= community
        else:
            # Add as new community if can't merge
            existing_communities.append(community)
    
    def _distance_based_clustering(self, graph: nx.Graph) -> List[Set[str]]:
        """Simple distance-based clustering fallback"""
        communities = []
        unassigned = set(graph.nodes())
        
        while unassigned:
            # Start new community from random node
            center = random.choice(list(unassigned))
            community = {center}
            
            # Add nearest neighbors until size limit
            while len(community) < self.max_region_size and unassigned:
                distances = []
                for node in unassigned - community:
                    if node in graph[center]:
                        distances.append(
                            (node, graph[center][node]['weight'])
                        )
                
                if not distances:
                    break
                    
                # Add closest node
                closest = min(distances, key=lambda x: x[1])[0]
                community.add(closest)
                unassigned.remove(closest)
            
            communities.append(community)
            unassigned -= community
        
        return communities
    
    def _update_region_metrics(self):
        """Update region performance metrics"""
        # Calculate region sizes
        region_sizes = [len(members) for members in self.regions.values()]
        
        # Calculate region overlap
        total_assignments = sum(len(regions) for regions in self.agent_region_map.values())
        avg_assignments = total_assignments / max(len(self.agent_region_map), 1)
        
        # Update metrics
        self.metrics['num_regions'].append(len(self.regions))
        self.metrics['avg_region_size'].append(np.mean(region_sizes))
        self.metrics['region_overlap'].append(avg_assignments - 1)

# Part B

class Message:
    """Represents a communication message between agents"""
    
    TYPES = {
        'STATE_UPDATE': 1,     # Share state information
        'ACTION_PLAN': 2,      # Share planned actions
        'COORDINATION': 3,     # Coordination requests/responses
        'RECOMMENDATION': 4,   # Regional recommendations
        'ALERT': 5            # Critical situation alerts
    }
    
    def __init__(self,
                msg_type: int,
                sender: str,
                content: dict,
                priority: float = 1.0,
                timestamp: Optional[float] = None):
        """
        Initialize message
        
        Args:
            msg_type: Type of message (from TYPES)
            sender: ID of sending agent
            content: Message content dictionary
            priority: Message priority (0-1)
            timestamp: Message creation time
        """
        self.type = msg_type
        self.sender = sender
        self.content = content
        self.priority = priority
        self.timestamp = timestamp or traci.simulation.getTime()
        
        # Track message handling
        self.processed = False
        self.response_to = None
        self.responses = []

class HierarchicalNetwork(nn.Module):
    """Neural network with hierarchical structure"""
    
    def __init__(self,
                local_size: int,
                regional_size: int,
                action_size: int,
                hidden_size: int = 128):
        """
        Initialize hierarchical network
        
        Args:
            local_size: Size of local state input
            regional_size: Size of regional state input
            action_size: Size of action space
            hidden_size: Size of hidden layers
        """
        super(HierarchicalNetwork, self).__init__()
        
        # Local state processing
        self.local_encoder = nn.Sequential(
            nn.Linear(local_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Regional state processing
        self.regional_encoder = nn.Sequential(
            nn.Linear(regional_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Action prediction
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        # Value prediction for advantage calculation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self,
               local_state: torch.Tensor,
               regional_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network
        
        Args:
            local_state: Local agent state
            regional_state: Optional regional state
            
        Returns:
            Tuple of (action_values, state_value)
        """
        # Process local state
        local_features = self.local_encoder(local_state)
        
        # Process regional state if available
        if regional_state is not None:
            regional_features = self.regional_encoder(regional_state)
        else:
            regional_features = torch.zeros_like(local_features)
        
        # Combine features
        combined = torch.cat([local_features, regional_features], dim=-1)
        
        # Predict action values and state value
        action_values = self.action_head(combined)
        state_value = self.value_head(combined)
        
        return action_values, state_value

class HierarchicalAgent:
    """Agent that operates within hierarchical communication structure"""
    
    def __init__(self,
                agent_id: str,
                state_size: int,
                action_size: int,
                config: dict):
        """Initialize hierarchical agent"""
        self.id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Network parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.regional_size = config.get('regional_size', state_size * 3)
        
        # Initialize networks
        self.policy_net = HierarchicalNetwork(
            state_size,
            self.regional_size,
            action_size
        ).to(self.device)
        
        self.target_net = HierarchicalNetwork(
            state_size,
            self.regional_size,
            action_size
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.get('learning_rate', 0.001)
        )
        
        # Memory buffers
        self.memory_size = config.get('memory_size', 10000)
        self.batch_size = config.get('batch_size', 32)
        self.memory = []
        
        # Learning parameters
        self.gamma = config.get('gamma', 0.95)
        self.tau = config.get('tau', 0.01)
        self.regional_weight = config.get('regional_weight', 0.3)
        
        # Action selection
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        
        # Communication components
        self.message_queue = []
        self.received_messages = defaultdict(list)
        self.regional_state = None
        self.coordinator = None
        
        # Performance tracking
        self.metrics = defaultdict(list)
        self.training_step = 0

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        # Convert state inputs to tensors if needed
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state)
            
        # Add experience to memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Trim memory if exceeds size
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size:]
            
        # Track metrics
        self.metrics['rewards'].append(reward)
        if done:
            self.metrics['episode_lengths'].append(len(self.metrics['rewards']))

    def update(self) -> Optional[float]:
        """Update agent's policy"""
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack([s.to(self.device) for s in states])
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack([s.to(self.device) for s in next_states])
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        current_Q, _ = self.policy_net(states, self.regional_state)
        current_Q = current_Q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get next Q values
        with torch.no_grad():
            next_Q, _ = self.target_net(next_states, self.regional_state)
            next_Q = next_Q.max(1)[0]
            target_Q = rewards + (1 - dones) * self.gamma * next_Q
        
        # Compute loss
        loss = F.mse_loss(current_Q, target_Q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if self.training_step % self.config.get('target_update', 10) == 0:
            for target_param, policy_param in zip(
                self.target_net.parameters(),
                self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * policy_param.data +
                    (1 - self.tau) * target_param.data
                )
        
        # Update epsilon
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )
        
        self.training_step += 1
        return loss.item()
    
    def process_messages(self):
        """Process received messages and update regional state"""
        if not self.message_queue:
            return
        
        # Sort messages by priority and timestamp
        self.message_queue.sort(key=lambda x: (-x.priority, x.timestamp))
        
        # Process messages
        regional_updates = []
        coordination_requests = []
        recommendations = []
        
        for message in self.message_queue:
            if message.processed:
                continue
                
            if message.type == Message.TYPES['STATE_UPDATE']:
                regional_updates.append(message.content)
            elif message.type == Message.TYPES['COORDINATION']:
                coordination_requests.append(message)
            elif message.type == Message.TYPES['RECOMMENDATION']:
                recommendations.append(message.content)
            
            # Mark as processed
            message.processed = True
            
            # Store in history
            self.received_messages[message.type].append(message)
        
        # Update regional state
        if regional_updates:
            self._update_regional_state(regional_updates)
        
        # Handle coordination requests
        if coordination_requests:
            self._handle_coordination(coordination_requests)
        
        # Clear processed messages
        self.message_queue = [m for m in self.message_queue if not m.processed]
    
    def _update_regional_state(self, updates: List[dict]):
        """Update regional state representation"""
        if not updates:
            return
        
        # Combine updates (simple averaging for now)
        combined_state = defaultdict(list)
        
        for update in updates:
            for key, value in update.items():
                if isinstance(value, (int, float)):
                    combined_state[key].append(value)
        
        # Average values
        self.regional_state = {
            key: np.mean(values) for key, values in combined_state.items()
        }
    
    def _handle_coordination(self, requests: List[Message]):
        """Handle coordination requests"""
        for request in requests:
            # Evaluate request
            if self._should_coordinate(request):
                # Generate response
                response = self._generate_coordination_response(request)
                
                # Send response
                if self.coordinator:
                    self.coordinator.relay_message(response)
                
                # Update request
                request.responses.append(response)
    
    def _should_coordinate(self, request: Message) -> bool:
        """Decide whether to coordinate with requesting agent"""
        if not request.content:
            return False
        
        # Check if coordination would be beneficial
        try:
            # Get requestor's state
            other_state = request.content.get('state')
            if other_state is None:
                return False
            
            # Calculate state similarity
            similarity = self._calculate_state_similarity(other_state)
            
            # Check coordination criteria
            return (similarity > self.config.get('coordination_threshold', 0.7) and
                   len(request.responses) < self.config.get('max_responses', 3))
        
        except Exception as e:
            print(f"Error evaluating coordination request: {e}")
            return False
    
    def _generate_coordination_response(self, request: Message) -> Message:
        """Generate response to coordination request"""
        return Message(
            msg_type=Message.TYPES['COORDINATION'],
            sender=self.id,
            content={
                'state': self._get_shareable_state(),
                'action': self.last_action,
                'value': float(self.last_value) if hasattr(self, 'last_value') else 0.0,
                'timestamp': traci.simulation.getTime()
            },
            priority=0.8,
            response_to=request.sender
        )
    
    def _calculate_state_similarity(self, other_state: dict) -> float:
        """Calculate similarity between agent states"""
        if not isinstance(other_state, dict):
            return 0.0
        
        my_state = self._get_shareable_state()
        
        # Get common keys
        common_keys = set(my_state.keys()) & set(other_state.keys())
        if not common_keys:
            return 0.0
        
        # Calculate similarity
        similarities = []
        for key in common_keys:
            try:
                my_val = float(my_state[key])
                other_val = float(other_state[key])
                
                # Calculate normalized difference
                max_val = max(abs(my_val), abs(other_val))
                if max_val > 0:
                    similarity = 1 - abs(my_val - other_val) / max_val
                else:
                    similarity = 1.0
                    
                similarities.append(similarity)
            except (ValueError, TypeError):
                continue
        
        return np.mean(similarities) if similarities else 0.0
    
    def _get_shareable_state(self) -> dict:
        """Get state information that can be shared with other agents"""
        try:
            state = {
                'queue_length': self.metrics.get('queue_length', [0])[-1],
                'waiting_time': self.metrics.get('waiting_time', [0])[-1],
                'throughput': self.metrics.get('throughput', [0])[-1]
            }
            return state
        except Exception:
            return {}

    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get regional state tensor if available
            if self.regional_state is not None:
                regional_tensor = torch.FloatTensor(
                    list(self.regional_state.values())
                ).unsqueeze(0).to(self.device)
            else:
                regional_tensor = None
            
            # Get action values and state value
            action_values, state_value = self.policy_net(state, regional_tensor)
            
            # Store value for coordination
            self.last_value = state_value.item()
            
            return action_values.argmax().item()
    
    def save(self, path: str):
        """Save agent state"""
        save_dict = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': dict(self.metrics),
            'training_step': self.training_step,
            'epsilon': self.epsilon
        }
        torch.save(save_dict, path)
    
    def load(self, path: str):
        """Load agent state"""
        if not os.path.exists(path):
            return
            
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.metrics = defaultdict(list, checkpoint['metrics'])
        self.training_step = checkpoint['training_step']
        self.epsilon = checkpoint['epsilon']

# Part C

class HierarchicalManager:
    """Manages hierarchical multi-agent system and training"""
    
    def __init__(self,
                state_size: int,
                action_size: int,
                net_file: str,
                config: dict):
        """Initialize hierarchical system manager"""
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Initialize region management
        self.region_manager = RegionManager(
            net_file,
            config=config,  # Pass the config
            max_region_size=config.get('max_region_size', 10),
            min_region_size=config.get('min_region_size', 3),
            overlap_threshold=config.get('overlap_threshold', 0.2)
        )
        
        # Initialize agents and coordinators
        self.agents = {}
        self.coordinators = {}
        
        # Performance tracking
        self.metrics = defaultdict(list)
        self.episode_rewards = defaultdict(list)
        self.training_steps = 0
        
        # Set up logging
        self.logger = logging.getLogger('HierarchicalManager')
        self.logger.setLevel(logging.INFO)
    
    def add_agent(self, agent_id: str):
        """Add new agent to the system"""
        self.agents[agent_id] = HierarchicalAgent(
            agent_id,
            self.state_size,
            self.action_size,
            self.config
        )
    
    def update_regions(self, agent_positions: Dict[str, Tuple[float, float]]):
        """Update regional organization based on agent positions"""
        # Update region assignments
        self.region_manager._identify_regions(agent_positions)
        
        # Create/update regional coordinators
        new_regions = set(self.region_manager.regions.keys())
        current_regions = set(self.coordinators.keys())
        
        # Remove obsolete coordinators
        for region_id in current_regions - new_regions:
            del self.coordinators[region_id]
        
        # Add new coordinators
        for region_id in new_regions - current_regions:
            members = self.region_manager.regions[region_id]
            self.coordinators[region_id] = RegionalCoordinator(
                region_id,
                members,
                self.state_size,
                self.action_size,
                self.config
            )
        
        # Update agent assignments
        for agent_id, agent in self.agents.items():
            regions = self.region_manager.agent_region_map[agent_id]
            if regions:
                # Assign primary coordinator (closest or most important region)
                primary_region = self._select_primary_region(agent_id, regions)
                agent.coordinator = self.coordinators[primary_region]
    
    def _select_primary_region(self, agent_id: str, regions: Set[str]) -> str:
        """Select primary region for an agent"""
        if len(regions) == 1:
            return list(regions)[0]
        
        # Calculate region scores based on size and position
        region_scores = []
        agent_pos = self.agents[agent_id].get_position()
        
        for region_id in regions:
            coordinator = self.coordinators[region_id]
            
            # Calculate average position of region members
            member_positions = [
                self.agents[member].get_position()
                for member in coordinator.member_agents
                if member in self.agents
            ]
            
            if member_positions:
                center = np.mean(member_positions, axis=0)
                distance = np.linalg.norm(center - agent_pos)
                
                # Score based on size and distance
                size_score = len(coordinator.member_agents) / self.config.get('max_region_size', 10)
                distance_score = 1 / (1 + distance)  # Normalize distance
                
                region_scores.append((
                    region_id,
                    0.7 * distance_score + 0.3 * size_score  # Weight factors
                ))
        
        if region_scores:
            return max(region_scores, key=lambda x: x[1])[0]
        return list(regions)[0]
    
    def step(self, states: Dict[str, np.ndarray], training: bool = True) -> Dict[str, int]:
        """
        Execute step for all agents
        
        Args:
            states: Dictionary mapping agent IDs to states
            training: Whether in training mode
            
        Returns:
            Dictionary mapping agent IDs to selected actions
        """
        # Process messages and update regional states
        self.process_communications()  # Changed from _process_communications
        
        # Select actions for all agents
        actions = {}
        for agent_id, state in states.items():
            if agent_id in self.agents:
                actions[agent_id] = self.agents[agent_id].select_action(state)
        
        # Update training steps
        if training:
            self.training_steps += 1
        
        return actions
    
    def process_communications(self):  # Changed from _process_communications
        """Process all pending communications"""
        # Process coordinator messages first
        for coordinator in self.coordinators.values():
            coordinator.process_messages()
        
        # Then process agent messages
        for agent in self.agents.values():
            agent.process_messages()
    
    def update(self,
              states: Dict[str, np.ndarray],
              actions: Dict[str, int],
              rewards: Dict[str, float],
              next_states: Dict[str, np.ndarray],
              dones: Dict[str, bool]) -> Dict[str, float]:
        """
        Update all agents
        
        Returns:
            Dictionary of agent losses
        """
        losses = {}
        
        # Calculate global reward component
        global_reward = sum(rewards.values()) / len(rewards)
        
        # Update each agent
        for agent_id in states:
            if agent_id not in self.agents:
                continue
                
            agent = self.agents[agent_id]
            
            # Get regional reward component if available
            regional_reward = 0.0
            if agent.coordinator:
                region_members = agent.coordinator.member_agents
                regional_reward = sum(
                    rewards[member] for member in region_members
                    if member in rewards
                ) / len(region_members)
            
            # Combine rewards
            combined_reward = (
                (1 - agent.regional_weight) * rewards[agent_id] +
                agent.regional_weight * (
                    0.7 * regional_reward + 0.3 * global_reward
                )
            )
            
            # Store experience
            agent.remember(
                states[agent_id],
                actions[agent_id],
                combined_reward,
                next_states[agent_id],
                dones[agent_id]
            )
            
            # Update agent
            loss = agent.update()
            if loss is not None:
                losses[agent_id] = loss
            
            # Track rewards
            self.episode_rewards[agent_id].append(combined_reward)
        
        return losses
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        metrics = {
            'avg_reward': np.mean([
                np.mean(rewards) for rewards in self.episode_rewards.values()
            ]),
            'num_regions': len(self.coordinators),
            'avg_region_size': np.mean([
                len(coord.member_agents) for coord in self.coordinators.values()
            ]),
            'message_rate': np.mean([
                len(agent.message_queue) for agent in self.agents.values()
            ])
        }
        
        return metrics
    
    def save(self, path: str):
        """Save all agents and system state"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save agents
        agent_dir = save_path / 'agents'
        agent_dir.mkdir(exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            agent.save(agent_dir / f'{agent_id}.pt')
        
        # Save metrics
        with open(save_path / 'metrics.json', 'w') as f:
            json.dump(dict(self.metrics), f, indent=4)  # Convert defaultdict to dict for JSON serialization
    
    def load(self, path: str):
        """Load agents and system state"""
        load_path = Path(path)
        
        # Load agents
        agent_dir = load_path / 'agents'
        for agent_file in agent_dir.glob('*.pt'):
            agent_id = agent_file.stem
            if agent_id in self.agents:
                self.agents[agent_id].load(agent_file)
        
        # Load metrics
        metrics_file = load_path / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                self.metrics = defaultdict(list, json.load(f))

def main():
    """Example usage and testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test hierarchical communication')
    parser.add_argument('--net-file', required=True,
                      help='Path to SUMO network file')
    parser.add_argument('--route-file', required=True,
                      help='Path to SUMO route file')
    parser.add_argument('--num-episodes', type=int, default=100,
                      help='Number of episodes to train')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'max_region_size': 8,
        'min_region_size': 3,
        'overlap_threshold': 0.2,
        'regional_weight': 0.3,
        'coordination_threshold': 0.7
    }
    
    # Initialize environment
    env = MultiAgentSumoEnvironment(
        net_file=args.net_file,
        route_file=args.route_file,
        use_gui=False
    )
    
    # Create manager
    manager = HierarchicalManager(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        net_file=args.net_file,
        config=config
    )
    
    # Add agents
    for tl_id in env.traffic_lights:
        manager.add_agent(tl_id)
    
    # Training loop
    try:
        print("\nStarting training...")
        for episode in range(args.num_episodes):
            states, _ = env.reset()
            episode_reward = defaultdict(float)
            done = False
            
            while not done:
                # Update regions based on current positions
                positions = env.get_agent_positions()
                manager.update_regions(positions)
                
                # Get actions
                actions = manager.step(states)
                
                # Execute in environment
                next_states, rewards, done, _, info = env.step(actions)
                
                # Update agents
                losses = manager.update(states, actions, rewards, next_states,
                                     {tl: done for tl in states})
                
                # Track rewards
                for tl_id, reward in rewards.items():
                    episode_reward[tl_id] += reward
                
                states = next_states
            
            # Log progress
            metrics = manager.get_metrics()
            print(f"\nEpisode {episode + 1}/{args.num_episodes}")
            print(f"Average Reward: {metrics['avg_reward']:.2f}")
            print(f"Number of Regions: {metrics['num_regions']}")
            print(f"Average Region Size: {metrics['avg_region_size']:.1f}")
            print(f"Message Rate: {metrics['message_rate']:.1f}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Save final state
        manager.save('models/hierarchical_final')
        env.close()

if __name__ == "__main__":
    main()