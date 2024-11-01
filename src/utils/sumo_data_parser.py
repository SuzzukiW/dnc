# src/utils/sumo_data_parser.py

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class SUMODataParser:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def parse_traffic_light_states(self) -> pd.DataFrame:
        """Parse traffic light states and switches."""
        # Parse TLS states
        tls_states_tree = ET.parse(self.data_dir / 'tls_states.xml')
        states = []
        
        for tls in tls_states_tree.getroot():
            tls_id = tls.get('id')
            for state in tls:
                states.append({
                    'tls_id': tls_id,
                    'time': float(state.get('time')),
                    'state': state.get('state'),
                    'program_id': state.get('programID')
                })
        
        return pd.DataFrame(states)
    
    def parse_traffic_light_switches(self) -> pd.DataFrame:
        """Parse traffic light switch information."""
        switches_tree = ET.parse(self.data_dir / 'tls_switches.xml')
        switches = []
        
        for switch in switches_tree.getroot():
            switches.append({
                'time': float(switch.get('time')),
                'tls_id': switch.get('id'),
                'program_id': switch.get('programID'),
                'from_state': switch.get('fromState'),
                'to_state': switch.get('toState')
            })
            
        return pd.DataFrame(switches)
    
    def parse_vehicle_info(self) -> pd.DataFrame:
        """Parse vehicle trip information."""
        tripinfo_tree = ET.parse(self.data_dir / 'tripinfo.xml')
        trips = []
        
        for trip in tripinfo_tree.getroot():
            if trip.tag == 'tripinfo':
                trips.append({
                    'id': trip.get('id'),
                    'depart': float(trip.get('depart')),
                    'arrival': float(trip.get('arrival')),
                    'duration': float(trip.get('duration')),
                    'waiting_time': float(trip.get('waitingTime')),
                    'time_loss': float(trip.get('timeLoss'))
                })
                
        return pd.DataFrame(trips)
    
    def parse_traffic_summary(self) -> pd.DataFrame:
        """Parse summary statistics."""
        summary_tree = ET.parse(self.data_dir / 'summary.xml')
        summaries = []
        
        for step in summary_tree.getroot():
            summaries.append({
                'time': float(step.get('time')),
                'loaded': int(step.get('loaded')),
                'inserted': int(step.get('inserted')),
                'running': int(step.get('running')),
                'waiting': int(step.get('waiting')),
                'mean_wait_time': float(step.get('meanWaitingTime', 0))
            })
            
        return pd.DataFrame(summaries)
    
    def create_training_dataset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Create structured dataset for training."""
        # Load all data
        tls_states = self.parse_traffic_light_states()
        switches = self.parse_traffic_light_switches()
        vehicle_info = self.parse_vehicle_info()
        summary = self.parse_traffic_summary()
        
        # Process data into state-action pairs
        states = {}
        rewards = {}
        
        # Group by traffic light
        for tls_id in tls_states['tls_id'].unique():
            tls_data = tls_states[tls_states['tls_id'] == tls_id].sort_values('time')
            
            # Create state features
            state_features = []
            reward_values = []
            
            for i in range(len(tls_data) - 1):
                time = tls_data.iloc[i]['time']
                next_time = tls_data.iloc[i + 1]['time']
                
                # Get traffic conditions at this time
                current_summary = summary[
                    (summary['time'] >= time) & 
                    (summary['time'] < next_time)
                ]
                
                # Create state vector
                state_vector = [
                    current_summary['running'].mean(),
                    current_summary['waiting'].mean(),
                    current_summary['mean_wait_time'].mean(),
                    len(tls_data.iloc[i]['state']),  # Number of lanes
                    time / 3600.0  # Normalized time of day
                ]
                
                # Calculate reward based on changes in waiting time
                reward = -current_summary['mean_wait_time'].mean()
                
                state_features.append(state_vector)
                reward_values.append(reward)
            
            # Convert to numpy arrays
            states[tls_id] = np.array(state_features)
            rewards[tls_id] = np.array(reward_values)
        
        return states, rewards
    
    def get_action_space_size(self) -> int:
        """Get the size of action space based on unique program IDs."""
        tls_states = self.parse_traffic_light_states()
        return len(tls_states['program_id'].unique())
    
    def get_state_space_size(self) -> int:
        """Get the size of state space."""
        return 5