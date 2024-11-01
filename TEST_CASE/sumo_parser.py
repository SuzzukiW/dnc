# TEST_CASE/sumo_parser.py

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class SUMODataParser:
    def __init__(self, data_dir: str):
        """
        Initialize the SUMO data parser
        Args:
            data_dir: Directory containing the SUMO XML files
        """
        self.data_dir = Path(data_dir)
        
    def parse_traffic_light_programs(self) -> Dict:
        """Parse traffic light programs from tls_programs.xml"""
        tree = ET.parse(self.data_dir / 'tls_programs.xml')
        root = tree.getroot()
        
        tl_programs = {}
        for tl in root.findall('.//tlLogic'):
            tl_id = tl.get('id')
            program_id = tl.get('programID')
            phases = []
            
            for phase in tl.findall('phase'):
                phases.append({
                    'duration': float(phase.get('duration')),
                    'state': phase.get('state')
                })
            
            if tl_id not in tl_programs:
                tl_programs[tl_id] = {}
            tl_programs[tl_id][program_id] = phases
            
        return tl_programs
    
    def parse_traffic_light_states(self) -> pd.DataFrame:
        """Parse traffic light states from tls_states.xml"""
        tree = ET.parse(self.data_dir / 'tls_states.xml')
        root = tree.getroot()
        
        data = []
        for tl_state in root.findall('.//tlsState'):
            data.append({
                'time': float(tl_state.get('time')),
                'id': tl_state.get('id'),
                'programID': tl_state.get('programID'),
                'phase': int(tl_state.get('phase')),
                'state': tl_state.get('state')
            })
            
        return pd.DataFrame(data)
    
    def parse_trip_info(self) -> pd.DataFrame:
        """Parse trip information from tripinfo.xml"""
        tree = ET.parse(self.data_dir / 'tripinfo.xml')
        root = tree.getroot()
        
        data = []
        for trip in root.findall('.//tripinfo'):
            data.append({
                'id': trip.get('id'),
                'depart': float(trip.get('depart')),
                'departLane': trip.get('departLane'),
                'departPos': float(trip.get('departPos')),
                'departSpeed': float(trip.get('departSpeed')),
                'departDelay': float(trip.get('departDelay')),
                'arrival': float(trip.get('arrival')),
                'arrivalLane': trip.get('arrivalLane'),
                'arrivalPos': float(trip.get('arrivalPos')),
                'arrivalSpeed': float(trip.get('arrivalSpeed')),
                'duration': float(trip.get('duration')),
                'routeLength': float(trip.get('routeLength')),
                'waitingTime': float(trip.get('waitingTime')),
                'waitingCount': int(trip.get('waitingCount'))
            })
            
        return pd.DataFrame(data)
    
    def parse_traffic_light_switches(self) -> pd.DataFrame:
        """Parse traffic light switches from tls_switches.xml"""
        tree = ET.parse(self.data_dir / 'tls_switches.xml')
        root = tree.getroot()
        
        data = []
        for switch in root.findall('.//switch'):
            data.append({
                'time': float(switch.get('time')),
                'id': switch.get('id'),
                'fromState': switch.get('fromState'),
                'toState': switch.get('toState')
            })
            
        return pd.DataFrame(data)
    
    def create_training_dataset(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Create a combined dataset for training the RL model
        Returns:
            DataFrame with combined features and statistics
            Dictionary with additional metadata
        """
        # Parse all data sources
        tl_programs = self.parse_traffic_light_programs()
        tl_states_df = self.parse_traffic_light_states()
        trip_info_df = self.parse_trip_info()
        tl_switches_df = self.parse_traffic_light_switches()
        
        # Process traffic light states
        tl_states_features = []
        unique_tls = tl_states_df['id'].unique()
        
        for tl_id in unique_tls:
            tl_data = tl_states_df[tl_states_df['id'] == tl_id]
            
            # Calculate features for each time step
            for _, row in tl_data.iterrows():
                time = row['time']
                
                # Get concurrent trips
                concurrent_trips = trip_info_df[
                    (trip_info_df['depart'] <= time) & 
                    (trip_info_df['arrival'] >= time)
                ]
                
                # Calculate features
                feature_dict = {
                    'time': time,
                    'tl_id': tl_id,
                    'phase': row['phase'],
                    'state': row['state'],
                    'program_id': row['programID'],
                    'active_vehicles': len(concurrent_trips),
                    'avg_waiting_time': concurrent_trips['waitingTime'].mean(),
                    'total_waiting_time': concurrent_trips['waitingTime'].sum()
                }
                
                # Add switches information
                recent_switches = len(tl_switches_df[
                    (tl_switches_df['time'] <= time) & 
                    (tl_switches_df['time'] > time - 60) &  # Last 60 seconds
                    (tl_switches_df['id'] == tl_id)
                ])
                feature_dict['recent_switches'] = recent_switches
                
                tl_states_features.append(feature_dict)
        
        # Create final dataset
        training_df = pd.DataFrame(tl_states_features)
        
        # Add metadata
        metadata = {
            'traffic_light_programs': tl_programs,
            'unique_traffic_lights': unique_tls,
            'simulation_duration': training_df['time'].max(),
            'total_vehicles': len(trip_info_df),
            'avg_trip_duration': trip_info_df['duration'].mean(),
            'avg_waiting_time': trip_info_df['waitingTime'].mean()
        }
        
        return training_df, metadata
    
    def get_state_space_size(self) -> int:
        """Calculate the size of the state space"""
        training_df, _ = self.create_training_dataset()
        
        # Count unique values for categorical features
        unique_phases = training_df['phase'].nunique()
        unique_states = len(set(''.join(training_df['state'].unique())))
        
        # Number of continuous features
        continuous_features = 4  # active_vehicles, avg_waiting_time, total_waiting_time, recent_switches
        
        return unique_phases * unique_states + continuous_features
    
    def get_action_space_size(self) -> int:
        """Calculate the size of the action space"""
        tl_programs = self.parse_traffic_light_programs()
        
        # Count unique phases across all programs
        unique_phases = set()
        for tl_id in tl_programs:
            for program in tl_programs[tl_id].values():
                for phase in program:
                    unique_phases.add(phase['state'])
        
        return len(unique_phases)