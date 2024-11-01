import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple
from src.utils.sumo_xml_parser import SUMOXMLParser  # Updated import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SUMODataProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.csv_dir = self.data_dir / "csv"
        self.csv_dir.mkdir(exist_ok=True)
        self.parser = SUMOXMLParser(data_dir)  # Updated class name
    
    def convert_to_csv(self):
        """Convert XML files to CSV using SUMO parser."""
        return self.parser.process_all()
    
    def load_traffic_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load traffic data from CSV files."""
        dfs = {}
        required_files = [
            'tls_states.csv',
            'tripinfo.csv',
            'summary.csv'
        ]
        
        for csv_file in required_files:
            try:
                df = pd.read_csv(self.csv_dir / csv_file)
                if 'time' in df.columns:
                    df['time'] = pd.to_numeric(df['time'], errors='coerce')
                dfs[csv_file] = df
                logger.info(f"Loaded {csv_file}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {str(e)}")
                dfs[csv_file] = pd.DataFrame()
        
        return (
            dfs.get('tls_states.csv', pd.DataFrame()),
            dfs.get('tripinfo.csv', pd.DataFrame()),
            dfs.get('summary.csv', pd.DataFrame())
        )
    
    def prepare_training_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Prepare training data from CSV files."""
        # Load all data
        tls_states, tripinfo, summary = self.load_traffic_data()
        
        if tls_states.empty or summary.empty:
            logger.error("Required data not available")
            return {}, {}
        
        # Process data by traffic light
        states_dict = {}
        rewards_dict = {}
        
        # Group by traffic light ID
        for tl_id in tls_states['id'].unique():
            tl_states = tls_states[tls_states['id'] == tl_id].sort_values('time')
            
            # Create state features
            state_features = []
            rewards = []
            
            for i in range(len(tl_states) - 1):
                current_time = tl_states.iloc[i]['time']
                next_time = tl_states.iloc[i + 1]['time']
                
                # Get traffic conditions
                current_summary = summary[
                    (summary['time'] >= current_time) & 
                    (summary['time'] < next_time)
                ]
                
                if not current_summary.empty:
                    # State vector creation
                    state = [
                        current_time / 3600.0,  # Normalized time
                        current_summary['running'].mean(),
                        current_summary['waiting'].mean(),
                        current_summary['meanSpeed'].mean() if 'meanSpeed' in current_summary else 0,
                        len(tl_states.iloc[i]['state'])  # Number of controlled lanes
                    ]
                    
                    # Calculate reward
                    reward = -current_summary['meanWaitingTime'].mean()
                    if 'ended' in current_summary.columns:
                        reward += 0.1 * current_summary['ended'].mean()
                    
                    state_features.append(state)
                    rewards.append(reward)
            
            if state_features:
                states_dict[tl_id] = np.array(state_features)
                rewards_dict[tl_id] = np.array(rewards)
                logger.info(f"Prepared data for traffic light {tl_id}: {len(state_features)} samples")
        
        return states_dict, rewards_dict
    
    def get_state_space_size(self) -> int:
        """Get the size of state space."""
        return 5  # [time, running, waiting, mean_speed, num_lanes]
    
    def get_action_space_size(self) -> int:
        """Get the size of action space based on unique program IDs."""
        try:
            tls_states = pd.read_csv(self.csv_dir / 'tls_states.csv')
            if not tls_states.empty and 'programID' in tls_states.columns:
                return len(tls_states['programID'].unique())
        except Exception:
            pass
        return 2  # Default binary action space if no data available