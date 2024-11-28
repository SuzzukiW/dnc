"""Utility functions for SUMO environment"""

def get_state_space_size(config):
    """Get the size of state space.
    
    Args:
        config (dict): Environment configuration
        
    Returns:
        int: Size of state space
    """
    # Default state space includes:
    # - Queue length for each lane
    # - Waiting time for each lane
    # - Average speed for each lane
    max_lanes = config.get('max_lanes', 20)  # Maximum number of lanes per intersection
    return max_lanes * 3  # 3 features per lane

def get_action_space_size(config):
    """Get the size of action space.
    
    Args:
        config (dict): Environment configuration
        
    Returns:
        int: Size of action space
    """
    # Default to number of traffic light phases
    # Each traffic light can switch between available phases
    return config.get('num_phases', 4)  # Default to 4 phases if not specified