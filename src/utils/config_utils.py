class Args:
    """Simple class to hold training arguments"""
    def __init__(self, config_dict):
        self.device = config_dict['hardware']['device']
        self.seed = config_dict['hardware']['seed']  # Add seed parameter
        self.hidden_size = config_dict['agent']['hidden_size']
        self.actor_lr = config_dict['agent']['learning_rate_actor']
        self.critic_lr = config_dict['agent']['learning_rate_critic']
        self.gamma = config_dict['agent']['gamma']
        self.tau = config_dict['agent']['tau']
        self.batch_size = config_dict['training']['batch_size']
        self.buffer_size = int(1e6)  # Fixed buffer size
        self.max_episode_len = config_dict['training']['max_steps']
        self.update_every = config_dict['training'].get('update_every', 4)  # Added update_every parameter with default value

def create_maddpg_args(config, device):
    """Create arguments for MADDPG agent"""
    # Override device if provided
    config['hardware']['device'] = device
    return Args(config)