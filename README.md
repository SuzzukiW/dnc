# CoLight: Cooperative Light Signal Optimization via Multi-Agent Reinforcement Learning

## Overview
This project implements a multi-agent reinforcement learning system to optimize traffic light patterns in a simulated city grid, reducing congestion and improving traffic flow efficiency through cooperative learning.

## Team
- George Jiang
- Xiang Fu

## Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

### Implementation Details
- Centralized training with decentralized execution
- Supports multiple agents in a cooperative learning environment
- Uses Ornstein-Uhlenbeck noise for exploration
- Configurable hyperparameters via YAML configuration

### Key Components
- `src/agents/maddpg_agent.py`: Multi-Agent DDPG Agent implementation
- `src/models/maddpg_network.py`: Neural network architectures
- `src/environment/maddpg_env.py`: Multi-agent environment wrapper
- `experiments/train_multi_agent_maddpg.py`: Training script
- `config/maddpg.yaml`: Configuration management

## Setup and Installation

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/your-repo/colight.git
cd colight
pip install -r requirements.txt
```

### Training
To train the MADDPG agent:
```bash
python experiments/train_multi_agent_maddpg.py
```

## Configuration
Modify `config/maddpg.yaml` to adjust:
- Environment settings
- Training hyperparameters
- Noise parameters
- Logging options

## Dependencies
- PyTorch
- NumPy
- Gymnasium
- YAML
- tqdm

## License
[Specify your license here]