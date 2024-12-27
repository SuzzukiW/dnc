# Decentralized Neural Traffic Orchestration

[_Xiang Fu_](https://xfu.fufoundation.co) and [_George Jiang_](https://www.linkedin.com/in/georgejiang1/)

This repository accompanies the paper "Decentralized Neural Traffic Orchestration" at CDS DS 340 Fall 2024, at Boston University.

## Setup

### Python Environment

1. Create and activate a new conda environment (recommended):
```bash
conda create -n maestro python=3.8
conda activate maestro
```

2. Install the package and dependencies:
```bash
# Install the package in development mode
pip install -e .

# Install required dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch
- SUMO (Simulation of Urban MObility)
- Other dependencies listed in `requirements.txt`

### GPU Support
This project supports GPU acceleration for training deep learning models. To use GPU:
- Ensure you have CUDA-compatible GPU
- Install CUDA toolkit and cuDNN
- PyTorch will automatically detect and use available GPU

### Weights & Biases Integration
We use Weights & Biases (WandB) for experiment tracking and visualization. To use WandB:
1. Install wandb: `pip install wandb`
2. Login to your WandB account: `wandb login`
3. Set your project name in the configuration files
4. Experiments will automatically log metrics and hyperparameters

## Dataset
The dataset is organized in the `data/` directory, containing:
- Traffic scenarios
- Network configurations
- Traffic demand patterns

### Data Structure
```
data/
├── scenarios/      # Different traffic network scenarios
├── demand/         # Traffic demand patterns
└── configs/        # Configuration files for different setups
```

## Using Checkpoints
Model checkpoints are stored in the `experiments/models/` directory. To use a checkpoint:
1. Navigate to the checkpoint directory
2. Load the model using the appropriate configuration file
3. Checkpoints contain model weights and training state

## Running Experiments

### Training
Training experiments are organized in the `experiments/train/` directory. To run an experiment:
1. Configure your experiment parameters in the config files
2. Execute the training script with your chosen configuration
3. Monitor progress through WandB dashboard

### Evaluation
Evaluation scripts are in the `experiments/evaluate/` directory:
1. Select the model checkpoint to evaluate
2. Choose the evaluation scenario
3. Run the evaluation script

## Analysis Tools

### Statistics and Tables
- Use scripts in the `tools/` directory for data analysis
- Generate performance metrics and statistical comparisons
- Export results in various formats (CSV, JSON)

### Visualization
The project includes tools for:
- Traffic flow visualization
- Performance metrics plotting
- Learning curves and training statistics

## Notes

### Naming Conventions
- Model checkpoints: `{model_name}_{scenario}_{timestamp}.pth`
- Configuration files: `{scenario}_{config_type}.yaml`
- Results: `{experiment_name}_{date}_{metrics}.csv`

### Implementation Details
- Based on PyTorch for deep learning components
- Uses SUMO's TraCI interface for traffic simulation
- Modular architecture for easy extension and modification
- Supports multiple traffic signal control algorithms

## License
MIT License