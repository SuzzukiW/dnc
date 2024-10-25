#!/bin/bash

# Create and activate a new conda environment (optional)
# conda create -n maestro python=3.8
# conda activate maestro

# Install the package in development mode
pip install -e .

# Install additional dependencies
pip install -r requirements.txt

# Run a quick test to verify installation
python -c "import src; print(src.__version__)"

echo "Setup complete! You can now run the training script with: python experiments/train.py"