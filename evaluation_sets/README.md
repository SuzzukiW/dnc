# Evaluation Metrics

## Overview
This module provides key performance metrics for traffic simulation and analysis.

## Metrics Implemented
- `average_waiting_time(vehicle_data)`: Calculates the average waiting time for vehicles in seconds.
- `total_throughput(vehicle_data)`: Calculates the total number of vehicles processed.
- `average_speed(vehicle_data)`: Calculates the average speed of vehicles in km/h.
- `max_waiting_time(vehicle_data)`: Calculates the maximum waiting time for vehicles in seconds.

## Usage
```python
from evaluation_sets.metrics import average_waiting_time, total_throughput, average_speed, max_waiting_time

# Example vehicle data
vehicle_data = [
    {'waiting_time': 10, 'speed': 50},
    {'waiting_time': 15, 'speed': 60},
    {'waiting_time': 5, 'speed': 40}
]

# Calculate metrics
avg_wait = average_waiting_time(vehicle_data)
throughput = total_throughput(vehicle_data)
avg_speed = average_speed(vehicle_data)
max_wait = max_waiting_time(vehicle_data)
```

## Requirements
- NumPy
- Python 3.7+
