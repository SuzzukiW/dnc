# evaluation_sets/metrics.py

"""
Evaluation Metrics Module for SUMO Traffic Simulation

This module provides key performance metrics specifically designed for SUMO 
(Simulation of Urban MObility) traffic simulations. 

Key SUMO-specific considerations:
- Waiting time is measured in seconds using traci.vehicle.getWaitingTime()
- Speed is typically measured in m/s using traci.vehicle.getSpeed()
- Vehicle data is extracted using SUMO's TraCI API

Functions are designed to be flexible and work with SUMO vehicle data dictionaries.
"""

from typing import List, Dict, Union, Optional
import numpy as np
import warnings

def _validate_sumo_vehicle_data(
    vehicle_data: List[Dict[str, Union[float, int]]], 
    metric_key: str
) -> List[float]:
    """
    Validate and extract metric values from SUMO vehicle data.
    
    This function is specifically designed to handle SUMO TraCI vehicle data,
    ensuring robust metric calculation even with imperfect or incomplete data.
    
    SUMO-specific notes:
    - Waiting time is measured in seconds (traci.vehicle.getWaitingTime())
    - Speed is typically in m/s (traci.vehicle.getSpeed())
    
    Args:
        vehicle_data (List[Dict]): List of SUMO vehicle data dictionaries
        metric_key (str): Key to extract from vehicle dictionaries (e.g., 'waiting_time', 'speed')
    
    Returns:
        List[float]: Validated metric values for SUMO vehicles
    
    Raises:
        ValueError: If input is not a list or contains invalid data
    """
    # Validate input type
    if not isinstance(vehicle_data, list):
        raise ValueError(f"Input must be a list of dictionaries, got {type(vehicle_data)}")
    
    # Handle empty list
    if not vehicle_data:
        warnings.warn(f"Empty SUMO vehicle data list for metric {metric_key}. Returning 0.", UserWarning)
        return [0.0]
    
    # Extract and validate metric values
    try:
        metric_values = []
        for vehicle in vehicle_data:
            # Validate each vehicle is a dictionary
            if not isinstance(vehicle, dict):
                warnings.warn(f"Invalid SUMO vehicle data: {vehicle}. Skipping.", UserWarning)
                continue
            
            # Extract metric value, default to 0 if not found
            value = vehicle.get(metric_key, 0)
            
            # Validate metric value is numeric
            if not isinstance(value, (int, float)):
                warnings.warn(f"Non-numeric SUMO {metric_key} value: {value}. Converting to float.", UserWarning)
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    warnings.warn(f"Could not convert SUMO {metric_key} to float. Using 0.", UserWarning)
                    value = 0.0
            
            # Ensure non-negative values
            value = max(0.0, float(value))
            metric_values.append(value)
        
        return metric_values
    
    except Exception as e:
        warnings.warn(f"Error processing SUMO vehicle data: {e}. Returning [0.0]", UserWarning)
        return [0.0]

def average_waiting_time(vehicle_data: List[Dict[str, Union[float, int]]]) -> float:
    """
    Calculate the average waiting time for vehicles in a SUMO simulation.
    
    Uses traci.vehicle.getWaitingTime() to measure vehicle waiting times.
    
    Args:
        vehicle_data (List[Dict]): List of SUMO vehicle data dictionaries 
                                   containing 'waiting_time' key.
    
    Returns:
        float: Average waiting time in seconds. Returns 0 if no valid vehicles are present.
    """
    waiting_times = _validate_sumo_vehicle_data(vehicle_data, 'waiting_time')
    return float(np.mean(waiting_times))

def total_throughput(vehicle_data: List[Dict[str, Union[float, int]]]) -> int:
    """
    Calculate the total number of vehicles that have passed through the SUMO system.
    
    Args:
        vehicle_data (List[Dict]): List of SUMO vehicle data dictionaries.
    
    Returns:
        int: Total number of valid vehicles processed in the SUMO simulation.
    """
    # Validate vehicle data, but don't care about a specific metric
    try:
        valid_vehicles = [v for v in vehicle_data if isinstance(v, dict)]
        return len(valid_vehicles)
    except Exception:
        return 0

def average_speed(vehicle_data: List[Dict[str, Union[float, int]]]) -> float:
    """
    Calculate the average speed of vehicles in a SUMO simulation.
    
    Uses traci.vehicle.getSpeed() to measure vehicle speeds.
    Converts from m/s to km/h for reporting.
    
    Args:
        vehicle_data (List[Dict]): List of SUMO vehicle data dictionaries 
                                   containing 'speed' key.
    
    Returns:
        float: Average speed in kilometers per hour. 
               Returns 0 if no valid vehicles are present.
    """
    speeds = _validate_sumo_vehicle_data(vehicle_data, 'speed')
    return float(np.mean(speeds)) * 3.6  # Convert m/s to km/h

def max_waiting_time(vehicle_data: List[Dict[str, Union[float, int]]]) -> float:
    """
    Calculate the maximum waiting time for vehicles in a SUMO simulation.
    
    Uses traci.vehicle.getWaitingTime() to measure vehicle waiting times.
    
    Args:
        vehicle_data (List[Dict]): List of SUMO vehicle data dictionaries 
                                   containing 'waiting_time' key.
    
    Returns:
        float: Maximum waiting time in seconds. 
               Returns 0 if no valid vehicles are present.
    """
    waiting_times = _validate_sumo_vehicle_data(vehicle_data, 'waiting_time')
    return float(np.max(waiting_times))
