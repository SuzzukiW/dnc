# quick_tests/__init__.py

"""
Quick testing module for traffic light control project.
This package contains configuration and scripts for running quick tests
of the traffic light control system.
"""

__version__ = '0.1.0'

from .config_loader import load_config

__all__ = ['load_config']