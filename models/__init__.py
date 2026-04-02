"""
Models package for AutoMap project.

Exports model classes implementing the BaseTimeSeriesModel interface.
"""

from .base_model import BaseTimeSeriesModel
from .ardl_model import ARDLModel
from .var_model import VARModel

__all__ = [
    'BaseTimeSeriesModel',
    'ARDLModel',
    'VARModel',
]
