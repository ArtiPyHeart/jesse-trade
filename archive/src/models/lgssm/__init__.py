"""
Linear Gaussian State Space Model (LGSSM) module.

This module provides a production-ready implementation of LGSSM for
time series feature extraction in quantitative trading.
"""

from .lgssm import LGSSM, LGSSMConfig
from .kalman_filter import KalmanFilter

__all__ = [
    'LGSSM',
    'LGSSMConfig',
    'KalmanFilter',
]

__version__ = '1.0.0'