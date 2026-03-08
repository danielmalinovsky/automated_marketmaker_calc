"""
Automated Market Maker Calculation Package

Modules:
    calc_v2: Core AMM payoff calculations and simulations
    research_visualizations: Research paper visualization suite
"""

from .calc_v2 import (
    PriceSim,
    VolumeSim,
    Payoff,
    Visualization,
    Analytics
)


__all__ = [
    'PriceSim',
    'VolumeSim',
    'Payoff',
    'Visualization',
    'Analytics',
    'VolumePathAnalysis',
]

__version__ = '2.0.0'
