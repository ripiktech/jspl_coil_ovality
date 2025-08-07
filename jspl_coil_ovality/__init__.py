"""
JSPL Steel Coil Ovality Detection Package

A comprehensive real-time steel coil detection and ovality analysis system.
"""

__version__ = "1.0.0"
__author__ = "Dhvanil"

from .config import DeploymentConfig, CandidateFrame
from .pipeline import DeploymentPipeline
from .detection import PreFilter, CoilDetector, EventStateManager
from .scoring import CombinedScoreCalculator
from .ovality_calculator import OvalityCalculator

__all__ = [
    "DeploymentConfig",
    "CandidateFrame", 
    "DeploymentPipeline",
    "PreFilter",
    "CoilDetector",
    "EventStateManager",
    "CombinedScoreCalculator",
    "OvalityCalculator"
]
