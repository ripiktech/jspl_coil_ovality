"""
JSPL Steel Coil Ovality Detection Package
=========================================

A comprehensive system for real-time steel coil detection and ovality analysis
from RTSP video streams using multi-stage processing pipeline.

Main Components:
- DeploymentPipeline: Main orchestration class
- DeploymentConfig: Configuration management
- OvalityCalculator: Ovality calculation from segmentation masks
- PreFilter: High-speed frame pre-filtering
- CoilDetector: YOLO-based coil detection
- CombinedScoreCalculator: Frame scoring and selection
"""

from .config import DeploymentConfig, CandidateFrame
from .pipeline import DeploymentPipeline
from .ovality_calculator import OvalityCalculator
from .detection import PreFilter, CoilDetector, EventStateManager
from .scoring import CombinedScoreCalculator

__version__ = "1.0.0"
__author__ = "Ripik Tech"
__description__ = "JSPL Steel Coil Ovality Detection System"

__all__ = [
    "DeploymentPipeline",
    "DeploymentConfig", 
    "CandidateFrame",
    "OvalityCalculator",
    "PreFilter",
    "CoilDetector", 
    "EventStateManager",
    "CombinedScoreCalculator"
]
