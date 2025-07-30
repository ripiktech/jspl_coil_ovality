"""
Configuration module for JSPL Steel Coil Ovality Detection Pipeline.
Contains all configuration classes and settings.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
import os


@dataclass
class DeploymentConfig:
    """Configuration for the deployment pipeline."""
    # RTSP Stream Configuration
    rtsp_url: str = "rtsp://admin:Ripik.ai@43.204.215.195:8554/angul_jspl"
    output_dir: str = "data/deployment_output"

    # Stage 1: Pre-filtering Configuration
    motion_threshold: float = 1000.0
    blur_threshold: float = 100.0
    min_brightness: int = 40
    max_brightness: int = 210

    # Stage 2: Coil Presence Confirmation
    proxy_model_path: str = "yolov8n.pt"
    proxy_conf_threshold: float = 0.30
    proxy_target_classes: List[str] = field(default_factory=lambda: ["toilet", "bowl", "vase"])
    roi: Tuple[int, int, int, int] = (750, 0, 2056, 1440)  # (x_min, y_min, x_max, y_max)
    detection_buffer_size: int = 10
    presence_threshold: int = 3
    absence_threshold: int = 5

    # Stage 3: Best Frame Selection
    segmentation_model_path: str = "models/yolov11n-seg.pt"  # Your fine-tuned model
    seg_conf_threshold: float = 0.5
    ideal_coil_size: Tuple[int, int] = (1000, 1000)
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        "segmentation": 0.4,
        "centering": 0.3,
        "size": 0.2,
        "quality": 0.1
    })

    def __post_init__(self):
        """Validate configuration after initialization."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Validate score weights sum to 1.0
        if abs(sum(self.score_weights.values()) - 1.0) > 0.001:
            raise ValueError("Score weights must sum to 1.0")


@dataclass
class CandidateFrame:
    """Holds information about a frame being considered for best frame."""
    frame: Optional[object] = None  # np.ndarray
    combined_score: float = 0.0
    segmentation_confidence: float = 0.0
    centering_score: float = 0.0
    size_score: float = 0.0
    quality_score: float = 0.0
    mask: Optional[object] = None  # np.ndarray
    box: Optional[object] = None  # np.ndarray 