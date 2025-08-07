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
    rtsp_url: str = "rtsp://admin:Ripik.ai@43.204.215.195:8554/angul_jspl"
    output_dir: str = "data/deployment_output"
    log_dir: str = "logs"
    log_filename: Optional[str] = None
    log_prefix: str = "steel_coil"
    log_postfix: str = "deployment"
    max_log_size: int = 50
    backup_count: int = 5
    logging_level: str = "INFO"

    # ClientMeta Configuration
    client_id: str = "default_client"
    use_case: str = "steel_coil_ovality"
    aws_bucket_name: str = "rpk-clnt-in"
    client_meta_version: int = 1
    client_meta_env: str = "R_PROD"
    client_meta_stage_env: str = "R_STAGE"

    # Stage 1: Pre-filtering Configuration
    motion_threshold: float = 1000.0
    blur_threshold: float = 100.0
    min_brightness: int = 40
    max_brightness: int = 210
    noise_classifier_path: str = "models/noise_classifier.pt"
    noise_conf_threshold: float = 0.5
    enable_noise_filtering: bool = True

    # Stage 2 & 3: Unified YOLOv11n-seg Model for Detection and Segmentation
    yolo_model_path: str = "models/yolov11n-seg-steel-coil.pt"
    yolo_detection_conf_threshold: float = 0.4  # For presence detection
    yolo_segmentation_conf_threshold: float = 0.6  # For final segmentation
    yolo_target_classes: List[str] = field(default_factory=lambda: ["steel_coil"])
    roi: Tuple[int, int, int, int] = (750, 0, 2056, 1440)
    detection_buffer_size: int = 10
    presence_threshold: int = 3
    absence_threshold: int = 5
    ideal_coil_size: Tuple[int, int] = (1000, 1000)
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        "segmentation": 0.6,
        "centering": 0.25,
        "size": 0.15
    })

@dataclass
class CandidateFrame:
    """Holds information about a frame being considered for best frame."""
    frame: Optional[object] = None  # np.ndarray
    combined_score: float = 0.0
    segmentation_confidence: float = 0.0
    centering_score: float = 0.0
    size_score: float = 0.0
    mask: Optional[object] = None  # np.ndarray
    box: Optional[object] = None  # np.ndarray 