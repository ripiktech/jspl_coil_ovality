"""
Pipeline module for the JSPL Steel Coil Ovality Detection system.
Orchestrates all components and manages the multi-stage processing.
"""

import cv2
import numpy as np
import time
import logging
import json
import os
from datetime import datetime
from typing import Optional

from ultralytics import YOLO

from .config import DeploymentConfig, CandidateFrame
from .detection import PreFilter, CoilDetector, EventStateManager
from .scoring import CombinedScoreCalculator
from .ovality_calculator import OvalityCalculator


class DeploymentPipeline:
    """Main deployment pipeline orchestrating all stages."""
    
    def __init__(self, config: DeploymentConfig):
        """
        Initialize the deployment pipeline.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.running = False
        
        # Initialize components
        self.pre_filter = PreFilter(config)
        self.coil_detector = CoilDetector(config)
        self.event_manager = EventStateManager(config)
        self.segmentation_model = YOLO(self.config.segmentation_model_path)
        self.score_calculator = CombinedScoreCalculator(config, self.segmentation_model)
        self.ovality_calculator = OvalityCalculator()
        
        # State variables
        self.best_candidate_frame: Optional[CandidateFrame] = None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Deployment pipeline initialized.")
    
    def process_frame(self, frame: np.ndarray) -> bool:
        """
        Process a single frame through all stages.
        
        Args:
            frame: Input frame
            
        Returns:
            True if frame was processed, False if rejected
        """
        # Stage 1: Pre-filtering
        if not self.pre_filter.should_process_frame(frame):
            return False
        
        # Stage 2: Coil presence confirmation
        is_coil_present = self.coil_detector.confirm_coil_presence(frame)
        state_changed = self.event_manager.update_state(is_coil_present)
        
        # Stage 3: Best frame selection (during coil event)
        if self.event_manager.is_in_event():
            self._process_candidate_frame(frame)
        
        # Stage 4: Final processing (when event ends)
        if state_changed and not self.event_manager.is_in_event():
            self._process_best_frame()
        
        return True
    
    def _process_candidate_frame(self, frame: np.ndarray):
        """Process a candidate frame during a coil event."""
        candidate = self.score_calculator.calculate(frame)
        
        if self.best_candidate_frame is None or candidate.combined_score > self.best_candidate_frame.combined_score:
            self.best_candidate_frame = candidate
            self.logger.info(f"New best frame found with score: {candidate.combined_score:.2f}")
    
    def _process_best_frame(self):
        """Process the best frame when coil event ends."""
        if self.best_candidate_frame and self.best_candidate_frame.mask is not None:
            self.logger.warning(f"Final best frame selected. Score: {self.best_candidate_frame.combined_score:.2f}")
            ovality = self.ovality_calculator.calculate(self.best_candidate_frame.mask)
            self._save_results(self.best_candidate_frame, ovality)
        else:
            self.logger.error("Coil event ended, but no valid best frame was selected.")
        
        self.best_candidate_frame = None
    
    def _save_results(self, candidate: CandidateFrame, ovality: Optional[float]):
        """Save the best frame image, mask, and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.join(self.config.output_dir, f"coil_{timestamp}")
        
        # Save image
        cv2.imwrite(f"{base_filename}_image.jpg", candidate.frame)
        
        # Save mask
        if candidate.mask is not None:
            mask_img = (candidate.mask * 255).astype(np.uint8)
            cv2.imwrite(f"{base_filename}_mask.png", mask_img)

        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "ovality": ovality,
            "scores": {
                "combined": candidate.combined_score,
                "segmentation_confidence": candidate.segmentation_confidence,
                "centering": candidate.centering_score,
                "size": candidate.size_score,
                "quality": candidate.quality_score
            }
        }
        
        with open(f"{base_filename}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        self.logger.warning(f"Results saved to {base_filename}_...")
    
    def run(self):
        """Start the main processing loop for the RTSP stream."""
        self.running = True
        self.logger.info(f"Connecting to RTSP stream: {self.config.rtsp_url}")

        cap = cv2.VideoCapture(self.config.rtsp_url)
        if not cap.isOpened():
            self.logger.error("Failed to open RTSP stream. Exiting.")
            return

        frame_count = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.logger.warning("Failed to read frame from stream. Reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(self.config.rtsp_url)
                continue

            frame_count += 1
            if frame_count % 2 != 0:  # Process every 2nd frame
                continue

            self.process_frame(frame)
        
        cap.release()
        self.logger.info("Deployment pipeline stopped.")
    
    def stop(self):
        """Stop the processing loop."""
        self.running = False 