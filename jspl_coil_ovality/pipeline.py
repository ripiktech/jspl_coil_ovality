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
import sys
import traceback
from datetime import datetime
from typing import Optional

from ultralytics import YOLO
from ripikutils.logsman import setup_logger, LoggerWriter
from ripikutils.stream import VideoStream

from .config import DeploymentConfig, CandidateFrame
from .detection import PreFilter, CoilDetector, EventStateManager
from .scoring import CombinedScoreCalculator
from .ovality_calculator import OvalityCalculator


class DeploymentPipeline:
    """Main deployment pipeline orchestrating all stages for steel coil detection."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.running = False
        
        self.logger = setup_logger(
            name=__name__,
            log_filename=config.log_filename,
            prefix=config.log_prefix,
            postfix=config.log_postfix,
            log_dir=config.log_dir,
            max_log_size=config.max_log_size,
            backup_count=config.backup_count,
            logging_level=getattr(logging, config.logging_level.upper())
        )
        
        sys.stdout = LoggerWriter(self.logger, logging.INFO)
        sys.stderr = LoggerWriter(self.logger, logging.ERROR)
        
        self.pre_filter = PreFilter(config)
        self.coil_detector = CoilDetector(config)
        self.event_manager = EventStateManager(config)
        
        self.segmentation_model = YOLO(self.config.yolo_model_path)
        self.logger.info(f"Loaded unified YOLOv11n-seg model for segmentation: {self.config.yolo_model_path}")
        
        self.score_calculator = CombinedScoreCalculator(config, self.segmentation_model)
        self.ovality_calculator = OvalityCalculator()
        
        self.best_candidate_frame: Optional[CandidateFrame] = None
        
        self.logger.info("Steel coil deployment pipeline initialized.")
    
    def process_frame(self, frame: np.ndarray) -> bool:
        try:
            # Stage 1: Pre-filtering
            if not self.pre_filter.should_process_frame(frame): return False
            
            # Stage 2: Steel coil presence confirmation
            is_coil_present = self.coil_detector.confirm_coil_presence(frame)
            state_changed = self.event_manager.update_state(is_coil_present)
            
            # Stage 3: Best frame selection (during steel coil event)
            if self.event_manager.is_in_event():
                self._process_candidate_frame(frame)
            
            # Stage 4: Final processing (when event ends)
            if state_changed and not self.event_manager.is_in_event():
                self._process_best_frame()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _process_candidate_frame(self, frame: np.ndarray):
        try:
            candidate = self.score_calculator.calculate(frame)
            
            if self.best_candidate_frame is None or candidate.combined_score > self.best_candidate_frame.combined_score:
                self.best_candidate_frame = candidate
                self.logger.info(f"New best frame found with score: {candidate.combined_score:.2f}")
        except Exception as e:
            self.logger.error(f"Error processing candidate frame: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _process_best_frame(self):
        if self.best_candidate_frame and self.best_candidate_frame.mask is not None:
            self.logger.warning(f"Final best frame selected. Score: {self.best_candidate_frame.combined_score:.2f}")
            
            try:
                ovality = self.ovality_calculator.calculate(self.best_candidate_frame.mask)
                self._save_results(self.best_candidate_frame, ovality)
            except Exception as e:
                self.logger.error(f"Error calculating ovality: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            self.logger.error("Event ended, but no valid best frame was selected.")
        
        self.best_candidate_frame = None
    
    def _save_results(self, candidate: CandidateFrame, ovality: Optional[float]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.join(self.config.output_dir, f"steel_coil_{timestamp}")
        
        try:
            cv2.imwrite(f"{base_filename}_image.jpg", candidate.frame)
            
            if candidate.mask is not None:
                mask_img = (candidate.mask * 255).astype(np.uint8)
                cv2.imwrite(f"{base_filename}_mask.png", mask_img)

            metadata = {
                "timestamp": datetime.now().isoformat(),
                "ovality": ovality,
                "ovality_interpretation": self._interpret_ovality(ovality),
                "scores": {
                    "combined": candidate.combined_score,
                    "segmentation_confidence": candidate.segmentation_confidence,
                    "centering": candidate.centering_score,
                    "size": candidate.size_score,
                },
                "model_info": {
                    "unified_model": self.config.yolo_model_path,
                    "model_type": "YOLOv11n-seg (detection + segmentation)"
                }
            }
            
            with open(f"{base_filename}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=4)
            
            self.logger.warning(f"Results saved to {base_filename}_...")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _interpret_ovality(self, ovality: Optional[float]) -> str:
        if ovality is None:
            return "Could not calculate ovality"
        elif ovality < 0.05:
            return "Circular (excellent)"
        elif ovality < 0.1:
            return "Slightly oval (good)"
        elif ovality < 0.2:
            return "Moderately oval (acceptable)"
        else:
            return "Highly oval (needs attention)"
    
    def run(self):
        self.running = True
        self.logger.info(f"Connecting to RTSP stream: {self.config.rtsp_url}")

        stream = VideoStream(self.config.rtsp_url)
        
        frame_count = 0
        while self.running:
            try:
                frame = stream.read()
                if frame is None:
                    self.logger.warning("Failed to read frame from stream. Reconnecting...")
                    time.sleep(1.0)
                    continue

                frame_count += 1
                # if frame_count % 2 != 0: continue
                self.process_frame(frame)
                
            except Exception as e:
                self.logger.error(f"Error processing frame {frame_count}: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        stream.stop()
        self.logger.info("Steel coil deployment pipeline stopped.")
    
    def stop(self):
        self.running = False 