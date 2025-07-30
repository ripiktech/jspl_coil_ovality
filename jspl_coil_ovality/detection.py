"""
Detection module for coil detection and pre-filtering.
Handles motion detection, coil presence confirmation, and ROI validation.
"""

import cv2
import numpy as np
import traceback
from typing import List, Optional
import logging
from ultralytics import YOLO
from ripikutils.logsman import setup_logger

from .config import DeploymentConfig
from .scoring import QualityAnalyzer


class PreFilter:
    """Handles high-speed pre-filtering of frames."""
    
    def __init__(self, config: DeploymentConfig):
        """
        Initialize the pre-filter.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=50, 
            varThreshold=50, 
            detectShadows=False
        )
        self.quality_analyzer = QualityAnalyzer(config)
        self.logger = setup_logger(name=__name__)
    
    def should_process_frame(self, frame: np.ndarray) -> bool:
        """
        Determine if a frame should be processed further.
        
        Args:
            frame: Input frame
            
        Returns:
            True if frame should be processed, False otherwise
        """
        try:
            # Check motion
            fg_mask = self.background_subtractor.apply(frame, learningRate=-1)
            motion_score = np.sum(fg_mask > 200)
            
            if motion_score < self.config.motion_threshold:
                return False

            # Check brightness
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_frame)
            if not (self.config.min_brightness < brightness < self.config.max_brightness):
                return False
                
            # Check blur/sharpness
            laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            if laplacian_var < self.config.blur_threshold:
                return False

            return True
            
        except Exception as e:
            self.logger.error(f"Error in pre-filtering: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False


class CoilDetector:
    """Handles steel coil detection using fine-tuned YOLO models."""
    
    def __init__(self, config: DeploymentConfig):
        """
        Initialize the coil detector.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.logger = setup_logger(name=__name__)
        
        try:
            self.proxy_model = YOLO(self.config.proxy_model_path)
            self.logger.info(f"Loaded fine-tuned detection model: {self.config.proxy_model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load detection model: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def confirm_coil_presence(self, frame: np.ndarray) -> bool:
        """
        Confirm if a steel coil is present in the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            True if steel coil is detected, False otherwise
        """
        try:
            results = self.proxy_model(frame, conf=self.config.proxy_conf_threshold, verbose=False)
            x_min, y_min, x_max, y_max = self.config.roi
            
            if results and results[0].boxes:
                for box in results[0].boxes:
                    class_name = self.proxy_model.names[int(box.cls)]
                    confidence = float(box.conf)
                    
                    if class_name in self.config.proxy_target_classes:
                        bx, by, bw, bh = box.xywh[0]
                        if (x_min < bx < x_max) and (y_min < by < y_max):
                            self.logger.debug(f"Steel coil detected: {class_name} (conf: {confidence:.3f}) at ({bx:.1f}, {by:.1f})")
                            return True
                        else:
                            self.logger.debug(f"Steel coil detected outside ROI: {class_name} at ({bx:.1f}, {by:.1f})")
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in coil detection: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def validate_roi(self, detection_center: tuple) -> bool:
        """
        Validate if a detection is within the ROI.
        
        Args:
            detection_center: (x, y) coordinates of detection center
            
        Returns:
            True if within ROI, False otherwise
        """
        try:
            x, y = detection_center
            x_min, y_min, x_max, y_max = self.config.roi
            
            return (x_min < x < x_max) and (y_min < y < y_max)
            
        except Exception as e:
            self.logger.error(f"Error in ROI validation: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False


class EventStateManager:
    """Manages the state of steel coil detection events."""
    
    def __init__(self, config: DeploymentConfig):
        """
        Initialize the event state manager.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.in_coil_event = False
        self.detection_buffer: List[bool] = []
        self.logger = setup_logger(name=__name__)
    
    def update_state(self, has_coil: bool) -> bool:
        """
        Update the steel coil event state based on detection results.
        
        Args:
            has_coil: Whether steel coil was detected in current frame
            
        Returns:
            True if state changed, False otherwise
        """
        try:
            self.detection_buffer.append(has_coil)
            
            if len(self.detection_buffer) > self.config.detection_buffer_size:
                self.detection_buffer.pop(0)

            state_changed = False
            
            # Check for event start
            if not self.in_coil_event and sum(self.detection_buffer) >= self.config.presence_threshold:
                self.in_coil_event = True
                self.logger.warning(">>> STEEL COIL EVENT STARTED <<<")
                state_changed = True
            
            # Check for event end
            elif self.in_coil_event and self.detection_buffer.count(False) >= self.config.absence_threshold:
                self.in_coil_event = False
                self.logger.warning("<<< STEEL COIL EVENT ENDED <<<")
                self.detection_buffer.clear()
                state_changed = True
            
            return state_changed
            
        except Exception as e:
            self.logger.error(f"Error updating event state: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def is_in_event(self) -> bool:
        """Check if currently in a steel coil event."""
        return self.in_coil_event
    
    def reset(self):
        """Reset the event state."""
        try:
            self.in_coil_event = False
            self.detection_buffer.clear()
            self.logger.info("Steel coil event state reset")
        except Exception as e:
            self.logger.error(f"Error resetting event state: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}") 