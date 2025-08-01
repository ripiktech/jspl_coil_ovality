"""
Detection module for coil detection and pre-filtering.
Handles motion detection, coil presence confirmation, and ROI validation.
"""

import cv2
import numpy as np
import traceback
import torch
import torch.nn.functional as F
from typing import List
from ultralytics import YOLO
from ripikutils.logsman import setup_logger
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

from .config import DeploymentConfig


class NoiseClassifier:
    """Handles binary classification for frame noise detection."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = setup_logger(name=__name__)
        
        if not config.enable_noise_filtering:
            self.model = None
            self.logger.info("Noise filtering disabled")
            return
            
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = torch.load(config.noise_classifier_path, map_location=self.device)
            self.model.eval()
            self.transformations = Compose([
                Resize((512, 512)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.logger.info(f"Loaded noise classifier: {config.noise_classifier_path}")
        except Exception as e:
            self.logger.error(f"Failed to load noise classifier: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.model = None
    
    def is_frame_noisy(self, frame: np.ndarray) -> bool:
        try:
            if self.model is None: 
                return False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)            
            image_tensor = self.transformations(pil_image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = F.softmax(output, dim=1)
                good_frame_prob = probabilities[0, 1].item()
            
            is_noisy = good_frame_prob < self.config.noise_conf_threshold
            
            if is_noisy:
                self.logger.debug(f"Frame classified as noisy (good_prob: {good_frame_prob:.3f})")
            
            return is_noisy
            
        except Exception as e:
            self.logger.error(f"Error in noise classification: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False


class PreFilter:
    """Handles high-speed pre-filtering of frames."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=False)
        self.noise_classifier = NoiseClassifier(config)
        self.logger = setup_logger(name=__name__)
    
    def should_process_frame(self, frame: np.ndarray) -> bool:
        try:
            fg_mask = self.background_subtractor.apply(frame, learningRate=-1)
            motion_score = np.sum(fg_mask > 200)
            if motion_score < self.config.motion_threshold: return False

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_frame)
            if not (self.config.min_brightness < brightness < self.config.max_brightness): return False
                
            laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            if laplacian_var < self.config.blur_threshold: return False
            if (self.config.enable_noise_filtering and self.noise_classifier.is_frame_noisy(frame)): return False

            return True
            
        except Exception as e:
            self.logger.error(f"Error in pre-filtering: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False


class CoilDetector:
    """Handles steel coil detection using fine-tuned YOLO models."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = setup_logger(name=__name__)
        self.yolo_detection_model = YOLO(self.config.yolo_detection_model_path)
        self.logger.info(f"Loaded YOLOv8n-steel-coil: {self.config.yolo_detection_model_path}")
    
    def confirm_coil_presence(self, frame: np.ndarray) -> bool:
        try:
            results = self.yolo_detection_model(frame, conf=self.config.yolo_detection_conf_threshold, verbose=False)
            x_min, y_min, x_max, y_max = self.config.roi
            
            if results and results[0].boxes:
                for box in results[0].boxes:
                    class_name = self.yolo_detection_model.names[int(box.cls)]
                    confidence = float(box.conf)
                    
                    if class_name in self.config.yolo_detection_target_classes:
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
        self.config = config
        self.in_coil_event = False
        self.detection_buffer: List[bool] = []
        self.logger = setup_logger(name=__name__)
    
    def update_state(self, has_coil: bool) -> bool:
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
        return self.in_coil_event
    
    def reset(self):
        try:
            self.in_coil_event = False
            self.detection_buffer.clear()
            self.logger.info("Steel coil event state reset")
        except Exception as e:
            self.logger.error(f"Error resetting event state: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}") 