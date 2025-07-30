"""
Scoring module for frame evaluation and candidate frame selection.
Provides methods to calculate various quality scores for frames.
"""

import cv2
import numpy as np
import traceback
from typing import Optional, Dict
import logging
from ultralytics import YOLO
from ripikutils.logsman import setup_logger

from .config import DeploymentConfig, CandidateFrame


class CombinedScoreCalculator:
    """Calculates the combined score for a candidate frame."""
    
    def __init__(self, config: DeploymentConfig, segmentation_model: YOLO):
        """
        Initialize the score calculator.
        
        Args:
            config: Deployment configuration
            segmentation_model: YOLO segmentation model
        """
        self.config = config
        self.seg_model = segmentation_model
        self.logger = setup_logger(name=__name__)
    
    def calculate(self, frame: np.ndarray) -> CandidateFrame:
        """
        Calculate comprehensive scores for a frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            CandidateFrame with all calculated scores
        """
        try:
            candidate = CandidateFrame(frame=frame)
            
            # Get segmentation results
            results = self.seg_model(frame, conf=self.config.seg_conf_threshold, verbose=False)
            
            best_conf = 0.0
            best_mask = None
            best_box = None
            
            if results and results[0].masks:
                masks = results[0].masks
                boxes = results[0].boxes
                
                for i in range(len(masks)):
                    if boxes[i].conf > best_conf:
                        best_conf = float(boxes[i].conf)
                        best_mask = masks.data[i].cpu().numpy()
                        best_box = boxes.xywh[i].cpu().numpy()

            candidate.segmentation_confidence = best_conf
            candidate.mask = best_mask
            candidate.box = best_box

            # Calculate individual scores
            if best_box is not None:
                candidate.centering_score = self._calculate_centering_score(best_box)
                candidate.size_score = self._calculate_size_score(best_box)
            else:
                candidate.centering_score = 0.0
                candidate.size_score = 0.0
            
            candidate.quality_score = self._calculate_quality_score(frame)
            
            # Calculate combined score
            candidate.combined_score = self._calculate_combined_score(candidate)
            
            self.logger.info(
                f"Frame scores: Cmb={candidate.combined_score:.2f}, "
                f"Seg={candidate.segmentation_confidence:.2f}, "
                f"Ctr={candidate.centering_score:.2f}, "
                f"Size={candidate.size_score:.2f}, "
                f"Qual={candidate.quality_score:.2f}"
            )
            
            return candidate
            
        except Exception as e:
            self.logger.error(f"Error calculating frame scores: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return a default candidate with zero scores
            return CandidateFrame(frame=frame)
    
    def _calculate_centering_score(self, box: np.ndarray) -> float:
        """Calculate how well the detection is centered in the ROI."""
        try:
            box_center_x, box_center_y = box[0], box[1]
            roi_center_x = (self.config.roi[0] + self.config.roi[2]) / 2
            roi_center_y = (self.config.roi[1] + self.config.roi[3]) / 2
            
            # Calculate distance from ROI center
            dist = np.sqrt((box_center_x - roi_center_x)**2 + (box_center_y - roi_center_y)**2)
            max_dist = np.sqrt(
                ((self.config.roi[2] - self.config.roi[0])/2)**2 + 
                ((self.config.roi[3] - self.config.roi[1])/2)**2
            )
            
            # Score decreases with distance from center
            centering_score = max(0, 1 - dist / max_dist) if max_dist > 0 else 0
            return centering_score
            
        except Exception as e:
            self.logger.error(f"Error calculating centering score: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0
    
    def _calculate_size_score(self, box: np.ndarray) -> float:
        """Calculate how close the detection size is to ideal size."""
        try:
            box_w, box_h = box[2], box[3]
            ideal_w, ideal_h = self.config.ideal_coil_size
            
            # Calculate size difference
            size_diff = abs(box_w - ideal_w) / ideal_w + abs(box_h - ideal_h) / ideal_h
            size_score = max(0, 1 - size_diff / 2)
            return size_score
            
        except Exception as e:
            self.logger.error(f"Error calculating size score: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0
    
    def _calculate_quality_score(self, frame: np.ndarray) -> float:
        """Calculate image quality score based on sharpness."""
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            
            # Normalize quality score
            quality_score = min(1.0, laplacian_var / (self.config.blur_threshold * 5))
            return quality_score
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0
    
    def _calculate_combined_score(self, candidate: CandidateFrame) -> float:
        """Calculate weighted combined score."""
        try:
            weights = self.config.score_weights
            
            combined_score = (
                weights['segmentation'] * candidate.segmentation_confidence +
                weights['centering'] * candidate.centering_score +
                weights['size'] * candidate.size_score +
                weights['quality'] * candidate.quality_score
            )
            
            return combined_score
            
        except Exception as e:
            self.logger.error(f"Error calculating combined score: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0


class QualityAnalyzer:
    """Analyzes frame quality for pre-filtering."""
    
    def __init__(self, config: DeploymentConfig):
        """
        Initialize the quality analyzer.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.logger = setup_logger(name=__name__)
    
    def analyze_frame_quality(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Analyze various quality metrics of a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            metrics = {
                'brightness': np.mean(gray_frame),
                'contrast': np.std(gray_frame),
                'sharpness': cv2.Laplacian(gray_frame, cv2.CV_64F).var(),
                'noise': self._estimate_noise(gray_frame)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing frame quality: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'brightness': 0.0,
                'contrast': 0.0,
                'sharpness': 0.0,
                'noise': 0.0
            }
    
    def _estimate_noise(self, gray_frame: np.ndarray) -> float:
        """Estimate noise level in the image."""
        try:
            # Simple noise estimation using high-pass filter
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            filtered = cv2.filter2D(gray_frame, cv2.CV_64F, kernel)
            noise_level = np.std(filtered)
            return noise_level
            
        except Exception as e:
            self.logger.error(f"Error estimating noise: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0 