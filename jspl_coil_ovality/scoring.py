"""
Scoring module for frame evaluation and candidate frame selection.
Provides methods to calculate various quality scores for frames.
"""

import cv2
import numpy as np
import traceback
from ultralytics import YOLO
from ripikutils.logsman import setup_logger

from .config import DeploymentConfig, CandidateFrame


class CombinedScoreCalculator:
    """Calculates the combined score for a candidate frame."""
    
    def __init__(self, config: DeploymentConfig, yolo_segmentation_model: YOLO):
        self.config = config
        self.yolo_segmentation_model = yolo_segmentation_model
        self.logger = setup_logger(name=__name__)
    
    def calculate(self, frame: np.ndarray) -> CandidateFrame:
        try:
            candidate = CandidateFrame(frame=frame)
            results = self.yolo_segmentation_model(frame, conf=self.config.yolo_segmentation_conf_threshold, verbose=False)
            best_conf, best_mask, best_box = 0.0, None, None
            
            if results and results[0].masks:
                masks = results[0].masks
                boxes = results[0].boxes
                
                for i in range(len(masks)):
                    if boxes[i].conf > best_conf:
                        best_conf, best_mask, best_box = float(boxes[i].conf), masks.data[i].cpu().numpy(), boxes.xywh[i].cpu().numpy()

            candidate.segmentation_confidence, candidate.mask, candidate.box = best_conf, best_mask, best_box

            if best_box is not None:
                candidate.centering_score = self._calculate_centering_score(best_box)
                candidate.size_score = self._calculate_size_score(best_box)
            else:
                candidate.centering_score = 0.0
                candidate.size_score = 0.0
            
            candidate.combined_score = self._calculate_combined_score(candidate)
            
            self.logger.info(
                f"Frame scores: Cmb={candidate.combined_score:.2f}, "
                f"Seg={candidate.segmentation_confidence:.2f}, "
                f"Ctr={candidate.centering_score:.2f}, "
                f"Size={candidate.size_score:.2f}"
            )
            
            return candidate
            
        except Exception as e:
            self.logger.error(f"Error calculating frame scores: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return CandidateFrame(frame=frame)
    
    def _calculate_centering_score(self, box: np.ndarray) -> float:
        """Calculate how well the detection is centered in the ROI."""
        try:
            box_center_x, box_center_y = box[0], box[1]
            roi_center_x = (self.config.roi[0] + self.config.roi[2]) / 2
            roi_center_y = (self.config.roi[1] + self.config.roi[3]) / 2
            
            dist = np.sqrt((box_center_x - roi_center_x)**2 + (box_center_y - roi_center_y)**2)
            max_dist = np.sqrt(
                ((self.config.roi[2] - self.config.roi[0])/2)**2 + 
                ((self.config.roi[3] - self.config.roi[1])/2)**2
            )
            
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
            size_diff = abs(box_w - ideal_w) / ideal_w + abs(box_h - ideal_h) / ideal_h
            size_score = max(0, 1 - size_diff / 2)
            return size_score
            
        except Exception as e:
            self.logger.error(f"Error calculating size score: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0
    
    def _calculate_combined_score(self, candidate: CandidateFrame) -> float:
        """Calculate weighted combined score."""
        try:
            weights = self.config.score_weights
            
            combined_score = (
                weights['segmentation'] * candidate.segmentation_confidence +
                weights['centering'] * candidate.centering_score +
                weights['size'] * candidate.size_score
            )
            
            return combined_score
            
        except Exception as e:
            self.logger.error(f"Error calculating combined score: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0
