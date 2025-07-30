"""
JSPL Steel Coil Ovality - Deployment Pipeline
==============================================
This script implements a multi-stage pipeline for real-time coil detection,
best frame selection, and ovality calculation from an RTSP stream.
"""

import cv2
import numpy as np
import time
import logging
import json
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from ultralytics import YOLO

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ========================================================
# 1. DEPLOYMENT CONFIGURATION
# ========================================================

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
    score_weights: dict = field(default_factory=lambda: {
        "segmentation": 0.4,
        "centering": 0.3,
        "size": 0.2,
        "quality": 0.1
    })

@dataclass
class CandidateFrame:
    """Holds information about a frame being considered for best frame."""
    frame: np.ndarray
    combined_score: float = 0.0
    segmentation_confidence: float = 0.0
    centering_score: float = 0.0
    size_score: float = 0.0
    quality_score: float = 0.0
    mask: Optional[np.ndarray] = None
    box: Optional[np.ndarray] = None


# ========================================================
# 2. HELPER MODULES
# ========================================================

class OvalityCalculator:
    """Calculates ovality from a segmentation mask."""
    def calculate(self, mask: np.ndarray) -> Optional[float]:
        if mask is None or mask.size == 0:
            return None
        
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            return None

        try:
            ellipse = cv2.fitEllipse(contour)
            (center_x, center_y), (minor_axis, major_axis), angle = ellipse
            if major_axis > 0:
                return (major_axis - minor_axis) / major_axis
        except cv2.error:
            return None
        return None

class CombinedScoreCalculator:
    """Calculates the combined score for a candidate frame."""
    def __init__(self, config: DeploymentConfig, segmentation_model: YOLO):
        self.config = config
        self.seg_model = segmentation_model

    def calculate(self, frame: np.ndarray) -> CandidateFrame:
        candidate = CandidateFrame(frame=frame)
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

        if best_box is not None:
            box_center_x, box_center_y = best_box[0], best_box[1]
            roi_center_x = (self.config.roi[0] + self.config.roi[2]) / 2
            roi_center_y = (self.config.roi[1] + self.config.roi[3]) / 2
            
            dist = np.sqrt((box_center_x - roi_center_x)**2 + (box_center_y - roi_center_y)**2)
            max_dist = np.sqrt(((self.config.roi[2] - self.config.roi[0])/2)**2 + ((self.config.roi[3] - self.config.roi[1])/2)**2)
            candidate.centering_score = max(0, 1 - dist / max_dist) if max_dist > 0 else 0

            box_w, box_h = best_box[2], best_box[3]
            ideal_w, ideal_h = self.config.ideal_coil_size
            size_diff = abs(box_w - ideal_w) / ideal_w + abs(box_h - ideal_h) / ideal_h
            candidate.size_score = max(0, 1 - size_diff / 2)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        candidate.quality_score = min(1.0, laplacian_var / (self.config.blur_threshold * 5))

        weights = self.config.score_weights
        candidate.combined_score = (
            weights['segmentation'] * candidate.segmentation_confidence +
            weights['centering'] * candidate.centering_score +
            weights['size'] * candidate.size_score +
            weights['quality'] * candidate.quality_score
        )
        
        logging.info(f"Frame scores: Cmb={candidate.combined_score:.2f}, Seg={candidate.segmentation_confidence:.2f}, Ctr={candidate.centering_score:.2f}, Size={candidate.size_score:.2f}, Qual={candidate.quality_score:.2f}")
        return candidate


# ========================================================
# 3. DEPLOYMENT PIPELINE
# ========================================================

class DeploymentPipeline:
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.running = False
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=False)
        self.proxy_model = YOLO(self.config.proxy_model_path)
        self.segmentation_model = YOLO(self.config.segmentation_model_path)
        self.score_calculator = CombinedScoreCalculator(config, self.segmentation_model)
        self.ovality_calculator = OvalityCalculator()

        self.in_coil_event = False
        self.detection_buffer: List[bool] = []
        self.best_candidate_frame: Optional[CandidateFrame] = None

        os.makedirs(self.config.output_dir, exist_ok=True)
        logging.info("Deployment pipeline initialized.")

    def _pre_filter(self, frame: np.ndarray) -> bool:
        """Stage 1: High-Speed Pre-Filtering."""
        fg_mask = self.background_subtractor.apply(frame, learningRate=-1)
        motion_score = np.sum(fg_mask > 200)
        
        if motion_score < self.config.motion_threshold:
            return False

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_frame)
        if not (self.config.min_brightness < brightness < self.config.max_brightness):
            return False
            
        laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        if laplacian_var < self.config.blur_threshold:
            return False

        return True

    def _confirm_coil_presence(self, frame: np.ndarray) -> bool:
        """Stage 2: Coil Presence Confirmation."""
        results = self.proxy_model(frame, conf=self.config.proxy_conf_threshold, verbose=False)
        x_min, y_min, x_max, y_max = self.config.roi
        
        if results and results[0].boxes:
            for box in results[0].boxes:
                class_name = self.proxy_model.names[int(box.cls)]
                if class_name in self.config.proxy_target_classes:
                    bx, by, bw, bh = box.xywh[0]
                    if (x_min < bx < x_max) and (y_min < by < y_max):
                        return True
        return False

    def _update_coil_event_state(self, has_coil: bool):
        """Manages the state of the 'Coil Event'."""
        self.detection_buffer.append(has_coil)
        if len(self.detection_buffer) > self.config.detection_buffer_size:
            self.detection_buffer.pop(0)

        if not self.in_coil_event and sum(self.detection_buffer) >= self.config.presence_threshold:
            self.in_coil_event = True
            logging.warning(">>> COIL EVENT STARTED <<<")
        
        elif self.in_coil_event and self.detection_buffer.count(False) >= self.config.absence_threshold:
            self.in_coil_event = False
            logging.warning("<<< COIL EVENT ENDED >>>")
            self._process_best_frame()
            self.detection_buffer.clear()

    def _process_candidate_frame(self, frame: np.ndarray):
        """Stage 3: Processes a single frame during a coil event to find the best one."""
        candidate = self.score_calculator.calculate(frame)
        
        if self.best_candidate_frame is None or candidate.combined_score > self.best_candidate_frame.combined_score:
            self.best_candidate_frame = candidate
            logging.info(f"New best frame found with score: {candidate.combined_score:.2f}")

    def _process_best_frame(self):
        """Stage 4: Final processing of the best frame."""
        if self.best_candidate_frame and self.best_candidate_frame.mask is not None:
            logging.warning(f"Final best frame selected. Score: {self.best_candidate_frame.combined_score:.2f}")
            ovality = self.ovality_calculator.calculate(self.best_candidate_frame.mask)
            self._save_results(self.best_candidate_frame, ovality)
        else:
            logging.error("Coil event ended, but no valid best frame was selected.")
        
        self.best_candidate_frame = None

    def _save_results(self, candidate: CandidateFrame, ovality: Optional[float]):
        """Saves the best frame image, its mask, and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.join(self.config.output_dir, f"coil_{timestamp}")
        
        cv2.imwrite(f"{base_filename}_image.jpg", candidate.frame)
        
        if candidate.mask is not None:
            mask_img = (candidate.mask * 255).astype(np.uint8)
            cv2.imwrite(f"{base_filename}_mask.png", mask_img)

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
        
        logging.warning(f"Results saved to {base_filename}_...")

    def run(self):
        """Starts the main processing loop for the RTSP stream."""
        self.running = True
        logging.info(f"Connecting to RTSP stream: {self.config.rtsp_url}")

        cap = cv2.VideoCapture(self.config.rtsp_url)
        if not cap.isOpened():
            logging.error("Failed to open RTSP stream. Exiting.")
            return

        frame_count = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame from stream. Reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(self.config.rtsp_url)
                continue

            frame_count += 1
            if frame_count % 2 != 0:
                continue

            if self._pre_filter(frame):
                is_coil_present = self._confirm_coil_presence(frame)
                self._update_coil_event_state(is_coil_present)

                if self.in_coil_event:
                    self._process_candidate_frame(frame)
        
        cap.release()
        logging.info("Deployment pipeline stopped.")

    def stop(self):
        """Stops the processing loop."""
        self.running = False

# ========================================================
# MAIN EXECUTION
# ========================================================

if __name__ == '__main__':
    config = DeploymentConfig()
    pipeline = DeploymentPipeline(config)
    
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.stop()
        logging.info("Process interrupted by user.")
