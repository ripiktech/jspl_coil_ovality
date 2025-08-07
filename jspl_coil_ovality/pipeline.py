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
import boto3
import asyncio
from datetime import datetime
from typing import Optional

from ultralytics import YOLO
from ripikutils.logsman import setup_logger, LoggerWriter
from ripikutils.stream import VideoStream
from ripikutils.response_handler import verify_response_mapping

from .config import DeploymentConfig, CandidateFrame
from .detection import PreFilter, CoilDetector, EventStateManager
from .scoring import CombinedScoreCalculator
from .ovality_calculator import OvalityCalculator


class DeploymentPipeline:
    """Main deployment pipeline orchestrating all stages for steel coil detection."""
    
    def __init__(self, config: DeploymentConfig, client_meta_wrapper=None, client_meta=None):
        self.config = config
        self.running = False
        self.client_meta_wrapper = client_meta_wrapper
        self.client_meta = client_meta
        
        # Initialize S3 client
        self.s3 = boto3.client('s3')
        
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
        
        if self.client_meta_wrapper is None or self.client_meta is None:
            raise Exception("ClientMeta connection required.")
        
        self.pre_filter = PreFilter(config)
        self.coil_detector = CoilDetector(config)
        self.event_manager = EventStateManager(config)
        self.segmentation_model = YOLO(self.config.yolo_model_path)
        self.score_calculator = CombinedScoreCalculator(config, self.segmentation_model)
        self.ovality_calculator = OvalityCalculator()
        self.best_candidate_frame: Optional[CandidateFrame] = None
        
        self.logger.info("Pipeline initialized.")
    
    def _upload_to_s3(self, file_path: str, s3_key: str) -> str:
        """Upload file to S3 and return the S3 URL."""
        try:
            self.s3.upload_file(file_path, self.config.aws_bucket_name, s3_key)
            s3_url = f"s3://{self.config.aws_bucket_name}/{s3_key}"
            self.logger.info(f"Uploaded to S3: {s3_url}")
            return s3_url
        except Exception as e:
            self.logger.error(f"Failed to upload to S3: {e}")
            return ""
    
    async def push_data_to_mongo_go(self, response: dict):
        """Push data to MongoDB using GO backend."""
        try:
            # This would typically call a GO service endpoint
            # For now, we'll simulate the push
            self.logger.info(f"Pushing data to MongoDB: {response.get('timestamp', 'unknown')}")
            # In real implementation, this would be an HTTP call to GO service
            # await self._call_go_service(response)
        except Exception as e:
            self.logger.error(f"Failed to push to MongoDB: {e}")
    
    def _prepare_response_for_backend(self, response: dict) -> dict:
        """Prepare response for backend with proper structure."""
        backend_response = {
            "client_id": response["client_id"],
            "camera_id": "steel_coil_camera",
            "timestamp": response["timestamp"],
            "ovality": response["ovality"],
            "ovality_interpretation": response["ovality_interpretation"],
            "scores": response["scores"],
            "model_info": response["model_info"],
            "s3_urls": response["s3_urls"],
            "metadata": {
                "processing_time": time.time(),
                "client_meta_connected": self.client_meta_wrapper is not None
            }
        }
        return backend_response
    
    def process_frame(self, frame: np.ndarray) -> bool:
        try:
            if not self.pre_filter.should_process_frame(frame): 
                return False
            
            is_coil_present = self.coil_detector.confirm_coil_presence(frame)
            state_changed = self.event_manager.update_state(is_coil_present)
            
            if self.event_manager.is_in_event():
                self._process_candidate_frame(frame)
            
            if state_changed and not self.event_manager.is_in_event():
                self._process_best_frame()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return False
    
    def _process_candidate_frame(self, frame: np.ndarray):
        try:
            candidate = self.score_calculator.calculate(frame)
            if self.best_candidate_frame is None or candidate.combined_score > self.best_candidate_frame.combined_score:
                self.best_candidate_frame = candidate
        except Exception as e:
            self.logger.error(f"Error processing candidate frame: {e}")
    
    def _process_best_frame(self):
        if self.best_candidate_frame and self.best_candidate_frame.mask is not None:
            try:
                ovality = self.ovality_calculator.calculate(self.best_candidate_frame.mask)
                self._save_results(self.best_candidate_frame, ovality)
            except Exception as e:
                self.logger.error(f"Error calculating ovality: {e}")
        self.best_candidate_frame = None
    
    def _save_results(self, candidate: CandidateFrame, ovality: Optional[float]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.join(self.config.output_dir, f"steel_coil_{timestamp}")
        
        try:
            # Save images locally
            image_path = f"{base_filename}_image.jpg"
            mask_path = f"{base_filename}_mask.png"
            
            cv2.imwrite(image_path, candidate.frame)
            
            if candidate.mask is not None:
                mask_img = (candidate.mask * 255).astype(np.uint8)
                cv2.imwrite(mask_path, mask_img)

            # Upload to S3
            s3_image_key = f"{self.config.client_id}/steel_coil/{timestamp}_image.jpg"
            s3_mask_key = f"{self.config.client_id}/steel_coil/{timestamp}_mask.png"
            
            s3_image_url = self._upload_to_s3(image_path, s3_image_key)
            s3_mask_url = self._upload_to_s3(mask_path, s3_mask_key) if candidate.mask is not None else ""

            response = {
                "timestamp": datetime.now().isoformat(),
                "client_id": self.config.client_id,
                "ovality": ovality,
                "ovality_interpretation": self._interpret_ovality(ovality),
                "scores": {
                    "combined": candidate.combined_score,
                    "segmentation_confidence": candidate.segmentation_confidence,
                    "centering": candidate.centering_score,
                    "size": candidate.size_score,
                },
                "model_info": {
                    "model": self.config.yolo_model_path,
                    "type": "YOLOv11n-seg"
                },
                "s3_urls": {
                    "original_image": s3_image_url,
                    "mask_image": s3_mask_url
                }
            }
            
            # Simple validation
            expected_mapping = {
                'timestamp': str,
                'client_id': str,
                'ovality': Optional[float],
                'ovality_interpretation': str,
                'scores': dict,
                'model_info': dict,
                's3_urls': dict
            }
            
            if verify_response_mapping(response, expected_mapping):
                with open(f"{base_filename}_metadata.json", 'w') as f:
                    json.dump(response, f, indent=4)
                self.logger.info(f"Results saved: {base_filename}")
                
                # Push to MongoDB
                backend_response = self._prepare_response_for_backend(response)
                asyncio.run(self.push_data_to_mongo_go(backend_response))
            else:
                self.logger.error("Response validation failed.")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def _interpret_ovality(self, ovality: Optional[float]) -> str:
        if ovality is None: return "No data"
        elif ovality < 0.05: return "Excellent"
        elif ovality < 0.10: return "Good"
        elif ovality < 0.15: return "Fair"
        else: return "Poor"

    def run(self):
        self.running = True
        self.logger.info(f"Starting pipeline: {self.config.rtsp_url}")

        stream = VideoStream(self.config.rtsp_url)
        frame_count = 0
        
        while self.running:
            try:
                frame = stream.read()
                if frame is None:
                    time.sleep(1.0)
                    continue

                frame_count += 1
                self.process_frame(frame)
                
            except Exception as e:
                self.logger.error(f"Error processing frame {frame_count}: {e}")
                continue
        
        stream.stop()
        self.logger.info("Pipeline stopped.")
    
    def stop(self):
        self.running = False 