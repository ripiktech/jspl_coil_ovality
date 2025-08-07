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
import nats
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
import os

from ultralytics import YOLO
from ripikutils.logsman import setup_logger, LoggerWriter
from ripikutils.stream import VideoStream
from ripikutils.response_handler import verify_response_mapping

from .config import DeploymentConfig, CandidateFrame
from .detection import PreFilter, CoilDetector, EventStateManager
from .scoring import CombinedScoreCalculator
from .ovality_calculator import OvalityCalculator

load_dotenv()

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
    
    def push_data_to_mongo(self, response: dict):
        """Push data directly to MongoDB using client meta configuration."""
        try:
            # Extract MongoDB info from client meta
            if self.client_meta and 'internal' in self.client_meta:
                internal = self.client_meta['internal']
                if 'mongodb' in internal:
                    mongodb_info = internal['mongodb']
                    
                    # Use actual MongoDB configuration
                    db_url = mongodb_info.get('dbUrl', "mongodb+srv://rpkuser:YIjmmpiXSs52XdaN@rpkjspl.0eedi.mongodb.net/")
                    db_name = mongodb_info.get('dbName', 'jspl-coil-ovality')
                    collection_name = mongodb_info.get('coll', {}).get('history', 'history')
                    
                    self.logger.info(f"Connecting to DB: {db_name}")
                    self.logger.info(f"DB URL: {db_url}")
                    
                    # Import pymongo here to avoid dependency issues
                    from pymongo import MongoClient
                    
                    # Connect to MongoDB
                    client = MongoClient(db_url)
                    db = client[db_name]
                    collection = db[collection_name]
                    
                    # Insert the response
                    result = collection.insert_one(response)
                    self.logger.info(f'Data pushed to MongoDB! Inserted ID: {result.inserted_id}')
                    
                    # Close connection
                    client.close()
                    
                else:
                    self.logger.error("MongoDB configuration not found in client meta")
            else:
                self.logger.error("Client meta not available for MongoDB push")
                
        except Exception as e:
            self.logger.error(f'Failed to push data to MongoDB: {e}')
            import traceback
            self.logger.error(f'Traceback: {traceback.format_exc()}')
            
    async def push_data_to_mongo_go(self, response: dict):
        """Push data to MongoDB using NATS."""
        try:
            NATS_CONN = await nats.connect("localhost:4222")
            js = NATS_CONN.jetstream()
            subject = "go_queue"
            local_save_root = "/home/ds/hsm/jspl-hsm_coil_id_mapping/images"
            bucket_name = "rpk-clnt-in-prd"
            
            outer_response = {
                "type": "s3-mongo",
                "bucketName": bucket_name,
                "images": {},
                "metadatas": []
            }
            
            # Process images in response
            for k in response:
                if k.endswith("Image"):
                    s3_path = response[k]
                    outer_response["images"][s3_path] = os.path.join(local_save_root, s3_path)
                    response[k] = f"https://{bucket_name}.s3.ap-south-1.amazonaws.com/{response[k]}"
            
            # Create metadata
            meta = {
                "database_name": "jspl-coil-ovality",
                "collection_name": "history",
                "metadata": response
            }
            outer_response["metadatas"].append(meta)
            
            # Publish message
            message = json.dumps(outer_response).encode("utf-8")
            ack = await js.publish(subject, message)
            
            self.logger.info(f"Published message with sequence: {ack.seq}")
            await NATS_CONN.close()
            
        except Exception as e:
            self.logger.error(f"Failed to push to MongoDB via NATS: {e}")
    
    def _upload_to_s3(self, file_path: str, s3_key: str, config) -> str:
        """Upload file to S3 using actual configuration."""
        try:
            # Create S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('access_key'),
                aws_secret_access_key=os.getenv('secret_key'),
                region_name=self.os.getenv('region')
            )
            
            s3_client.upload_file(file_path, os.getenv('bucket_name'), s3_key)
            s3_url = f"https://{os.getenv('bucket_name')}.s3.{os.getenv('region')}.amazonaws.com/{s3_key}"
            self.logger.info(f"Uploaded to S3: {s3_url}")
            return s3_url
            
        except Exception as e:
            self.logger.error(f"Failed to upload to S3: {e}")
            return ""
    
    def _prepare_response_for_backend(self, response: dict) -> dict:
        """Prepare response for backend with proper structure."""
        backend_response = {
            "cameraId": response["cameraId"],
            "clientId": response["clientId"],
            "cameraGpId": response["cameraGpId"],
            "plantId": response["plantId"],
            "usecase": response["usecase"],
            "entityId": response["entityId"],
            "shapeStatus": response["shapeStatus"],
            "isCoilPresent": response["isCoilPresent"],
            "shapeIndex": response["shapeIndex"],
            "isAlert": response["isAlert"],
            "originalImage": response["originalImage"],
            "annotatedImage": response["annotatedImage"],
            "createdAt": response["createdAt"],
            "timestamp": response["timestamp"]
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

            # Generate entityId synchronized with coil camera
            entity_id = self._get_synchronized_entity_id()
            
            # Check if we should process this entity
            if not self._should_process_ovality(entity_id):
                self.logger.info(f"Skipping ovality processing for entity {entity_id}")
                return
            
            # Determine shape status and index based on ovality
            shape_status, shape_index, is_alert = self._determine_shape_status(ovality)

            response = {
                "cameraId": "camovality",
                "clientId": self.config.client_id,
                "cameraGpId": "cameraGp1",
                "plantId": "angul",
                "usecase": "coilovality",
                "entityId": entity_id,
                "shapeStatus": shape_status,
                "isCoilPresent": True,
                "shapeIndex": shape_index,
                "isAlert": is_alert,
                "originalImage": s3_image_url,
                "annotatedImage": s3_mask_url,
                "createdAt": entity_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Push to MongoDB
            self.push_data_to_mongo(response)
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def _determine_shape_status(self, ovality: Optional[float]) -> tuple[str, int, bool]:
        """Determine shape status, index and alert based on ovality value."""
        if ovality is None or ovality < 0.15:
            return "Circular", 0, False
        elif ovality < 0.20:
            return "Ovality", 1, True
        else:
            return "Ovality", 2, True
    
    def _get_synchronized_entity_id(self) -> int:
        """Generate entityId synchronized with coil camera system."""
        return int(time.time() * 1000)
    
    def _should_process_ovality(self, entity_id: int) -> bool:
        """Check if we should process ovality for this entity."""
        return True

    def delete_old_images(directory="./images", days_threshold=2, image_extensions=None, dry_run=False):
        """
        Recursively deletes image files in the specified directory and its subdirectories
        that are older than the specified number of days.
        
        Args:
            directory (str): The directory to search through
            days_threshold (int): Number of days, files older than this will be deleted
            image_extensions (list, optional): List of image file extensions to check.
                                            Defaults to common image formats.
            dry_run (bool, optional): If True, only prints files that would be deleted
                                    without actually deleting them. Defaults to False.
        
        Returns:
            tuple: (count of deleted files, total size of deleted files in bytes)
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        
        # Convert days to seconds
        seconds_threshold = days_threshold * 24 * 60 * 60
        current_time = time.time()
        
        # Initialize counters
        deleted_count = 0
        deleted_size = 0
        
        # Check if the directory exists
        if not os.path.exists(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            return (0, 0)
        
        # Walk through the directory tree
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Check if the file has an image extension
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    file_path = os.path.join(root, file)
                    
                    # Get file modification time
                    file_mtime = os.path.getmtime(file_path)
                    
                    # Check if the file is older than the threshold
                    if current_time - file_mtime > seconds_threshold:
                        try:
                            # Get file size before deletion
                            file_size = os.path.getsize(file_path)
                            
                            # Format the file's modification time
                            mod_time = datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d %H:%M:%S')
                            days_old = (current_time - file_mtime) / (24 * 60 * 60)
                            
                            if dry_run:
                                print(f"Would delete: {file_path} (modified: {mod_time}, {days_old:.1f} days old)")
                            else:
                                os.remove(file_path)
                                print(f"Deleted: {file_path} (modified: {mod_time}, {days_old:.1f} days old)")
                            
                            deleted_count += 1
                            deleted_size += file_size
                        except Exception as e:
                            print(f"Error deleting {file_path}: {e}")
        
        # Format the total size in a human-readable format
        size_units = ['B', 'KB', 'MB', 'GB']
        size_index = 0
        size_display = deleted_size
        
        while size_display >= 1024 and size_index < len(size_units) - 1:
            size_display /= 1024
            size_index += 1
        
        # Print summary
        if dry_run:
            print(f"\nDry run summary: Would delete {deleted_count} files ({size_display:.2f} {size_units[size_index]})")
        else:
            print(f"\nDeleted {deleted_count} files ({size_display:.2f} {size_units[size_index]})")
        
        return (deleted_count, deleted_size)

    def run(self):
        self.running = True
        self.logger.info(f"Starting pipeline: {self.config.rtsp_url}")

        stream = VideoStream(self.config.rtsp_url)
        frame_count = 0
        
        while self.running:
            try:
                frame = stream.read()
                if frame is None:
                    no_frame_count += 1
                    if no_frame_count % 10 == 0:
                        self.logger.warning(f"No frame received for {no_frame_count} attempts")
                    time.sleep(1.0)
                    continue

                no_frame_count = 0  # Reset counter when frame is received
                frame_count += 1
                
                if frame_count % 30 == 0:  # Log every 30th frame
                    self.logger.info(f"Processing frame {frame_count}")
                
                self.process_frame(frame)
                
            except Exception as e:
                self.logger.error(f"Error processing frame {frame_count}: {e}")
                continue
        
        stream.stop()
        self.logger.info("Pipeline stopped.")
    
    def stop(self):
        self.running = False 