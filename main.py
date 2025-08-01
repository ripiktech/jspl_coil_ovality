#!/usr/bin/env python3
"""
JSPL Steel Coil Ovality Detection - Main Entry Point
====================================================

This is the main entry point for the JSPL Steel Coil Ovality Detection system.
It provides a clean interface to run the deployment pipeline with proper
error handling and logging setup.

Usage:
    python main.py
    python main.py --rtsp-url "rtsp://your-stream-url"
    python main.py --debug
"""

import argparse
import logging
import sys
import signal
import os
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jspl_coil_ovality.pipeline import DeploymentPipeline
from jspl_coil_ovality.config import DeploymentConfig
from ripikutils.logsman import setup_logger, LoggerWriter


def signal_handler(signum, frame):
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}. Shutting down gracefully...")
    if hasattr(signal_handler, 'pipeline'):
        signal_handler.pipeline.stop()
    sys.exit(0)


def validate_model_paths(config: DeploymentConfig) -> bool:
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(config.yolo_detection_model_path):
        logger.error(f"Detection model not found: {config.yolo_detection_model_path}")
        logger.info("Please ensure your fine-tuned YOLOv8n model is placed in the models/ directory")
        return False
    
    if not os.path.exists(config.yolo_segmentation_model_path):
        logger.error(f"Segmentation model not found: {config.yolo_segmentation_model_path}")
        logger.info("Please ensure your fine-tuned YOLOv11n-seg model is placed in the models/ directory")
        return False
    
    logger.info("All model files validated successfully")
    return True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="JSPL Steel Coil Ovality Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --rtsp-url "rtsp://admin:pass@192.168.1.100:554/stream"
  python main.py --debug
        """
    )
    
    parser.add_argument('--rtsp-url', type=str, help='RTSP stream URL (default: from config)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    return parser.parse_args()


def create_config_from_args(args) -> DeploymentConfig:
    config = DeploymentConfig()
    if args.rtsp_url: config.rtsp_url = args.rtsp_url
    if args.debug: config.logging_level = "DEBUG"
    return config


def main():
    args = parse_arguments()
    
    try:
        config = create_config_from_args(args)
        print(f"Configuration loaded successfully")
        print(f"RTSP URL: {config.rtsp_url}")
        print(f"Log Level: {config.logging_level}")
    except Exception as e:
        print(f"Failed to create configuration: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    
    logger = setup_logger(
        name=__name__,
        log_filename=config.log_filename,
        prefix=config.log_prefix,
        postfix=config.log_postfix,
        log_dir=config.log_dir,
        max_log_size=config.max_log_size,
        backup_count=config.backup_count,
        logging_level=getattr(logging, config.logging_level.upper())
    )
    
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)
    
    print("=" * 60)
    print("JSPL Steel Coil Ovality Detection System")
    print("=" * 60)
    
    if not validate_model_paths(config):
        sys.exit(1)
    
    try:
        pipeline = DeploymentPipeline(config)
        print("Steel coil pipeline initialized successfully")
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal_handler.pipeline = pipeline
    
    try:
        print("Starting steel coil deployment pipeline...")
        pipeline.run()
    except KeyboardInterrupt:
        print("Pipeline interrupted by user")
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        pipeline.stop()
        print("Steel coil pipeline stopped")


if __name__ == "__main__":
    main()
