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

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from jspl_coil_ovality import DeploymentPipeline, DeploymentConfig
from ripikutils.logsman import setup_logger, LoggerWriter


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}. Shutting down gracefully...")
    if hasattr(signal_handler, 'pipeline'):
        signal_handler.pipeline.stop()
    sys.exit(0)


def validate_model_paths(config: DeploymentConfig) -> bool:
    """Validate that the required model files exist."""
    logger = logging.getLogger(__name__)
    
    # Check detection model
    if not os.path.exists(config.proxy_model_path):
        logger.error(f"Detection model not found: {config.proxy_model_path}")
        logger.info("Please ensure your fine-tuned YOLOv8n model is placed in the models/ directory")
        return False
    
    # Check segmentation model
    if not os.path.exists(config.segmentation_model_path):
        logger.error(f"Segmentation model not found: {config.segmentation_model_path}")
        logger.info("Please ensure your fine-tuned YOLOv11n-seg model is placed in the models/ directory")
        return False
    
    logger.info("All model files validated successfully")
    return True


def parse_arguments():
    """Parse command line arguments."""
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
    
    parser.add_argument(
        '--rtsp-url',
        type=str,
        help='RTSP stream URL (default: from config)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> DeploymentConfig:
    """Create configuration from command line arguments."""
    config = DeploymentConfig()
    
    if args.rtsp_url:
        config.rtsp_url = args.rtsp_url
    
    if args.debug:
        config.logging_level = "DEBUG"
    
    return config


def main():
    """Main entry point for the steel coil deployment pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    try:
        config = create_config_from_args(args)
        print(f"Configuration loaded successfully")
        print(f"RTSP URL: {config.rtsp_url}")
        print(f"Log Level: {config.logging_level}")
    except Exception as e:
        print(f"Failed to create configuration: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    
    # Setup ripikutils logger
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
    
    # Configure stdout and stderr to be logged
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)
    
    print("=" * 60)
    print("JSPL Steel Coil Ovality Detection System")
    print("=" * 60)
    
    # Validate model paths
    if not validate_model_paths(config):
        sys.exit(1)
    
    # Create pipeline
    try:
        pipeline = DeploymentPipeline(config)
        print("Steel coil pipeline initialized successfully")
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal_handler.pipeline = pipeline
    
    # Run the pipeline
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
