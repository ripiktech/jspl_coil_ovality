#!/usr/bin/env python3
"""
JSPL Steel Coil Ovality Detection - Main Entry Point
====================================================

This is the main entry point for the JSPL Steel Coil Ovality Detection system.
It provides a clean interface to run the deployment pipeline with proper
error handling and logging setup.

Usage:
    python main.py
    python main.py --config custom_config.yaml
    python main.py --rtsp-url "rtsp://your-stream-url"
"""

import argparse
import logging
import sys
import signal
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from jspl_coil_ovality import DeploymentPipeline, DeploymentConfig


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('steel_coil_deployment.log')
        ]
    )


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logging.info(f"Received signal {signum}. Shutting down gracefully...")
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
  python main.py --log-level DEBUG
  python main.py --detection-model "models/custom-yolov8n.pt"
  python main.py --segmentation-model "models/custom-yolov11n-seg.pt"
        """
    )
    
    parser.add_argument(
        '--rtsp-url',
        type=str,
        help='RTSP stream URL (default: from config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results (default: data/deployment_output)'
    )
    
    parser.add_argument(
        '--detection-model',
        type=str,
        help='Path to fine-tuned YOLOv8n detection model'
    )
    
    parser.add_argument(
        '--segmentation-model',
        type=str,
        help='Path to fine-tuned YOLOv11n-segmentation model'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--motion-threshold',
        type=float,
        help='Motion detection threshold (default: 1000.0)'
    )
    
    parser.add_argument(
        '--blur-threshold',
        type=float,
        help='Blur detection threshold (default: 100.0)'
    )
    
    parser.add_argument(
        '--detection-confidence',
        type=float,
        help='Detection confidence threshold (default: 0.4)'
    )
    
    parser.add_argument(
        '--segmentation-confidence',
        type=float,
        help='Segmentation confidence threshold (default: 0.6)'
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> DeploymentConfig:
    """Create configuration from command line arguments."""
    config = DeploymentConfig()
    
    if args.rtsp_url:
        config.rtsp_url = args.rtsp_url
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    if args.detection_model:
        config.proxy_model_path = args.detection_model
    
    if args.segmentation_model:
        config.segmentation_model_path = args.segmentation_model
    
    if args.motion_threshold:
        config.motion_threshold = args.motion_threshold
    
    if args.blur_threshold:
        config.blur_threshold = args.blur_threshold
    
    if args.detection_confidence:
        config.proxy_conf_threshold = args.detection_confidence
    
    if args.segmentation_confidence:
        config.seg_conf_threshold = args.segmentation_confidence
    
    return config


def main():
    """Main entry point for the steel coil deployment pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("JSPL Steel Coil Ovality Detection System")
    logger.info("=" * 60)
    
    # Create configuration
    try:
        config = create_config_from_args(args)
        logger.info(f"Configuration loaded successfully")
        logger.info(f"RTSP URL: {config.rtsp_url}")
        logger.info(f"Output Directory: {config.output_dir}")
        logger.info(f"Detection Model: {config.proxy_model_path}")
        logger.info(f"Segmentation Model: {config.segmentation_model_path}")
    except Exception as e:
        logger.error(f"Failed to create configuration: {e}")
        sys.exit(1)
    
    # Validate model paths
    if not validate_model_paths(config):
        sys.exit(1)
    
    # Create pipeline
    try:
        pipeline = DeploymentPipeline(config)
        logger.info("Steel coil pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal_handler.pipeline = pipeline
    
    # Run the pipeline
    try:
        logger.info("Starting steel coil deployment pipeline...")
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)
    finally:
        pipeline.stop()
        logger.info("Steel coil pipeline stopped")


if __name__ == "__main__":
    main()
