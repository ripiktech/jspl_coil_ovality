#!/usr/bin/env python3
"""
JSPL Steel Coil Ovality Detection

Usage:
    python main.py --client-id CLIENT_ID
    python main.py --client-id CLIENT_ID --rtsp-url "rtsp://your-stream-url"
    python main.py --client-id CLIENT_ID --debug
"""

import argparse
import logging
import sys
import signal
import os
from pathlib import Path
import schedule

sys.path.insert(0, str(Path(__file__).parent))

from jspl_coil_ovality.pipeline import DeploymentPipeline
from jspl_coil_ovality.config import DeploymentConfig
from ripikutils.logsman import setup_logger, LoggerWriter
from ripikvisionpy.commons.clientMeta.ClientMetaWrapperV3 import ClientMetaWrapperV3


def signal_handler(signum, frame):
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}. Shutting down...")
    if hasattr(signal_handler, 'pipeline'):
        signal_handler.pipeline.stop()
    sys.exit(0)


def validate_client_meta_connection(config: DeploymentConfig):
    """Validate clientMeta connection."""
    try:
        client_meta_wrapper = ClientMetaWrapperV3(version=config.client_meta_version)
        client_meta_stage_wrapper = ClientMetaWrapperV3(env=config.client_meta_stage_env, version=config.client_meta_version)
        
        if '-stage' in config.client_id:
            client_meta_wrapper = client_meta_stage_wrapper
            config.aws_bucket_name = 'rpk-clnt-in-dev'
        
        client_meta = client_meta_wrapper.fetch_client_info(config.client_id, config.use_case, True)
        
        if client_meta is None:
            raise Exception("Unable to fetch client meta.")
        
        return client_meta_wrapper, client_meta
        
    except Exception as e:
        raise Exception(f"Failed to connect to clientMeta: {e}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="JSPL Steel Coil Ovality Detection System")
    parser.add_argument('--rtsp-url', type=str, help='RTSP stream URL')
    parser.add_argument('--model', type=str, help='Path to YOLOv11n-seg model')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--client-id', type=str, required=True, help='Client ID for clientMeta connection')
    return parser.parse_args()


def create_config_from_args(args) -> DeploymentConfig:
    config = DeploymentConfig()
    if args.rtsp_url: config.rtsp_url = args.rtsp_url
    if args.model: config.yolo_model_path = args.model
    if args.debug: config.logging_level = "DEBUG"
    if args.client_id: config.client_id = args.client_id
    return config


def main():
    args = parse_arguments()
    schedule.every().day.at("03:00").do(DeploymentPipeline.delete_old_images)
    
    try:
        config = create_config_from_args(args)
        print(f"Client ID: {config.client_id}")
        print(f"RTSP URL: {config.rtsp_url}")
    except Exception as e:
        print(f"Failed to create configuration: {e}")
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
    print("JSPL Steel Coil Ovality")
    print("=" * 60)
    
    try:
        print("Validating clientMeta connection...")
        client_meta_wrapper, client_meta = validate_client_meta_connection(config)
        print("ClientMeta connection validated.")
    except Exception as e:
        print(f"ERROR: {e}")
        print("DS logic will not start until clientMeta connection is established.")
        sys.exit(1)
    
    if not os.path.exists(config.yolo_model_path):
        print(f"Model not found: {config.yolo_model_path}")
        sys.exit(1)
    
    try:
        pipeline = DeploymentPipeline(config, client_meta_wrapper, client_meta)
        print("Pipeline initialized successfully")
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal_handler.pipeline = pipeline
    
    try:
        print("Starting pipeline...")
        pipeline.run()
    except KeyboardInterrupt:
        print("Pipeline interrupted by user")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)
    finally:
        pipeline.stop()
        print("Pipeline stopped")


if __name__ == "__main__":
    main()
