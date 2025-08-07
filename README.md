# JSPL Steel Coil Ovality Detection System

A comprehensive real-time steel coil detection and ovality analysis system using fine-tuned YOLO models and multi-stage processing pipeline.

## Overview

This system provides automated detection and ovality analysis of steel coils from RTSP video streams. It uses a sophisticated multi-stage pipeline to:

1. **Pre-filter** frames for motion, brightness, and quality
2. **Detect** steel coils using fine-tuned YOLOv11n-segmentation
3. **Select** the best frame during coil events
4. **Analyze** ovality using the same YOLOv11n-segmentation model
5. **Save** results with comprehensive metadata

## Architecture

The system is built using proper OOP principles with modular components:

```
jspl_coil_ovality/
├── main.py                           # Entry point with CLI
├── jspl_coil_ovality/
│   ├── __init__.py                  # Package initialization
│   ├── config.py                    # Configuration management
│   ├── detection.py                 # Detection and pre-filtering
│   ├── scoring.py                   # Frame scoring algorithms
│   ├── ovality_calculator.py        # Ovality calculation
│   └── pipeline.py                  # Main orchestration
├── models/                          # Fine-tuned models directory
├── data/deployment_output/          # Results directory
└── requirements.txt                 # Dependencies
```

## Quick Start

### Prerequisites

1. **Fine-tuned Model**: Place your trained model in the `models/` directory:
   - `models/yolov11n-seg-steel-coil.pt` - Unified detection and segmentation model

2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

```bash
# Run with default configuration
python main.py

# Run with custom RTSP URL
python main.py --rtsp-url "rtsp://admin:pass@192.168.1.100:554/stream"

# Run with custom model
python main.py --model "models/custom-yolov11n-seg.pt"

# Run with debug logging
python main.py --log-level DEBUG
```

## ClientMeta Validation

The system now includes mandatory clientMeta connection validation to ensure the DS logic only starts when the backend is up and running.

### ClientMeta Integration

The system validates clientMeta connection before starting any DS processing:

1. **Initial Validation**: Checks clientMeta connection during startup
2. **Periodic Validation**: Validates connection every 30 seconds during runtime
3. **Graceful Handling**: Logs errors but continues running if connection is lost

### Usage with ClientMeta

```bash
# Production environment
python main.py --client-id jspl_production

# Stage environment (uses stage clientMeta)
python main.py --client-id jspl_stage

# With custom RTSP URL
python main.py --client-id jspl_production --rtsp-url "rtsp://your-stream"

# Debug mode
python main.py --client-id jspl_production --debug
```

### ClientMeta Configuration

The system automatically handles different environments:

- **Production**: Uses `R_PROD` environment
- **Stage**: Uses `R_STAGE` environment (when client_id contains '-stage')
- **AWS Bucket**: Automatically switches to dev bucket for stage environment

### Error Handling

- **Connection Failed**: System exits with error message
- **Connection Lost**: System logs error but continues processing
- **Backend Down**: DS logic pauses until connection is restored

## Configuration

### Model Path
- **Unified Model**: `models/yolov11n-seg-steel-coil.pt` (handles both detection and segmentation)

### Confidence Thresholds
- **Detection**: 0.4 (higher for fine-tuned models)
- **Segmentation**: 0.6 (higher for fine-tuned models)

### Scoring Weights
- **Segmentation Confidence**: 50%
- **Centering Score**: 25%
- **Size Score**: 15%
- **Quality Score**: 10%

## Noise Classifier Integration

The system now includes an optional binary noise classifier to improve pre-filtering robustness:

### Configuration
```python
# Enable/disable noise filtering
config.enable_noise_filtering = True

# Path to your pretrained noise classifier
config.noise_classifier_path = "models/noise_classifier.pt"

# Confidence threshold for noise detection
config.noise_conf_threshold = 0.5
```

### Supported Model Formats
- **PyTorch**: `.pt` files (default)
- **ONNX**: `.onnx` files (requires `onnxruntime`)
- **TensorFlow**: `.pb` files (requires `tensorflow`)

### Integration Points
The noise classifier is integrated into the pre-filtering stage and will reject frames classified as noisy before they reach the expensive detection and segmentation stages.

### Testing
Use the provided test script to validate your noise classifier:
```bash
# Test with sample frame
python test_noise_classifier.py

# Test with video file
python test_noise_classifier.py --video path/to/test_video.mp4
```

## Output

The system saves comprehensive results for each detected steel coil:

### Files Generated
- `steel_coil_YYYYMMDD_HHMMSS_image.jpg` - Best frame image
- `steel_coil_YYYYMMDD_HHMMSS_mask.png` - Segmentation mask
- `steel_coil_YYYYMMDD_HHMMSS_metadata.json` - Analysis results

### Metadata Structure
```json
{
  "timestamp": "2025-01-30T17:17:07",
  "ovality": 0.15,
  "ovality_interpretation": "Slightly oval (good)",
  "scores": {
    "combined": 0.85,
    "segmentation_confidence": 0.92,
    "centering": 0.88,
    "size": 0.75,
    "quality": 0.82
  },
  "model_info": {
    "unified_model": "models/yolov11n-seg-steel-coil.pt",
    "model_type": "YOLOv11n-seg (detection + segmentation)"
  }
}
```

## Advanced Configuration

### Command Line Options
```bash
python main.py \
  --rtsp-url "rtsp://your-stream" \
  --model "models/custom-yolov11n-seg.pt" \
  --detection-confidence 0.4 \
  --segmentation-confidence 0.6 \
  --motion-threshold 1200 \
  --blur-threshold 150 \
  --log-level DEBUG
```

### Environment Variables
- `RTSP_URL`: Default RTSP stream URL
- `OUTPUT_DIR`: Default output directory
- `LOG_LEVEL`: Default logging level

## Ovality Analysis

The system provides detailed ovality analysis:

- **< 0.05**: Circular (excellent)
- **0.05 - 0.1**: Slightly oval (good)
- **0.1 - 0.2**: Moderately oval (acceptable)
- **> 0.2**: Highly oval (needs attention)

## Monitoring

### Log Files
- `steel_coil_deployment.log` - Main application log
- Console output with real-time status

### Key Log Messages
```
>>> STEEL COIL EVENT STARTED <<<
New best steel coil frame found with score: 0.85
<<< STEEL COIL EVENT ENDED <<<
Steel coil results saved to steel_coil_20250130_171707_...
```