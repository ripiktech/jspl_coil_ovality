# JSPL Steel Coil Ovality Detection System

A comprehensive real-time steel coil detection and ovality analysis system using fine-tuned YOLO models and multi-stage processing pipeline.

## Overview

This system provides automated detection and ovality analysis of steel coils from RTSP video streams. It uses a sophisticated multi-stage pipeline to:

1. **Pre-filter** frames for motion, brightness, and quality
2. **Detect** steel coils using fine-tuned YOLOv8n
3. **Select** the best frame during coil events
4. **Analyze** ovality using fine-tuned YOLOv11n-segmentation
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

1. **Fine-tuned Models**: Place your trained models in the `models/` directory:
   - `models/yolov8n-steel-coil.pt` - Detection model
   - `models/yolov11n-seg-steel-coil.pt` - Segmentation model

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

# Run with custom models
python main.py --detection-model "models/custom-detection.pt" --segmentation-model "models/custom-segmentation.pt"

# Run with debug logging
python main.py --log-level DEBUG
```

## Configuration

### Model Paths
- **Detection Model**: `models/yolov8n-steel-coil.pt`
- **Segmentation Model**: `models/yolov11n-seg-steel-coil.pt`

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
    "detection_model": "models/yolov8n-steel-coil.pt",
    "segmentation_model": "models/yolov11n-seg-steel-coil.pt"
  }
}
```

## Advanced Configuration

### Command Line Options
```bash
python main.py \
  --rtsp-url "rtsp://your-stream" \
  --detection-model "models/custom-detection.pt" \
  --segmentation-model "models/custom-segmentation.pt" \
  --detection-confidence 0.5 \
  --segmentation-confidence 0.7 \
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

## Development

### Adding New Features
1. **New Detection Classes**: Update `config.py` → `proxy_target_classes`
2. **Custom Scoring**: Modify `scoring.py` → `CombinedScoreCalculator`
3. **Additional Analysis**: Extend `ovality_calculator.py`

### Testing
```bash
# Test with sample images
python -c "from jspl_coil_ovality import OvalityCalculator; print('System ready')"
```

## Performance

### Optimizations
- **Frame Skipping**: Processes every 2nd frame
- **Early Rejection**: Fast pre-filtering
- **Memory Efficient**: Streaming processing
- **Error Recovery**: Automatic reconnection

### Expected Performance
- **Detection Rate**: 95%+ with fine-tuned models
- **Processing Speed**: 15-30 FPS (depending on hardware)
- **Accuracy**: High precision ovality measurement

---

**Note**: This system is specifically designed for steel coil detection and ovality analysis. Ensure your fine-tuned models are trained on steel coil datasets for optimal performance.

