# Deepfake Detector

A comprehensive deepfake detection system utilizing hybrid architectures (ViT-B/16 + 3D-ResNet18) and cross-attention fusion for spatial and temporal analysis.

## Project Structure

- `data/`: Ingestion, augmentation, and dataset utilities.
- `models/`: Hybrid model architecture and Grad-CAM visualization.
- `train/`: Training loops using PyTorch Lightning.
- `inference/`: Live webcam and batch video analysis.

## Features

- **Hybrid Detection**: Spatial analysis via ViT and temporal analysis via 3D-ResNet.
- **Explainability**: Integrated Grad-CAM for both spatial and temporal explanations.
- **Real-time Inference**: Live heatmap overlay during video stream.
- **Scalable Server**: FastAPI server with WebSocket support for browser integration.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

- **Training**: `python train/train.py`
- **Real-time Inference**: `python inference/realtime.py`
- **Video Analysis**: `python inference/video.py --path <video_path>`
