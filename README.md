# Tennis Court and Net Segmentation

This project provides a complete pipeline for tennis court and net segmentation, including corner detection and pole annotation. It includes tools for manual annotation, dataset creation, model training, benchmarking, and video inference for both court boundaries and net poles.

## What The Project Does

- Annotates tennis court corners from match footage
- Annotates tennis net pole positions (left-top, left-bottom, right-top, right-bottom)
- Converts annotations into segmentation masks for both court and net
- Trains U-Net segmentation models for court and net detection
- Benchmarks multiple encoders and resolutions for court segmentation
- Runs video inference and overlays detected court boundaries and net poles

## Project Layout

### Court Segmentation
- [annotate_corners.py](./annotate_corners.py): Manual annotation tool for tennis court corners
- [create_dataset.py](./create_dataset.py): Converts annotated court frames into image/mask pairs
- [train_seg_model.py](./train_seg_model.py): Basic single-model training script for court segmentation
- [train_resolution_benchmarks.py](./train_resolution_benchmarks.py): Benchmark runner across encoder and resolution combinations for court models
- [test_seg_model.py](./test_seg_model.py): Video inference and court-corner visualization
- [inference.py](./inference.py): Quick evaluation script for court checkpoint accuracy and inference timing

### Net Segmentation
- [annotation_net.py](./annotation_net.py): Manual annotation tool for tennis net pole positions
- [store_net_data.py](./store_net_data.py): Converts net annotations into image/mask pairs for training
- [train_net_model.py](./train_net_model.py): Training script for net segmentation model
- [test_net_model.py](./test_net_model.py): Video inference for net pole detection and visualization

### Documentation
- [Court_Detection.md](./Court_Detection.md): Baseline benchmark notes for court detection
- [Court_Detection_all_models.md](./Court_Detection_all_models.md): Multi-model benchmark summary for court detection

## Data

### Court Data
The court training data folder is expected at:
- `data/images`
- `data/masks`

Generate it from your tennis court annotation JSON files with:
```bash
python create_dataset.py
```

### Net Data
The net training data folder is expected at:
- `net_data/images`
- `net_data/masks`
- `net_data/labels` (JSON files with pole coordinates)

Generate it from your tennis net annotation JSON files with:
```bash
python store_net_data.py
```

The repository includes annotation JSON files derived from tennis videos that you can use as inputs for dataset generation.

## Recommended Models

### Court Segmentation
Based on the saved benchmark results, `U-Net + MobileNetV2 @ 192 x 192` is the best default tradeoff for tennis court segmentation because it stays very close to the highest accuracy while running much faster on CPU.

Default court inference uses:
- checkpoint: `experiments/checkpoints/unet_mobilenet_v2_192.pth`
- encoder: `mobilenet_v2`
- image size: `192`

### Net Segmentation
Default net inference uses:
- checkpoint: `tennis_net_model.pth`
- metadata: `tennis_net_model.json`
- encoder: `resnet18`
- image size: `256`

## Training

### Court Training
Run the benchmark suite for court models:
```bash
python train_resolution_benchmarks.py
```

Run the simple training script for court segmentation:
```bash
python train_seg_model.py
```

### Net Training
Train the net segmentation model:
```bash
python train_net_model.py
```

## Inference

### Court Inference
Run video testing with the court defaults:
```bash
python test_seg_model.py
```

Run timing and pixel-accuracy evaluation on the court dataset:
```bash
python inference.py
```

### Net Inference
Run video testing for net pole detection:
```bash
python test_net_model.py
```

## Near-Camera Player Pose Filtering

If your pose model is detecting the opponent, staff, or audience, first run person
detection and then keep only the near-camera player before pose estimation.

Use the helper in [near_player_pose_filter.py](./near_player_pose_filter.py):

```python
from near_player_pose_filter import (
    build_pose_target_from_yolo,
    crop_pose_input,
    select_near_camera_player,
)

person_result = detector(frame)[0]
detections = build_pose_target_from_yolo(person_result)

target_player = select_near_camera_player(
    detections,
    frame.shape,
    court_corners=detected_court_corners,  # order: TL, TR, BR, BL
)

if target_player is not None:
    player_crop, crop_box = crop_pose_input(frame, target_player)
    pose_result = pose_model(player_crop)
```

This filter prefers:
- people whose feet land inside the near half of the detected court
- larger player boxes, which usually means closer to the camera
- lower foot position in the frame, which also favors the near-side player
