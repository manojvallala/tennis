# Long Tennis Court Segmentation

This project is a long-tennis court segmentation and corner-detection pipeline adapted from a similar court-segmentation repository. The codebase, documentation, and default inference settings have been aligned so the repo consistently reflects the long-tennis use case.

## What The Project Does

- Annotates court corners from tennis match footage
- Converts annotations into segmentation masks
- Trains U-Net segmentation models
- Benchmarks multiple encoders and resolutions
- Runs video inference and overlays detected long-tennis court boundaries

## Project Layout

- [annotate_corners.py](./annotate_corners.py): manual annotation tool for tennis court corners
- [create_dataset.py](./create_dataset.py): converts annotated frames into image/mask pairs
- [train_seg_model.py](./train_seg_model.py): basic single-model training script
- [train_resolution_benchmarks.py](./train_resolution_benchmarks.py): benchmark runner across encoder and resolution combinations
- [test_seg_model.py](./test_seg_model.py): video inference and court-corner visualization
- [inference.py](./inference.py): quick evaluation script for checkpoint accuracy and inference timing
- [Court_Detection.md](./Court_Detection.md): baseline benchmark notes
- [Court_Detection_all_models.md](./Court_Detection_all_models.md): multi-model benchmark summary

## Data

The training data folder is expected at:

- `data/images`
- `data/masks`

Generate it from your long-tennis annotation JSON files with:

```bash
python create_dataset.py
```

The repository also includes annotation JSON files derived from tennis videos that you can use as inputs for dataset generation.

## Recommended Model

Based on the saved benchmark results, `U-Net + MobileNetV2 @ 192 x 192` is the best default tradeoff for this long-tennis project because it stays very close to the highest accuracy while running much faster on CPU.

Default inference now uses:

- checkpoint: `experiments/checkpoints/unet_mobilenet_v2_192.pth`
- encoder: `mobilenet_v2`
- image size: `192`

## Training

Run the benchmark suite:

```bash
python train_resolution_benchmarks.py
```

Run the simple training script:

```bash
python train_seg_model.py
```

## Inference

Run video testing with the long-tennis defaults:

```bash
python test_seg_model.py
```

Run timing and pixel-accuracy evaluation on the dataset:

```bash
python inference.py
```
