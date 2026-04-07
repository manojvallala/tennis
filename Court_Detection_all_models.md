# Long Tennis Court Segmentation Benchmark

## Benchmark Setup

- Architecture family: `U-Net`
- Encoders tested: `ResNet-18`, `MobileNetV2`
- Input resolutions tested: `256 x 256`, `192 x 192`, `128 x 128`
- Dataset split used for all experiments: `150` training images and `37` validation images
- Random seed: `42`
- Training epochs: `8`
- Encoder initialization: `ImageNet`
- Loss function: `BCEWithLogitsLoss`
- Evaluation device for inference timing: `CPU`
- Pixel accuracy = correctly predicted mask pixels / total pixels in the validation set
- Average inference time = average model forward-pass time per validation frame, measured with batch size `1` after a warm-up pass

Saved experiment outputs:

- [summary.csv](experiments/results/summary.csv)
- [summary.json](experiments/results/summary.json)
- [model_comparison.csv](experiments/results/model_comparison.csv)
- [model_comparison.json](experiments/results/model_comparison.json)

## Model Comparison

| Model | Input Resolution | Little Explanation | Pixel Accuracy | Parameters | Weight Size (MB) | Avg Inference Time (ms/frame) | Best Epoch | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U-Net + ResNet-18 | `256 x 256` | Baseline U-Net with a ResNet-18 encoder. | `99.62%` | `14.33M` | `54.76` | `81.51` | `6` | [unet_resnet18_256.pth](experiments/checkpoints/unet_resnet18_256.pth) |
| U-Net + MobileNetV2 | `256 x 256` | MobileNet-based U-Net trained at the same resolution as the baseline. | `99.66%` | `6.63M` | `25.54` | `59.78` | `7` | [unet_mobilenet_v2_256.pth](experiments/checkpoints/unet_mobilenet_v2_256.pth) |
| U-Net + MobileNetV2 | `192 x 192` | Lower-resolution MobileNet experiment to reduce compute while preserving accuracy. | `99.64%` | `6.63M` | `25.54` | `35.31` | `8` | [unet_mobilenet_v2_192.pth](experiments/checkpoints/unet_mobilenet_v2_192.pth) |
| U-Net + MobileNetV2 | `128 x 128` | Smallest tested resolution for maximum speed reduction. | `99.60%` | `6.63M` | `25.54` | `29.01` | `7` | [unet_mobilenet_v2_128.pth](experiments/checkpoints/unet_mobilenet_v2_128.pth) |

## Key Findings

- `U-Net + MobileNetV2 @ 256` achieved the best pixel accuracy among the tested models: `99.66%`.
- Compared with `U-Net + ResNet-18 @ 256`, `U-Net + MobileNetV2 @ 256` used `53.74%` fewer parameters, produced a `53.36%` smaller checkpoint, and reduced average inference time by `26.66%`.
- `U-Net + MobileNetV2 @ 192` stayed within `0.01` percentage points of the `256 x 256` MobileNet result while reducing average inference time by `40.93%`.
- `U-Net + MobileNetV2 @ 128` was the fastest tested configuration at `29.01 ms/frame`, with a `0.06` percentage-point drop in pixel accuracy relative to `U-Net + MobileNetV2 @ 256`.
- Parameter count and checkpoint size remain the same across MobileNet resolutions because the network architecture is unchanged; only the input size is reduced.

## Practical Recommendation

- If maximum pixel accuracy is the priority, use `U-Net + MobileNetV2 @ 256`.
- If the goal is the best balance between speed and accuracy, use `U-Net + MobileNetV2 @ 192`.
- If minimum inference time is the priority, use `U-Net + MobileNetV2 @ 128`.

## Run Training

Use [`train_resolution_benchmarks.py`](train_resolution_benchmarks.py) to train and benchmark the models.

Train the full benchmark set:

```bash
python train_resolution_benchmarks.py
```

Train only the MobileNet resolution sweep used for this comparison:

```bash
python train_resolution_benchmarks.py --epochs 8 --experiments mobilenet_v2:256 mobilenet_v2:192 mobilenet_v2:128
```

Train the ResNet-18 baseline explicitly:

```bash
python train_resolution_benchmarks.py --epochs 8 --experiments resnet18:256
```

Train a single model at a specific resolution:

```bash
python train_resolution_benchmarks.py --epochs 8 --experiments mobilenet_v2:192
```

Useful options:

- `--epochs` to change the number of training epochs
- `--batch-size` to change the training batch size
- `--learning-rate` to change the optimizer learning rate
- `--val-ratio` to change the validation split
- `--output-dir` to save checkpoints and result files to a different folder

The training script saves checkpoints to [`experiments/checkpoints`](experiments/checkpoints) and benchmark summaries to [`experiments/results`](experiments/results).

## Run Video Testing

Use [`test_seg_model.py`](test_seg_model.py) to run the trained checkpoints on a video.

Example commands:

```bash
python test_seg_model.py --model-path experiments/checkpoints/unet_resnet18_256.pth --encoder-name resnet18 --image-size 256

python test_seg_model.py --model-path experiments/checkpoints/unet_mobilenet_v2_256.pth --encoder-name mobilenet_v2 --image-size 256

python test_seg_model.py --model-path experiments/checkpoints/unet_mobilenet_v2_192.pth --encoder-name mobilenet_v2 --image-size 192

python test_seg_model.py --model-path experiments/checkpoints/unet_mobilenet_v2_128.pth --encoder-name mobilenet_v2 --image-size 128
```
