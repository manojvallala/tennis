# Long Tennis Court Detection Benchmark

## Baseline Model

This document summarizes the baseline long-tennis court segmentation model used in the repository.

| Checkpoint | Description | Pixel Accuracy | Parameters | Weight Size (MB) | Avg Inference Time (ms/frame) |
| --- | --- | --- | --- | --- | --- |
| `experiments/checkpoints/unet_resnet18_256.pth` | U-Net with a ResNet-18 encoder used to segment the tennis court area from each frame. | `99.62%` | `14.33M` | `54.76` | `81.51` |

## How Metrics Are Calculated

### Pixel Accuracy

Pixel accuracy is computed by comparing the predicted binary mask with the ground-truth mask for every pixel in the validation set.

`Pixel Accuracy = Correctly Predicted Pixels / Total Pixels`

Baseline numbers:

- Correctly predicted pixels: `2,415,522`
- Total pixels: `2,424,832`
- Final value: `99.62%`

### Parameters

Parameters are the total number of learnable values in the model.

- Total parameters: `14,328,209`
- Final value: `14.33M`

### Weight Size

Weight size is the saved checkpoint size on disk.

- Checkpoint file size: `57,417,131 bytes`
- Final value: `54.76 MB`

### Average Inference Time

Average inference time measures the model forward pass on the validation split with batch size `1` after a warm-up pass.

- Timed validation frames: `37`
- Total forward-pass time: `3015.88 ms`
- Final value: `81.51 ms/frame`
- Device: `CPU`

## Inference Pipeline

The long-tennis inference pipeline in [test_seg_model.py](./test_seg_model.py) does the following:

1. Resizes each video frame to the chosen input size.
2. Runs the segmentation model to predict the tennis court mask.
3. Thresholds the output into a binary mask.
4. Resizes the mask back to the original frame dimensions.
5. Cleans the mask with morphological operations.
6. Finds the largest valid court contour.
7. Estimates and refines the four court corners.
8. Draws the segmented area and corner labels on the frame.

## Runtime Notes

The display loop is optimized for smoother playback:

- `SOURCE_FRAME_STRIDE = 2` skips every other source frame
- `INFERENCE_FRAME_STRIDE = 4` reuses the last prediction between inference steps

That means the live demo output is optimized for viewing and is not exactly the same as the isolated per-frame benchmark timing.

## Suggested Next Step

For a better speed/accuracy balance in this long-tennis project, use the MobileNetV2 checkpoints described in [Court_Detection_all_models.md](./Court_Detection_all_models.md).
