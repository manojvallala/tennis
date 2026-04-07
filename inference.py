import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import time

image_dir = 'data/images'
mask_dir = 'data/masks'
model_path = 'experiments/checkpoints/unet_mobilenet_v2_192.pth'
size = (192, 192)
files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.Unet(encoder_name='mobilenet_v2', encoder_weights=None, in_channels=3, classes=1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
correct = 0
pixels = 0
elapsed_ms = []
with torch.inference_mode():
    for name in files:
        image = cv2.imread(os.path.join(image_dir, name))
        mask = cv2.imread(os.path.join(mask_dir, name), 0)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        tensor = image.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        tensor = torch.from_numpy(np.ascontiguousarray(tensor)).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred = model(tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)
        pred = (torch.sigmoid(pred).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        gt = (mask > 127).astype(np.uint8)
        correct += int((pred == gt).sum())
        pixels += pred.size
print(f'device={device}')
print(f'samples={len(files)}')
print(f'correct_pixels={correct}')
print(f'total_pixels={pixels}')
print(f'pixel_accuracy={correct / pixels:.12f}')
print(f'total_inference_ms={sum(elapsed_ms):.12f}')
print(f'avg_inference_ms={sum(elapsed_ms)/len(elapsed_ms):.12f}')
