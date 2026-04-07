import argparse
import json
import os
import random

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class NetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(
                f"Mask directory not found: {mask_dir}. Run 'python store_net_data.py --clear-output' first.",
            )

        image_names = {
            name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))
        }
        mask_names = {
            name for name in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, name))
        }
        self.images = sorted(image_names & mask_names)
        missing_masks = sorted(image_names - mask_names)
        if missing_masks:
            print(
                f"Skipping {len(missing_masks)} image files without matching masks in {mask_dir}.",
                flush=True,
            )

    def __len__(self):
        return len(self.images)

    def positive_fraction(self, indices=None):
        target_indices = indices if indices is not None else range(len(self.images))
        fractions = []
        for idx in target_indices:
            mask_path = os.path.join(self.mask_dir, self.images[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            fractions.append(float((mask > 127).mean()))
        if not fractions:
            return 0.05
        return float(sum(fractions) / len(fractions))

    def augment_sample(self, image, mask):
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        if random.random() < 0.7:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-18.0, 18.0)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        if random.random() < 0.2:
            image = cv2.GaussianBlur(image, (3, 3), 0)

        return image, mask

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask: {mask_path}")

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            image, mask = self.augment_sample(image, mask)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        edge_mask = cv2.morphologyEx((mask * 255).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)
        edge_mask = (edge_mask > 0).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)
        edge_mask = np.expand_dims(edge_mask, axis=0)
        return (
            torch.from_numpy(np.ascontiguousarray(image)),
            torch.from_numpy(np.ascontiguousarray(mask)),
            torch.from_numpy(np.ascontiguousarray(edge_mask)),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tennis net segmentation model.")
    parser.add_argument("--image-dir", default="net_data/images")
    parser.add_argument("--mask-dir", default="net_data/masks")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--encoder-name", default="resnet18")
    parser.add_argument("--output-model", default="tennis_net_model.pth")
    parser.add_argument("--output-metadata", default="tennis_net_model.json")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def make_dataloaders(dataset, batch_size, val_split, seed, num_workers):
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if len(indices) < 5:
        train_indices = indices
        val_indices = []
    else:
        val_count = max(1, int(round(len(indices) * val_split)))
        val_count = min(val_count, len(indices) - 1)
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]

    train_dataset = NetDataset(dataset.image_dir, dataset.mask_dir, dataset.image_size, augment=True)
    val_dataset = NetDataset(dataset.image_dir, dataset.mask_dir, dataset.image_size, augment=False)

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, train_indices, val_indices


def compute_iou_from_logits(logits, masks, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = ((preds + masks) > 0).float().sum(dim=(1, 2, 3)).clamp_min(1.0)
    return float((intersection / union).mean().item())


def weighted_bce_loss(logits, masks, edge_masks, pos_weight):
    pixel_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        masks,
        pos_weight=pos_weight,
        reduction="none",
    )
    weight_map = 1.0 + 3.0 * edge_masks + 0.5 * masks
    return (pixel_loss * weight_map).mean()


def evaluate_thresholds(logits_list, masks_list, thresholds):
    best_threshold = thresholds[0]
    best_iou = -1.0

    for threshold in thresholds:
        scores = [compute_iou_from_logits(logits, masks, threshold=threshold) for logits, masks in zip(logits_list, masks_list)]
        mean_iou = float(sum(scores) / max(len(scores), 1))
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_threshold = threshold

    return best_threshold, best_iou


def main():
    args = parse_args()

    dataset = NetDataset(args.image_dir, args.mask_dir, args.image_size)
    if len(dataset) == 0:
        raise RuntimeError(f"No training samples found in {args.image_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_loader, val_loader, train_indices, val_indices = make_dataloaders(
        dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = smp.Unet(
        encoder_name=args.encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    positive_fraction = dataset.positive_fraction(train_indices)
    pos_weight_value = max((1.0 - positive_fraction) / max(positive_fraction, 1e-4), 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_iou = -1.0
    best_threshold = 0.45

    for epoch in range(args.epochs):
        model.train()
        train_loss_total = 0.0

        for images, masks, edge_masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            edge_masks = edge_masks.to(device)

            preds = model(images)
            loss = 0.6 * weighted_bce_loss(preds, masks, edge_masks, pos_weight) + 0.4 * dice_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        train_loss = train_loss_total / max(len(train_loader), 1)

        val_loss = 0.0
        val_iou = 0.0
        if len(val_indices) > 0:
            model.eval()
            val_logits = []
            val_masks = []
            with torch.inference_mode():
                for images, masks, edge_masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    edge_masks = edge_masks.to(device)
                    preds = model(images)
                    loss = 0.6 * weighted_bce_loss(preds, masks, edge_masks, pos_weight) + 0.4 * dice_loss(preds, masks)
                    val_loss += loss.item()
                    val_logits.append(preds.detach().cpu())
                    val_masks.append(masks.detach().cpu())

            val_loss /= max(len(val_loader), 1)
            epoch_threshold, val_iou = evaluate_thresholds(
                val_logits,
                val_masks,
                thresholds=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
            )
        else:
            val_loss = train_loss
            val_iou = 0.0
            epoch_threshold = best_threshold

        if len(val_indices) == 0 or val_iou >= best_val_iou:
            torch.save(model.state_dict(), args.output_model)
            best_val_iou = val_iou
            best_threshold = epoch_threshold

        print(
            f"Epoch {epoch + 1}/{args.epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val IoU: {val_iou:.4f}, "
            f"Best Threshold: {epoch_threshold:.2f}",
        )

    with open(args.output_metadata, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "task": "tennis_net_segmentation",
                "encoder_name": args.encoder_name,
                "image_size": args.image_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "val_split": args.val_split,
                "seed": args.seed,
                "image_dir": args.image_dir,
                "mask_dir": args.mask_dir,
                "train_samples": len(train_indices),
                "val_samples": len(val_indices),
                "positive_fraction": positive_fraction,
                "pos_weight": pos_weight_value,
                "recommended_threshold": best_threshold,
                "best_val_iou": best_val_iou,
            },
            handle,
            indent=2,
        )
    print(f"Saved model to {args.output_model}")
    print(f"Saved metadata to {args.output_metadata}")


if __name__ == "__main__":
    main()
