from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset


class CourtDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, files: list[str], image_size: int):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.files = files
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.files[index]
        image_path = self.image_dir / name
        mask_path = self.mask_dir / name

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask: {mask_path}")

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.ascontiguousarray(image)

        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)
        mask = np.ascontiguousarray(mask)

        return torch.from_numpy(image), torch.from_numpy(mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and benchmark segmentation models across different resolutions.",
    )
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--mask-dir", default="data/masks")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "resnet18:256",
            "mobilenet_v2:256",
            "mobilenet_v2:192",
            "mobilenet_v2:128",
        ],
        help="List of encoder:resolution pairs to train and benchmark.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--output-dir", default="experiments")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_files(image_dir: Path, val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    files = sorted(path.name for path in image_dir.iterdir() if path.is_file())
    if not files:
        raise RuntimeError(f"No images found in {image_dir}")

    rng = random.Random(seed)
    rng.shuffle(files)

    val_count = max(1, int(round(len(files) * val_ratio)))
    val_files = sorted(files[:val_count])
    train_files = sorted(files[val_count:])

    if not train_files:
        raise RuntimeError("Validation split consumed the full dataset.")

    return train_files, val_files


def build_model(encoder_name: str, encoder_weights: str | None) -> torch.nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )


def describe_encoder(encoder_name: str) -> str:
    names = {
        "resnet18": "U-Net with ResNet-18 encoder.",
        "mobilenet_v2": "U-Net with MobileNetV2 encoder.",
    }
    return names.get(encoder_name, f"U-Net with {encoder_name} encoder.")


def pixel_stats_from_logits(logits: torch.Tensor, masks: torch.Tensor) -> tuple[int, int]:
    preds = (torch.sigmoid(logits) > 0.5).to(torch.uint8)
    targets = (masks > 0.5).to(torch.uint8)
    correct = int((preds == targets).sum().item())
    total = targets.numel()
    return correct, total


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = loss_fn(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct, pixels = pixel_stats_from_logits(logits, masks)
        total_correct += correct
        total_pixels += pixels

    return total_loss / len(loader.dataset), total_correct / total_pixels


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, int, int]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = loss_fn(logits, masks)

        total_loss += loss.item() * images.size(0)
        correct, pixels = pixel_stats_from_logits(logits, masks)
        total_correct += correct
        total_pixels += pixels

    return total_loss / len(loader.dataset), total_correct, total_pixels


@torch.inference_mode()
def benchmark_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, int]:
    model.eval()
    elapsed_ms: list[float] = []
    warmup_done = False

    for images, _ in loader:
        images = images.to(device)

        if not warmup_done:
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            warmup_done = True

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    total_ms = float(sum(elapsed_ms))
    avg_ms = total_ms / len(elapsed_ms)
    return total_ms, avg_ms, len(elapsed_ms)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "run_name",
        "encoder_name",
        "image_size",
        "description",
        "checkpoint_path",
        "best_epoch",
        "train_samples",
        "val_samples",
        "pixel_accuracy",
        "correct_pixels",
        "total_pixels",
        "parameters",
        "checkpoint_size_mb",
        "avg_inference_ms",
        "total_inference_ms",
        "timed_frames",
        "device",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def train_and_benchmark(
    *,
    encoder_name: str,
    image_size: int,
    train_files: list[str],
    val_files: list[str],
    args: argparse.Namespace,
    device: torch.device,
    checkpoints_dir: Path,
    results_dir: Path,
) -> dict:
    run_name = f"unet_{encoder_name}_{image_size}"
    checkpoint_path = checkpoints_dir / f"{run_name}.pth"
    history_path = results_dir / f"{run_name}.json"

    train_dataset = CourtDataset(
        image_dir=Path(args.image_dir),
        mask_dir=Path(args.mask_dir),
        files=train_files,
        image_size=image_size,
    )
    val_dataset = CourtDataset(
        image_dir=Path(args.image_dir),
        mask_dir=Path(args.mask_dir),
        files=val_files,
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    timing_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(encoder_name, args.encoder_weights)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_state_dict = None
    best_epoch = 0
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        val_loss, correct_pixels, total_pixels = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )
        val_accuracy = correct_pixels / total_pixels

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_pixel_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_pixel_accuracy": val_accuracy,
                "val_correct_pixels": correct_pixels,
                "val_total_pixels": total_pixels,
            }
        )

        print(
            f"[{run_name}] epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f}",
            flush=True,
        )

        if val_accuracy > best_val_accuracy or (
            np.isclose(val_accuracy, best_val_accuracy) and val_loss < best_val_loss
        ):
            best_epoch = epoch
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state_dict is None:
        raise RuntimeError(f"No checkpoint captured for {run_name}")

    torch.save(best_state_dict, checkpoint_path)
    model.load_state_dict(best_state_dict)

    final_val_loss, correct_pixels, total_pixels = evaluate(
        model=model,
        loader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )
    total_inference_ms, avg_inference_ms, timed_frames = benchmark_inference(
        model=model,
        loader=timing_loader,
        device=device,
    )

    parameters = sum(param.numel() for param in model.parameters())
    checkpoint_size_bytes = checkpoint_path.stat().st_size
    checkpoint_size_mb = checkpoint_size_bytes / (1024 * 1024)

    result = {
        "run_name": run_name,
        "encoder_name": encoder_name,
        "image_size": image_size,
        "description": describe_encoder(encoder_name),
        "checkpoint_path": str(checkpoint_path.as_posix()),
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "train_samples": len(train_files),
        "val_samples": len(val_files),
        "pixel_accuracy": correct_pixels / total_pixels,
        "correct_pixels": correct_pixels,
        "total_pixels": total_pixels,
        "parameters": parameters,
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "checkpoint_size_mb": checkpoint_size_mb,
        "total_inference_ms": total_inference_ms,
        "avg_inference_ms": avg_inference_ms,
        "timed_frames": timed_frames,
        "device": str(device),
        "best_val_loss": final_val_loss,
        "history": history,
    }

    save_json(history_path, result)
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_files, val_files = split_files(
        image_dir=image_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    summary: list[dict] = []
    for item in args.experiments:
        encoder_name, image_size_text = item.split(":", maxsplit=1)
        image_size = int(image_size_text)
        result = train_and_benchmark(
            encoder_name=encoder_name,
            image_size=image_size,
            train_files=train_files,
            val_files=val_files,
            args=args,
            device=device,
            checkpoints_dir=checkpoints_dir,
            results_dir=results_dir,
        )
        summary.append(result)

    summary_payload = {
        "device": str(device),
        "seed": args.seed,
        "train_samples": len(train_files),
        "val_samples": len(val_files),
        "experiments": summary,
    }
    save_json(results_dir / "summary.json", summary_payload)
    write_summary_csv(results_dir / "summary.csv", summary)

    print(f"Saved benchmark summary to {(results_dir / 'summary.json').as_posix()}", flush=True)
    print(f"Saved benchmark table to {(results_dir / 'summary.csv').as_posix()}", flush=True)


if __name__ == "__main__":
    main()
