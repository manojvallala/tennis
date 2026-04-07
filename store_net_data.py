import argparse
import json
from pathlib import Path

import cv2
import numpy as np


POINT_KEYS = ["left_top", "left_bottom", "right_top", "right_bottom"]
MASK_ORDER = ["left_top", "right_top", "right_bottom", "left_bottom"]
VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a tennis net segmentation dataset from net annotation files.",
    )
    parser.add_argument("--annotations-dir", default=".")
    parser.add_argument("--output-dir", default="net_data")
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def discover_annotation_files(annotations_dir: Path) -> list[Path]:
    files = sorted(annotations_dir.glob("*_net_annotations.json"))
    if not files:
        raise FileNotFoundError(f"No *_net_annotations.json files found in {annotations_dir}")
    return files


def resolve_video_path(annotation_path: Path, payload: dict) -> Path:
    candidates: list[Path] = []
    for key in ("video_path", "video_name"):
        value = payload.get(key)
        if value:
            path = Path(value)
            candidates.append(path if path.is_absolute() else annotation_path.parent / path)

    stem = annotation_path.name.removesuffix("_net_annotations.json")
    for extension in VIDEO_EXTENSIONS:
        candidates.append(annotation_path.parent / f"{stem}{extension}")
        candidates.append(annotation_path.parent / f"{stem}{extension.upper()}")

    seen = set()
    for candidate in candidates:
        normalized = str(candidate.resolve(strict=False)).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find a matching video for {annotation_path.name}")


def canonicalize_points(points: dict) -> dict[str, list[int]] | None:
    raw_points = []
    for key in POINT_KEYS:
        value = points.get(key)
        if value is None or len(value) != 2:
            return None
        raw_points.append([int(value[0]), int(value[1])])

    sorted_by_x = sorted(raw_points, key=lambda point: (point[0], point[1]))
    left_pair = sorted(sorted_by_x[:2], key=lambda point: point[1])
    right_pair = sorted(sorted_by_x[2:], key=lambda point: point[1])
    ordered = [left_pair[0], left_pair[1], right_pair[0], right_pair[1]]
    return {key: point for key, point in zip(POINT_KEYS, ordered)}


def clear_output_dir(directory: Path):
    for path in directory.glob("*"):
        if path.is_file():
            path.unlink()


def ensure_output_dirs(output_dir: Path, clear_output: bool) -> tuple[Path, Path]:
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if clear_output:
        clear_output_dir(images_dir)
        clear_output_dir(masks_dir)
    return images_dir, masks_dir


def build_net_mask(frame_shape: tuple[int, int, int], points: dict[str, list[int]]) -> np.ndarray:
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    left_top = tuple(points["left_top"])
    left_bottom = tuple(points["left_bottom"])
    right_top = tuple(points["right_top"])
    right_bottom = tuple(points["right_bottom"])

    left_height = float(np.linalg.norm(np.asarray(left_bottom) - np.asarray(left_top)))
    right_height = float(np.linalg.norm(np.asarray(right_bottom) - np.asarray(right_top)))
    avg_height = max(2.0, 0.5 * (left_height + right_height))

    top_thickness = max(2, int(round(avg_height * 0.14)))
    bottom_thickness = max(2, int(round(avg_height * 0.10)))
    pole_thickness = max(2, int(round(avg_height * 0.12)))

    cv2.line(mask, left_top, right_top, 255, top_thickness, cv2.LINE_AA)
    cv2.line(mask, left_bottom, right_bottom, 255, bottom_thickness, cv2.LINE_AA)
    cv2.line(mask, left_top, left_bottom, 255, pole_thickness, cv2.LINE_AA)
    cv2.line(mask, right_top, right_bottom, 255, pole_thickness, cv2.LINE_AA)
    return mask


def store_annotation_samples(annotation_path: Path, images_dir: Path, masks_dir: Path, start_index: int) -> int:
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    annotations = payload.get("annotations", {})
    if not annotations:
        print(f"Skipping {annotation_path.name}: no annotations found")
        return 0

    video_path = resolve_video_path(annotation_path, payload)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    sample_count = 0
    for frame_idx_text, points in sorted(annotations.items(), key=lambda item: int(item[0])):
        normalized_points = canonicalize_points(points)
        if normalized_points is None:
            print(f"Skipping frame {frame_idx_text} from {annotation_path.name}: incomplete points")
            continue

        frame_idx = int(frame_idx_text)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Skipping frame {frame_idx} from {video_path.name}: unable to read frame")
            continue

        sample_index = start_index + sample_count
        image_path = images_dir / f"{sample_index}.png"
        mask_path = masks_dir / f"{sample_index}.png"
        net_mask = build_net_mask(frame.shape, normalized_points)

        cv2.imwrite(str(image_path), frame)
        cv2.imwrite(str(mask_path), net_mask)
        sample_count += 1

    cap.release()
    print(f"Processed {annotation_path.name} with {sample_count} samples")
    return sample_count


def main():
    args = parse_args()
    annotation_files = discover_annotation_files(Path(args.annotations_dir))
    images_dir, masks_dir = ensure_output_dirs(Path(args.output_dir), args.clear_output)

    total_samples = 0
    for annotation_path in annotation_files:
        try:
            written = store_annotation_samples(annotation_path, images_dir, masks_dir, total_samples)
        except FileNotFoundError as error:
            print(f"Skipping {annotation_path.name}: {error}")
            continue
        total_samples += written

    print(f"Stored {total_samples} net samples")
    print(f"Images saved to: {images_dir}")
    print(f"Masks saved to: {masks_dir}")


if __name__ == "__main__":
    main()
