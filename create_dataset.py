import argparse
import json
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a tennis court segmentation dataset from annotation JSON files.",
    )
    parser.add_argument(
        "--annotations-dir",
        default=".",
        help="Directory containing *_court_annotations.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for images/ and masks/.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete existing PNG files in the output images/ and masks/ folders before writing.",
    )
    return parser.parse_args()


def discover_annotation_files(annotations_dir: Path) -> list[Path]:
    files = sorted(annotations_dir.glob("*_court_annotations.json"))
    if not files:
        raise FileNotFoundError(
            f"No *_court_annotations.json files found in {annotations_dir}",
        )
    return files


def possible_video_paths(annotation_path: Path, payload: dict) -> list[Path]:
    candidates: list[Path] = []

    for key in ("video_path", "video_name"):
        value = payload.get(key)
        if not value:
            continue

        path = Path(value)
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(annotation_path.parent / path)
            candidates.append(Path.cwd() / path)

    stem = annotation_path.name.removesuffix("_court_annotations.json")
    for extension in VIDEO_EXTENSIONS:
        candidates.append(annotation_path.parent / f"{stem}{extension}")
        candidates.append(annotation_path.parent / f"{stem}{extension.upper()}")

    deduped: list[Path] = []
    seen = set()
    for candidate in candidates:
        normalized = str(candidate.resolve(strict=False)).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)

    return deduped


def resolve_video_path(annotation_path: Path, payload: dict) -> Path:
    for candidate in possible_video_paths(annotation_path, payload):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find a matching video for annotation file: {annotation_path.name}",
    )


def clear_existing_pngs(directory: Path) -> None:
    for png_file in directory.glob("*.png"):
        png_file.unlink()


def create_output_dirs(output_dir: Path, clear_output: bool) -> tuple[Path, Path]:
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    if clear_output:
        clear_existing_pngs(images_dir)
        clear_existing_pngs(masks_dir)

    return images_dir, masks_dir


def write_samples_for_annotation(
    annotation_path: Path,
    images_dir: Path,
    masks_dir: Path,
    start_index: int,
) -> int:
    with annotation_path.open(encoding="utf-8") as handle:
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
        frame_idx = int(frame_idx_text)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Skipping frame {frame_idx} from {video_path.name}: unable to read frame")
            continue

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        file_index = start_index + sample_count
        image_path = images_dir / f"{file_index}.png"
        mask_path = masks_dir / f"{file_index}.png"

        cv2.imwrite(str(image_path), frame)
        cv2.imwrite(str(mask_path), mask)
        sample_count += 1

    cap.release()
    print(
        f"Processed {annotation_path.name} with {sample_count} samples from {video_path.name}",
    )
    return sample_count


def main():
    args = parse_args()
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)

    annotation_files = discover_annotation_files(annotations_dir)
    images_dir, masks_dir = create_output_dirs(output_dir, args.clear_output)

    total_samples = 0
    skipped_annotations: list[str] = []
    for annotation_path in annotation_files:
        try:
            written = write_samples_for_annotation(
                annotation_path=annotation_path,
                images_dir=images_dir,
                masks_dir=masks_dir,
                start_index=total_samples,
            )
        except FileNotFoundError as error:
            print(f"Skipping {annotation_path.name}: {error}")
            skipped_annotations.append(annotation_path.name)
            continue
        total_samples += written

    print(f"Dataset created with {total_samples} samples")
    print(f"Images saved to: {images_dir}")
    print(f"Masks saved to: {masks_dir}")
    if skipped_annotations:
        print("Skipped annotation files without a matching video:")
        for name in skipped_annotations:
            print(f"- {name}")


if __name__ == "__main__":
    main()
