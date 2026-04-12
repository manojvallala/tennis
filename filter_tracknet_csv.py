import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from near_player_pose_filter import CourtDetector, order_court_corners


DEFAULT_VIDEO_PATH = "Roger_Federer_v_Rafael_Nadal_Extended_Highlights_Australian_Open_2009_Final_1080P_fixed.mp4"
DEFAULT_CSV_PATH = "Roger_Federer_v_Rafael_Nadal_Extended_Highlights_Australian_Open_2009_Final_1080P_fixed_ball.csv"
DEFAULT_OUTPUT_CSV = "Roger_Federer_v_Rafael_Nadal_Extended_Highlights_Australian_Open_2009_Final_1080P_fixed_ball_filtered.csv"
DEFAULT_COURT_MODEL_PATH = "experiments/checkpoints/unet_mobilenet_v2_192.pth"
DEFAULT_COURT_ENCODER = "mobilenet_v2"
DEFAULT_COURT_IMAGE_SIZE = 192
DEFAULT_COURT_STRIDE = 15
DEFAULT_MAX_JUMP = 380.0
DEFAULT_POLYGON_MARGIN = 90.0
DEFAULT_MAX_INTERPOLATION_GAP = 6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter TrackNetV3 CSV predictions using court geometry and motion sanity checks."
    )
    parser.add_argument("--video-path", default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--court-model-path", default=DEFAULT_COURT_MODEL_PATH)
    parser.add_argument("--court-encoder-name", default=DEFAULT_COURT_ENCODER)
    parser.add_argument("--court-image-size", type=int, default=DEFAULT_COURT_IMAGE_SIZE)
    parser.add_argument("--court-stride", type=int, default=DEFAULT_COURT_STRIDE)
    parser.add_argument("--max-jump", type=float, default=DEFAULT_MAX_JUMP)
    parser.add_argument("--polygon-margin", type=float, default=DEFAULT_POLYGON_MARGIN)
    parser.add_argument("--max-interpolation-gap", type=int, default=DEFAULT_MAX_INTERPOLATION_GAP)
    return parser.parse_args()


def load_csv_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def write_csv_rows(output_path: Path, rows):
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Frame", "Visibility", "X", "Y"])
        writer.writeheader()
        writer.writerows(rows)


def build_playable_region(court_corners: np.ndarray) -> np.ndarray:
    ordered = np.asarray(court_corners, dtype=np.float32)
    ordered = order_court_corners(ordered)
    return ordered.astype(np.float32)


def expand_polygon(polygon: np.ndarray, margin: float) -> np.ndarray:
    center = np.mean(polygon, axis=0, keepdims=True)
    vectors = polygon - center
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1.0)
    expanded = polygon + (vectors / norms) * margin
    return expanded.astype(np.float32)


def point_is_plausible(point, polygon):
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


def interpolate_short_gaps(rows, max_gap):
    valid_indices = [idx for idx, row in enumerate(rows) if int(row["Visibility"]) == 1]
    if len(valid_indices) < 2:
        return rows

    for left_idx, right_idx in zip(valid_indices, valid_indices[1:]):
        gap = right_idx - left_idx - 1
        if gap <= 0 or gap > max_gap:
            continue

        left_x = int(rows[left_idx]["X"])
        left_y = int(rows[left_idx]["Y"])
        right_x = int(rows[right_idx]["X"])
        right_y = int(rows[right_idx]["Y"])

        for step, missing_idx in enumerate(range(left_idx + 1, right_idx), start=1):
            alpha = step / (gap + 1)
            interp_x = int(round((1.0 - alpha) * left_x + alpha * right_x))
            interp_y = int(round((1.0 - alpha) * left_y + alpha * right_y))
            rows[missing_idx]["Visibility"] = "1"
            rows[missing_idx]["X"] = str(interp_x)
            rows[missing_idx]["Y"] = str(interp_y)

    return rows


def main():
    args = parse_args()
    video_path = Path(args.video_path)
    csv_path = Path(args.csv_path)
    output_csv = Path(args.output_csv)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = load_csv_rows(csv_path)
    court_detector = CourtDetector(
        model_path=args.court_model_path,
        encoder_name=args.court_encoder_name,
        image_size=args.court_image_size,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    last_valid_point = None
    cached_polygon = None
    cached_frame = -1
    filtered_rows = []

    for row in rows:
        frame_idx = int(row["Frame"])
        vis = int(row["Visibility"])
        x_coord = int(row["X"])
        y_coord = int(row["Y"])

        if cached_polygon is None or frame_idx - cached_frame >= args.court_stride:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if ok:
                detected_corners, _ = court_detector.predict(frame)
                if detected_corners is not None:
                    cached_polygon = expand_polygon(
                        build_playable_region(detected_corners),
                        args.polygon_margin,
                    )
                    cached_frame = frame_idx

        keep = vis == 1
        current_point = (x_coord, y_coord)

        if keep and cached_polygon is not None:
            keep = point_is_plausible(current_point, cached_polygon)

        if keep and last_valid_point is not None:
            jump = float(np.linalg.norm(np.asarray(current_point) - np.asarray(last_valid_point)))
            if jump > args.max_jump:
                keep = False

        if keep:
            last_valid_point = current_point
            filtered_rows.append(
                {"Frame": str(frame_idx), "Visibility": "1", "X": str(x_coord), "Y": str(y_coord)}
            )
        else:
            filtered_rows.append(
                {"Frame": str(frame_idx), "Visibility": "0", "X": "0", "Y": "0"}
            )

    cap.release()
    filtered_rows = interpolate_short_gaps(filtered_rows, args.max_interpolation_gap)
    write_csv_rows(output_csv, filtered_rows)
    print(f"Filtered CSV written to: {output_csv}")


if __name__ == "__main__":
    main()
