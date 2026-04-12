import argparse
import csv
from pathlib import Path

import cv2


DEFAULT_VIDEO_PATH = "Roger_Federer_v_Rafael_Nadal_Extended_Highlights_Australian_Open_2009_Final_1080P_fixed.mp4"
DEFAULT_CSV_PATH = "Roger_Federer_v_Rafael_Nadal_Extended_Highlights_Australian_Open_2009_Final_1080P_fixed_ball_filtered.csv"
DEFAULT_OUTPUT_PATH = "Roger_Federer_v_Rafael_Nadal_Extended_Highlights_Australian_Open_2009_Final_1080P_first30_ball.mp4"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay ball CSV coordinates onto a video and export a clip."
    )
    parser.add_argument("--video-path", default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=30.0)
    return parser.parse_args()


def load_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["Frame"]): row for row in csv.DictReader(handle)}


def main():
    args = parse_args()
    video_path = Path(args.video_path)
    csv_path = Path(args.csv_path)
    output_path = Path(args.output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, int(round(args.start_sec * fps)))
    end_frame = int(round(args.end_sec * fps))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create output video: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        row = rows.get(frame_idx)
        if row is not None and int(row["Visibility"]) == 1:
            x_coord = int(row["X"])
            y_coord = int(row["Y"])
            cv2.circle(frame, (x_coord, y_coord), 10, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (x_coord, y_coord), 4, (0, 0, 255), -1, cv2.LINE_AA)

        cv2.putText(
            frame,
            f"frame={frame_idx}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved overlay video to: {output_path}")


if __name__ == "__main__":
    main()
