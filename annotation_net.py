#!/usr/bin/env python3
"""
Tennis Net Pole Annotation Tool

Click on the 4 net-pole points in order:
    1. Left pole top
    2. Left pole bottom
    3. Right pole top
    4. Right pole bottom

Controls:
    LEFT CLICK      - Place/move point
    RIGHT CLICK     - Remove last point
    ENTER/SPACE     - Save annotation and go to next frame
    BACKSPACE       - Go to previous frame
    S               - Skip frame
    R               - Reset points for current frame
    Q/ESC           - Save and quit
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2


POINT_NAMES = ["Left-Top", "Left-Bottom", "Right-Top", "Right-Bottom"]
POINT_KEYS = ["left_top", "left_bottom", "right_top", "right_bottom"]
POINT_COLORS = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (0, 255, 0),
]


def canonicalize_points(points: list[list[int]]) -> list[list[int]]:
    if len(points) != 4:
        return points

    sorted_by_x = sorted(points, key=lambda point: (point[0], point[1]))
    left_pair = sorted(sorted_by_x[:2], key=lambda point: point[1])
    right_pair = sorted(sorted_by_x[2:], key=lambda point: point[1])
    ordered = [left_pair[0], left_pair[1], right_pair[0], right_pair[1]]
    return [[int(x_coord), int(y_coord)] for x_coord, y_coord in ordered]


class NetPoleAnnotator:
    def __init__(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        skip_frames: int = 30,
        resume_path: Optional[str] = None,
    ):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.skip_frames = skip_frames
        self.output_path = (
            Path(output_path)
            if output_path
            else self.video_path.parent / f"{self.video_path.stem}_net_annotations.json"
        )

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.annotations: dict[int, dict[str, list[int]]] = {}
        self.current_points: list[list[int]] = []

        if resume_path and Path(resume_path).exists():
            self.load_annotations(resume_path)
            self.output_path = Path(resume_path)

        self.frame_indices = list(range(0, self.total_frames, self.skip_frames))
        self.current_list_idx = 0
        if self.annotations:
            for index, frame_idx in enumerate(self.frame_indices):
                if frame_idx not in self.annotations:
                    self.current_list_idx = index
                    break

        self.current_frame_idx = self.frame_indices[self.current_list_idx]
        self.current_frame = None

    def load_annotations(self, path: str):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.annotations = {
            int(frame_idx): value for frame_idx, value in payload.get("annotations", {}).items()
        }
        print(f"Loaded {len(self.annotations)} existing annotations.")

    def save_annotations(self):
        payload = {
            "video_path": str(self.video_path),
            "video_name": self.video_path.name,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "skip_frames": self.skip_frames,
            "point_order": POINT_KEYS,
            "annotations": self.annotations,
        }
        with open(self.output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved {len(self.annotations)} annotations to {self.output_path}")

    def load_frame(self, frame_idx: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.current_points) < 4:
                self.current_points.append([x, y])
            else:
                best_index = 0
                best_distance = float("inf")
                for index, (px, py) in enumerate(self.current_points):
                    distance = (x - px) ** 2 + (y - py) ** 2
                    if distance < best_distance:
                        best_distance = distance
                        best_index = index
                self.current_points[best_index] = [x, y]
        elif event == cv2.EVENT_RBUTTONDOWN and self.current_points:
            self.current_points.pop()

    def current_annotation_as_dict(self) -> dict[str, list[int]]:
        ordered_points = canonicalize_points(self.current_points)
        return {
            key: [int(point[0]), int(point[1])]
            for key, point in zip(POINT_KEYS, ordered_points)
        }

    def draw_overlay(self, frame):
        display = frame.copy()
        height, width = display.shape[:2]

        for index, (x, y) in enumerate(self.current_points):
            color = POINT_COLORS[index]
            cv2.circle(display, (x, y), 8, color, -1)
            cv2.circle(display, (x, y), 10, (255, 255, 255), 2)
            cv2.putText(
                display,
                POINT_NAMES[index],
                (x + 12, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        if len(self.current_points) >= 2:
            cv2.line(display, tuple(self.current_points[0]), tuple(self.current_points[1]), (255, 255, 255), 2)
        if len(self.current_points) >= 4:
            cv2.line(display, tuple(self.current_points[2]), tuple(self.current_points[3]), (255, 255, 255), 2)

        cv2.rectangle(display, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.putText(display, f"Frame: {self.current_frame_idx}/{self.total_frames - 1}", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Progress: {len(self.annotations)}/{len(self.frame_indices)} annotated", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if len(self.current_points) < 4:
            cv2.putText(display, f"Click: {POINT_NAMES[len(self.current_points)]}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, POINT_COLORS[len(self.current_points)], 2)
        else:
            cv2.putText(display, "All 4 points placed. ENTER to save.", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.rectangle(display, (0, height - 30), (width, height), (0, 0, 0), -1)
        cv2.putText(display, "[Click] Place | [Right-Click] Undo | [Enter] Save | [S] Skip | [R] Reset | [Q] Quit", (20, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        return display

    def run(self):
        print("=" * 60)
        print("TENNIS NET POLE ANNOTATION TOOL")
        print("=" * 60)
        print(f"Video: {self.video_path.name}")
        print(f"Frames to annotate: {len(self.frame_indices)} (every {self.skip_frames} frames)")
        print(f"Output: {self.output_path}")
        print("-" * 60)

        window_name = "Tennis Net Pole Annotator"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while True:
            self.current_frame = self.load_frame(self.current_frame_idx)
            if self.current_frame is None:
                print(f"Error loading frame {self.current_frame_idx}")
                break

            if self.current_frame_idx in self.annotations and not self.current_points:
                existing = self.annotations[self.current_frame_idx]
                self.current_points = canonicalize_points(
                    [list(existing[key]) for key in POINT_KEYS if key in existing],
                )

            cv2.imshow(window_name, self.draw_overlay(self.current_frame))
            key = cv2.waitKeyEx(30)
            if key == -1:
                continue

            key_lower = key & 0xFF
            if key_lower in (13, 32):
                if len(self.current_points) == 4:
                    self.current_points = canonicalize_points(self.current_points)
                    self.annotations[self.current_frame_idx] = self.current_annotation_as_dict()
                    print(f"[SAVED] Frame {self.current_frame_idx}")
                    self.current_points = []
                    self.current_list_idx = min(self.current_list_idx + 1, len(self.frame_indices) - 1)
                    self.current_frame_idx = self.frame_indices[self.current_list_idx]
                else:
                    print(f"Need 4 points, have {len(self.current_points)}")
            elif key_lower == ord("s"):
                print(f"[SKIPPED] Frame {self.current_frame_idx}")
                self.current_points = []
                self.current_list_idx = min(self.current_list_idx + 1, len(self.frame_indices) - 1)
                self.current_frame_idx = self.frame_indices[self.current_list_idx]
            elif key_lower == ord("r"):
                self.current_points = []
                print(f"[RESET] Frame {self.current_frame_idx}")
            elif key_lower == 8 or key in (0x250000, 2424832):
                self.current_points = []
                self.current_list_idx = max(0, self.current_list_idx - 1)
                self.current_frame_idx = self.frame_indices[self.current_list_idx]
            elif key in (0x270000, 2555904):
                self.current_points = []
                self.current_list_idx = min(self.current_list_idx + 1, len(self.frame_indices) - 1)
                self.current_frame_idx = self.frame_indices[self.current_list_idx]
            elif key_lower == ord("q") or key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.save_annotations()


def main():
    parser = argparse.ArgumentParser(description="Tennis net pole annotation tool")
    parser.add_argument("video", type=str, nargs="?", help="Path to video file")
    parser.add_argument("--output", "-o", type=str, help="Output JSON path")
    parser.add_argument("--skip", "-s", type=int, default=30, help="Annotate every N frames")
    parser.add_argument("--resume", "-r", type=str, help="Resume from existing annotation JSON")
    args = parser.parse_args()

    if args.resume:
        with open(args.resume, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        annotator = NetPoleAnnotator(payload["video_path"], resume_path=args.resume)
    elif args.video:
        annotator = NetPoleAnnotator(args.video, args.output, args.skip)
    else:
        parser.print_help()
        return

    annotator.run()


if __name__ == "__main__":
    main()
