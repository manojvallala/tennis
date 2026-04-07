#!/usr/bin/env python3
"""
Court Corner Annotation Tool

Click on the 4 corners of the tennis court in order:
    1. Top-left
    2. Top-right
    3. Bottom-right
    4. Bottom-left

Controls:
    LEFT CLICK      - Place/move corner point
    RIGHT CLICK     - Remove last point
    ENTER/SPACE     - Save annotation and go to next frame
    BACKSPACE       - Go to previous frame
    S               - Skip frame (no annotation)
    R               - Reset points for current frame
    Q/ESC           - Save and quit

Usage:
    python annotate_corners.py video.mp4
    python annotate_corners.py video.mp4 --output annotations.json
    python annotate_corners.py video.mp4 --skip 10  # annotate every 10th frame
    python annotate_corners.py --resume annotations.json  # resume from existing
"""

import cv2
import json
import argparse
from pathlib import Path
from typing import Optional


# Corner labels and colors
CORNER_NAMES = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
CORNER_COLORS = [
    (0, 0, 255),    # Red - TL
    (0, 255, 0),    # Green - TR
    (255, 0, 0),    # Blue - BR
    (0, 255, 255),  # Yellow - BL
]

class CornerAnnotator:
    def __init__(self, video_path: str, output_path: Optional[str] = None,
                 skip_frames: int = 1, resume_path: Optional[str] = None):
        self.video_path = Path(video_path)
        self.skip_frames = skip_frames

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Setup output path
        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = self.video_path.parent / f"{self.video_path.stem}_court_annotations.json"

        # Load video
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Current state
        self.current_frame_idx = 0
        self.current_points = []  # List of (x, y) for current frame
        self.current_frame = None

        # All annotations: {frame_idx: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}
        self.annotations = {}

        # Resume from existing annotations
        if resume_path and Path(resume_path).exists():
            self.load_annotations(resume_path)
            self.output_path = Path(resume_path)

        # Frame indices to annotate
        self.frame_indices = list(range(0, self.total_frames, skip_frames))
        self.current_list_idx = 0

        # Find starting point if resuming
        if self.annotations:
            # Find first unannotated frame
            for i, fidx in enumerate(self.frame_indices):
                if fidx not in self.annotations:
                    self.current_list_idx = i
                    break
            else:
                self.current_list_idx = 0  # All done, start from beginning

        self.current_frame_idx = self.frame_indices[self.current_list_idx]

    def load_annotations(self, path: str):
        """Load existing annotations from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
            # Convert string keys back to int
            self.annotations = {int(k): v for k, v in data.get('annotations', {}).items()}
            print(f"Loaded {len(self.annotations)} existing annotations.")

    def save_annotations(self):
        """Save annotations to JSON."""
        data = {
            'video_path': str(self.video_path),
            'video_name': self.video_path.name,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'skip_frames': self.skip_frames,
            'corner_order': CORNER_NAMES,
            'annotations': self.annotations
        }
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.annotations)} annotations to {self.output_path}")

    def load_frame(self, frame_idx: int):
        """Load a specific frame from video."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.current_points) < 4:
                self.current_points.append([x, y])
            else:
                # Find closest point and move it
                min_dist = float('inf')
                min_idx = 0
                for i, (px, py) in enumerate(self.current_points):
                    dist = (x - px) ** 2 + (y - py) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = i
                self.current_points[min_idx] = [x, y]

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_points:
                self.current_points.pop()

    def draw_overlay(self, frame):
        """Draw points and UI on frame."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw existing points
        for i, (x, y) in enumerate(self.current_points):
            color = CORNER_COLORS[i]
            cv2.circle(display, (x, y), 8, color, -1)
            cv2.circle(display, (x, y), 10, (255, 255, 255), 2)
            cv2.putText(display, CORNER_NAMES[i], (x + 15, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw lines connecting points
        if len(self.current_points) >= 2:
            for i in range(len(self.current_points)):
                if i + 1 < len(self.current_points):
                    pt1 = tuple(self.current_points[i])
                    pt2 = tuple(self.current_points[i + 1])
                    cv2.line(display, pt1, pt2, (255, 255, 255), 2)
            # Close the quadrilateral
            if len(self.current_points) == 4:
                pt1 = tuple(self.current_points[3])
                pt2 = tuple(self.current_points[0])
                cv2.line(display, pt1, pt2, (255, 255, 255), 2)

        # Top bar - info
        cv2.rectangle(display, (0, 0), (w, 80), (0, 0, 0), -1)

        progress = len(self.annotations)
        total_to_annotate = len(self.frame_indices)

        cv2.putText(display, f"Frame: {self.current_frame_idx}/{self.total_frames-1}",
                    (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Progress: {progress}/{total_to_annotate} annotated",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show which point to place next
        if len(self.current_points) < 4:
            next_corner = CORNER_NAMES[len(self.current_points)]
            next_color = CORNER_COLORS[len(self.current_points)]
            cv2.putText(display, f"Click: {next_corner}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, next_color, 2)
        else:
            cv2.putText(display, "All 4 corners placed. ENTER to save.",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show if already annotated
        if self.current_frame_idx in self.annotations:
            cv2.putText(display, "[ANNOTATED]", (w - 180, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Bottom bar - controls
        cv2.rectangle(display, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.putText(display, "[Click] Place | [Right-Click] Undo | [Enter] Save | [S] Skip | [R] Reset | [Q] Quit",
                    (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return display

    def run(self):
        """Run the annotation tool."""
        print("=" * 60)
        print("COURT CORNER ANNOTATION TOOL")
        print("=" * 60)
        print(f"Video: {self.video_path.name}")
        print(f"Frames to annotate: {len(self.frame_indices)} (every {self.skip_frames} frames)")
        print(f"Output: {self.output_path}")
        print(f"Already annotated: {len(self.annotations)}")
        print("\nClick corners in order: TL -> TR -> BR -> BL")
        print("-" * 60)

        window_name = "Court Corner Annotator"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while True:
            # Load current frame
            self.current_frame = self.load_frame(self.current_frame_idx)
            if self.current_frame is None:
                print(f"Error loading frame {self.current_frame_idx}")
                break

            # Load existing annotation if available
            if self.current_frame_idx in self.annotations and not self.current_points:
                self.current_points = [list(p) for p in self.annotations[self.current_frame_idx]]

            # Draw and display
            display = self.draw_overlay(self.current_frame)
            cv2.imshow(window_name, display)

            key = cv2.waitKeyEx(30)
            if key == -1:
                continue

            key_lower = key & 0xFF

            # ENTER/SPACE - Save and next
            if key_lower == 13 or key_lower == 32:
                if len(self.current_points) == 4:
                    self.annotations[self.current_frame_idx] = self.current_points.copy()
                    print(f"[SAVED] Frame {self.current_frame_idx} - {len(self.annotations)} total")
                    self.current_points = []
                    self.current_list_idx = min(self.current_list_idx + 1, len(self.frame_indices) - 1)
                    self.current_frame_idx = self.frame_indices[self.current_list_idx]
                else:
                    print(f"Need 4 points, have {len(self.current_points)}")

            # S - Skip frame
            elif key_lower == ord('s'):
                print(f"[SKIPPED] Frame {self.current_frame_idx}")
                self.current_points = []
                self.current_list_idx = min(self.current_list_idx + 1, len(self.frame_indices) - 1)
                self.current_frame_idx = self.frame_indices[self.current_list_idx]

            # R - Reset points
            elif key_lower == ord('r'):
                self.current_points = []
                print(f"[RESET] Frame {self.current_frame_idx}")

            # BACKSPACE - Previous frame
            elif key_lower == 8 or key == 0x250000 or key == 2424832:  # Backspace or Left
                self.current_points = []
                self.current_list_idx = max(0, self.current_list_idx - 1)
                self.current_frame_idx = self.frame_indices[self.current_list_idx]

            # Right arrow - Next frame (without saving)
            elif key == 0x270000 or key == 2555904:
                self.current_points = []
                self.current_list_idx = min(self.current_list_idx + 1, len(self.frame_indices) - 1)
                self.current_frame_idx = self.frame_indices[self.current_list_idx]

            # Q/ESC - Quit
            elif key_lower == ord('q') or key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.save_annotations()


def main():
    parser = argparse.ArgumentParser(description="Court Corner Annotation Tool")
    parser.add_argument('video', type=str, nargs='?', help='Path to video file')
    parser.add_argument('--output', '-o', type=str, help='Output JSON path')
    parser.add_argument('--skip', '-s', type=int, default=30,
                        help='Annotate every N frames (default: 30)')
    parser.add_argument('--resume', '-r', type=str,
                        help='Resume from existing annotations JSON')

    args = parser.parse_args()

    if args.resume:
        # Load video path from resume file
        with open(args.resume, 'r') as f:
            data = json.load(f)
            video_path = data['video_path']
        annotator = CornerAnnotator(video_path, resume_path=args.resume)
    elif args.video:
        annotator = CornerAnnotator(args.video, args.output, args.skip)
    else:
        parser.print_help()
        return

    annotator.run()


if __name__ == "__main__":
    main()
