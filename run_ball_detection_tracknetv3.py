import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_VIDEO_PATH = (
    "Roger_Federer_v_Rafael_Nadal_Extended_Highlights_Australian_Open_2009_Final_1080P.mp4"
)
DEFAULT_TRACKNET_REPO = r"C:\Users\manoj\TrackNetV3"
DEFAULT_TRACKNET_FILE = r"C:\Users\manoj\TrackNetV3\ckpts\TrackNet_best.pt"
DEFAULT_INPAINTNET_FILE = r"C:\Users\manoj\TrackNetV3\ckpts\InpaintNet_best.pt"
DEFAULT_SAVE_DIR = r"C:\Users\manoj\TrackNetV3\prediction"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TrackNetV3 inference on a tennis video via the official repo."
    )
    parser.add_argument("--video-path", default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--tracknet-repo", default=DEFAULT_TRACKNET_REPO)
    parser.add_argument("--tracknet-file", default=DEFAULT_TRACKNET_FILE)
    parser.add_argument("--inpaintnet-file", default=DEFAULT_INPAINTNET_FILE)
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--large-video", action="store_true")
    parser.add_argument("--video-range", default=None, help='Example: "324,330"')
    parser.add_argument("--max-sample-num", type=int, default=120)
    parser.add_argument("--output-video", action="store_true")
    return parser.parse_args()


def require_existing_path(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def main():
    args = parse_args()

    tracknet_repo = require_existing_path(args.tracknet_repo, "TrackNetV3 repo")
    predict_script = require_existing_path(
        str(tracknet_repo / "predict.py"),
        "TrackNetV3 predict.py",
    )
    video_path = require_existing_path(args.video_path, "Video file")
    tracknet_file = require_existing_path(args.tracknet_file, "TrackNet checkpoint")
    inpaintnet_file = require_existing_path(args.inpaintnet_file, "InpaintNet checkpoint")

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.python_exe,
        str(predict_script),
        "--video_file",
        str(video_path),
        "--tracknet_file",
        str(tracknet_file),
        "--inpaintnet_file",
        str(inpaintnet_file),
        "--batch_size",
        str(args.batch_size),
        "--save_dir",
        str(save_dir),
    ]

    if args.output_video:
        command.append("--output_video")
    if args.large_video:
        command.append("--large_video")
    if args.video_range:
        command.extend(["--video_range", args.video_range])
    if args.max_sample_num:
        command.extend(["--max_sample_num", str(args.max_sample_num)])

    print("Running TrackNetV3:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))

    subprocess.run(command, check=True, cwd=str(tracknet_repo))


if __name__ == "__main__":
    main()
