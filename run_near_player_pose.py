import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from ultralytics import YOLO

from near_player_pose_filter import (
    build_pose_target_from_yolo,
    crop_pose_input,
    select_near_camera_player,
)


DEFAULT_VIDEO_PATH = "Roger_Federer_v_Rafael_Nadal_Extended_Highlights_Australian_Open_2009_Final_1080P.mp4"
DEFAULT_COURT_MODEL_PATH = "experiments/checkpoints/unet_mobilenet_v2_192.pth"
DEFAULT_COURT_ENCODER = "mobilenet_v2"
DEFAULT_COURT_IMAGE_SIZE = 192
DEFAULT_PERSON_MODEL = "yolov8n.pt"
DEFAULT_POSE_MODEL = "yolov8n-pose.pt"
DEFAULT_PERSON_CONF = 0.25
DEFAULT_POSE_CONF = 0.25
DEFAULT_COURT_INFERENCE_STRIDE = 6
PLACEHOLDER_MODEL_NAMES = {"your_person_model.pt", "your_pose_model.pt"}
POSE_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pose estimation only for the near-camera tennis player."
    )
    parser.add_argument("--video-path", default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--court-model-path", default=DEFAULT_COURT_MODEL_PATH)
    parser.add_argument("--court-encoder-name", default=DEFAULT_COURT_ENCODER)
    parser.add_argument("--court-image-size", type=int, default=DEFAULT_COURT_IMAGE_SIZE)
    parser.add_argument("--person-model", default=DEFAULT_PERSON_MODEL)
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL)
    parser.add_argument("--person-conf", type=float, default=DEFAULT_PERSON_CONF)
    parser.add_argument("--pose-conf", type=float, default=DEFAULT_POSE_CONF)
    parser.add_argument("--court-inference-stride", type=int, default=DEFAULT_COURT_INFERENCE_STRIDE)
    parser.add_argument("--show-court", action="store_true")
    parser.add_argument("--show-person-box", action="store_true")
    return parser.parse_args()


def resolve_yolo_model_path(model_arg, fallback_name):
    model_name = Path(model_arg).name.lower()
    if model_name in PLACEHOLDER_MODEL_NAMES:
        print(
            f"Model path '{model_arg}' is a placeholder example, not a real file. "
            f"Using default '{fallback_name}' instead."
        )
        return fallback_name
    return model_arg


def resize_for_display(image, max_width=1280, max_height=720):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def build_court_model(device, encoder_name, model_path):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_court_frame(frame, device, image_size):
    img = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.ascontiguousarray(np.expand_dims(img, axis=0))
    return torch.from_numpy(img).to(device)


def order_corners(points):
    points = np.asarray(points, dtype=np.float32)
    y_sorted = points[np.argsort(points[:, 1])]
    top = y_sorted[:2][np.argsort(y_sorted[:2, 0])]
    bottom = y_sorted[2:][np.argsort(y_sorted[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def clean_mask(mask):
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def line_intersection(line1, line2):
    point1, direction1 = line1
    point2, direction2 = line2
    cross = direction1[0] * direction2[1] - direction1[1] * direction2[0]
    if abs(float(cross)) < 1e-6:
        return None

    delta = point2 - point1
    scale = (delta[0] * direction2[1] - delta[1] * direction2[0]) / cross
    return point1 + scale * direction1


def fit_edge_line(points):
    if len(points) < 20:
        return None

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    point = np.array([x0.item(), y0.item()], dtype=np.float32)
    direction = np.array([vx.item(), vy.item()], dtype=np.float32)
    return point, direction


def refine_corners_with_edges(contour_points, rough_corners, image_shape):
    rough_corners = order_corners(rough_corners)
    edge_segments = [
        (rough_corners[0], rough_corners[1]),
        (rough_corners[1], rough_corners[2]),
        (rough_corners[2], rough_corners[3]),
        (rough_corners[3], rough_corners[0]),
    ]

    assigned_points = [[] for _ in range(4)]
    max_distance = max(image_shape[:2]) * 0.05

    for point in contour_points:
        point = point.astype(np.float32)
        best_edge_idx = None
        best_distance = float("inf")

        for edge_idx, (start, end) in enumerate(edge_segments):
            edge_vector = end - start
            denom = float(np.dot(edge_vector, edge_vector))
            if denom < 1e-6:
                continue

            t = float(np.dot(point - start, edge_vector) / denom)
            if t < -0.1 or t > 1.1:
                continue

            projection = start + t * edge_vector
            distance = float(np.linalg.norm(point - projection))

            if distance < best_distance:
                best_distance = distance
                best_edge_idx = edge_idx

        if best_edge_idx is not None and best_distance < max_distance:
            assigned_points[best_edge_idx].append(point)

    fitted_lines = []
    for points in assigned_points:
        line = fit_edge_line(points)
        if line is None:
            return rough_corners
        fitted_lines.append(line)

    refined_corners = []
    line_pairs = [(3, 0), (0, 1), (1, 2), (2, 3)]
    for fallback_idx, (line_a, line_b) in enumerate(line_pairs):
        intersection = line_intersection(fitted_lines[line_a], fitted_lines[line_b])
        if intersection is None:
            refined_corners.append(rough_corners[fallback_idx])
            continue

        x_coord = np.clip(intersection[0], 0, image_shape[1] - 1)
        y_coord = np.clip(intersection[1], 0, image_shape[0] - 1)
        refined_corners.append([x_coord, y_coord])

    return order_corners(np.asarray(refined_corners, dtype=np.float32))


def extract_court_corners(mask):
    cleaned_mask = clean_mask(mask)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    min_area = cleaned_mask.shape[0] * cleaned_mask.shape[1] * 0.05
    if cv2.contourArea(contour) < min_area:
        return None

    perimeter = cv2.arcLength(contour, True)
    candidates = []
    for epsilon_scale in np.linspace(0.008, 0.06, 16):
        approx = cv2.approxPolyDP(contour, epsilon_scale * perimeter, True)
        if len(approx) == 4:
            ordered = order_corners(approx.reshape(-1, 2))
            area = cv2.contourArea(ordered.astype(np.float32))
            candidates.append((area, ordered))

    if not candidates:
        return None

    rough_corners = max(candidates, key=lambda item: item[0])[1]
    return refine_corners_with_edges(
        contour.reshape(-1, 2),
        rough_corners,
        cleaned_mask.shape,
    )


def infer_court_corners(frame, model, device, image_size):
    tensor = preprocess_court_frame(frame, device, image_size)
    with torch.inference_mode():
        pred = model(tensor)
        pred = torch.sigmoid(pred).squeeze().detach().cpu().numpy().astype(np.float32)

    pred = cv2.resize(pred, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    pred = (pred > 0.5).astype(np.uint8) * 255
    return extract_court_corners(pred)


def draw_court(frame, corners):
    if corners is None:
        return frame

    output = frame.copy()
    points = corners.astype(np.int32)
    cv2.polylines(output, [points], True, (0, 255, 255), 3, cv2.LINE_AA)
    return output


def draw_person_box(frame, bbox):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        "near player",
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_pose_on_frame(frame, pose_result, crop_box):
    output = frame.copy()
    if not pose_result or pose_result[0].keypoints is None:
        return output

    x1, y1, _, _ = crop_box
    keypoints_xy = pose_result[0].keypoints.xy
    keypoints_conf = pose_result[0].keypoints.conf
    if keypoints_xy is None:
        return output

    xy = keypoints_xy.detach().cpu().numpy()
    conf = keypoints_conf.detach().cpu().numpy() if keypoints_conf is not None else None
    if len(xy) == 0:
        return output

    points_to_draw = []
    for idx, point in enumerate(xy[0]):
        px = int(round(point[0] + x1))
        py = int(round(point[1] + y1))
        point_conf = conf[0][idx] if conf is not None else 1.0
        points_to_draw.append((px, py, float(point_conf)))
        if point_conf < 0.2:
            continue
        cv2.circle(output, (px, py), 4, (0, 0, 255), -1, cv2.LINE_AA)

    for start_idx, end_idx in POSE_EDGES:
        if start_idx >= len(points_to_draw) or end_idx >= len(points_to_draw):
            continue
        x_start, y_start, conf_start = points_to_draw[start_idx]
        x_end, y_end, conf_end = points_to_draw[end_idx]
        if conf_start < 0.2 or conf_end < 0.2:
            continue
        cv2.line(
            output,
            (x_start, y_start),
            (x_end, y_end),
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    person_model_path = resolve_yolo_model_path(args.person_model, DEFAULT_PERSON_MODEL)
    pose_model_path = resolve_yolo_model_path(args.pose_model, DEFAULT_POSE_MODEL)

    court_model = build_court_model(
        device=device,
        encoder_name=args.court_encoder_name,
        model_path=args.court_model_path,
    )
    person_model = YOLO(person_model_path)
    pose_model = YOLO(pose_model_path)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video_path}")

    cv2.namedWindow("Near Player Pose", cv2.WINDOW_NORMAL)
    frame_index = 0
    cached_court_corners = None

    while True:
        frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        if cached_court_corners is None or frame_index % args.court_inference_stride == 0:
            detected_corners = infer_court_corners(
                frame=frame,
                model=court_model,
                device=device,
                image_size=args.court_image_size,
            )
            if detected_corners is not None:
                cached_court_corners = detected_corners

        person_result = person_model(frame, verbose=False, conf=args.person_conf)[0]
        detections = build_pose_target_from_yolo(person_result)
        target_player = select_near_camera_player(
            detections=detections,
            frame_shape=frame.shape,
            court_corners=cached_court_corners,
        )

        output = frame.copy()
        if args.show_court:
            output = draw_court(output, cached_court_corners)

        if target_player is not None:
            player_crop, crop_box = crop_pose_input(frame, target_player)
            pose_result = pose_model(player_crop, verbose=False, conf=args.pose_conf)
            output = draw_pose_on_frame(output, pose_result, crop_box)
            if args.show_person_box:
                output = draw_person_box(output, crop_box)

        elapsed_ms = (time.perf_counter() - frame_start) * 1000.0
        cv2.putText(
            output,
            f"frame={frame_index}  {elapsed_ms:.1f} ms",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Near Player Pose", resize_for_display(output))
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
