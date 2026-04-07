import argparse
import json
import os
import time

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch


DEFAULT_MODEL_PATH = "tennis_net_model.pth"
DEFAULT_METADATA_PATH = "tennis_net_model.json"
DEFAULT_VIDEO_PATH = "The Longest Grand Slam Rally Ever_ _ Australian Open 2013.mp4"
DEFAULT_ENCODER_NAME = "resnet18"
DEFAULT_INFERENCE_SIZE = 256
DEFAULT_INFERENCE_FRAME_STRIDE = 4
DEFAULT_SOURCE_FRAME_STRIDE = 2
DEFAULT_THRESHOLD = 0.45
POINT_LABELS = ["LT", "LB", "RT", "RB"]
POINT_COLORS = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (0, 255, 0),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run tennis net segmentation inference on a video.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--video-path", default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--encoder-name", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--inference-frame-stride", type=int, default=DEFAULT_INFERENCE_FRAME_STRIDE)
    parser.add_argument("--source-frame-stride", type=int, default=DEFAULT_SOURCE_FRAME_STRIDE)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-mask-ratio", type=float, default=0.004)
    parser.add_argument("--max-mask-ratio", type=float, default=0.08)
    parser.add_argument("--stale-frame-limit", type=int, default=3)
    parser.add_argument("--detection-score-threshold", type=float, default=0.47)
    parser.add_argument("--confirm-frames", type=int, default=2)
    return parser.parse_args()


def resize_for_display(image, max_width=1280, max_height=720):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def load_metadata(metadata_path):
    if not metadata_path or not os.path.isfile(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_model(device, encoder_name, model_path):
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


def preprocess_frame(frame, device, image_size):
    img = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.ascontiguousarray(np.expand_dims(img, axis=0))
    return torch.from_numpy(img).to(device)


def order_pole_points(points):
    points = np.asarray(points, dtype=np.float32)
    x_sorted = points[np.argsort(points[:, 0])]
    left_pair = x_sorted[:2][np.argsort(x_sorted[:2, 1])]
    right_pair = x_sorted[2:][np.argsort(x_sorted[2:, 1])]
    return np.array([left_pair[0], left_pair[1], right_pair[0], right_pair[1]], dtype=np.float32)


def clean_mask(mask):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, horizontal_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cleanup_kernel)
    return mask


def filter_mask_components(mask, min_mask_ratio, max_mask_ratio):
    image_area = float(mask.shape[0] * mask.shape[1])
    nonzero_ratio = float(np.count_nonzero(mask) / max(image_area, 1.0))
    if nonzero_ratio < min_mask_ratio or nonzero_ratio > max_mask_ratio:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_component = None
    best_score = -1.0
    width = mask.shape[1]

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        x_coord = int(stats[label_idx, cv2.CC_STAT_LEFT])
        component_width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        if area < image_area * min_mask_ratio * 0.4:
            continue
        if component_width < width * 0.25:
            continue
        if component_height < mask.shape[0] * 0.02:
            continue

        center_x = x_coord + component_width / 2.0
        center_penalty = abs(center_x - width / 2.0) / max(width / 2.0, 1.0)
        score = area * (1.0 - 0.35 * center_penalty)
        if score > best_score:
            best_score = score
            best_component = label_idx

    if best_component is None:
        return None

    filtered = np.zeros_like(mask)
    filtered[labels == best_component] = 255
    filtered_ratio = float(np.count_nonzero(filtered) / max(image_area, 1.0))
    if filtered_ratio < min_mask_ratio or filtered_ratio > max_mask_ratio:
        return None
    return filtered


def line_intersection(line1, line2):
    point1, direction1 = line1
    point2, direction2 = line2
    cross = direction1[0] * direction2[1] - direction1[1] * direction2[0]
    if abs(float(cross)) < 1e-6:
        return None

    delta = point2 - point1
    scale = (delta[0] * direction2[1] - delta[1] * direction2[0]) / cross
    return point1 + scale * direction1


def point_on_line_at_x(line, x_coord):
    point, direction = line
    if abs(float(direction[0])) < 1e-6:
        return None
    scale = (float(x_coord) - float(point[0])) / float(direction[0])
    return point + scale * direction


def is_plausible_net(points, image_shape):
    if points is None or len(points) != 4:
        return False

    points = order_pole_points(points)
    height, width = image_shape[:2]

    left_height = float(np.linalg.norm(points[1] - points[0]))
    right_height = float(np.linalg.norm(points[3] - points[2]))
    top_width = float(np.linalg.norm(points[2] - points[0]))
    bottom_width = float(np.linalg.norm(points[3] - points[1]))
    avg_width = (top_width + bottom_width) * 0.5
    avg_height = (left_height + right_height) * 0.5

    if avg_width < width * 0.25:
        return False
    if avg_height < height * 0.04 or avg_height > height * 0.45:
        return False
    if points[0][0] >= points[2][0] or points[1][0] >= points[3][0]:
        return False
    if points[0][1] >= points[1][1] or points[2][1] >= points[3][1]:
        return False

    polygon = np.asarray([points[0], points[2], points[3], points[1]], dtype=np.float32)
    area_ratio = abs(cv2.contourArea(polygon)) / max(float(height * width), 1.0)
    if area_ratio < 0.01 or area_ratio > 0.22:
        return False

    return True


def interpolate_edge_point(start, end, alpha):
    return (1.0 - alpha) * start + alpha * end


def sample_line_values(image, start, end, sample_count=64):
    height, width = image.shape[:2]
    values = []
    for alpha in np.linspace(0.05, 0.95, sample_count):
        point = interpolate_edge_point(start, end, alpha)
        x_coord = int(np.clip(round(float(point[0])), 0, width - 1))
        y_coord = int(np.clip(round(float(point[1])), 0, height - 1))
        values.append(float(image[y_coord, x_coord]))
    return np.asarray(values, dtype=np.float32)


def compute_detection_score(frame, mask, points):
    if mask is None or points is None or not is_plausible_net(points, frame.shape):
        return 0.0

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2]
    saturation_channel = hsv[:, :, 1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ordered = order_pole_points(points)
    refined_mask = build_refined_net_mask(frame.shape, ordered)
    if refined_mask is None:
        return 0.0

    overlap = cv2.bitwise_and(mask, refined_mask)
    overlap_ratio = float(np.count_nonzero(overlap)) / max(float(np.count_nonzero(refined_mask)), 1.0)

    mask_ratio = float(np.count_nonzero(mask)) / max(float(mask.size), 1.0)
    coverage_score = float(np.clip(overlap_ratio, 0.0, 1.0))
    ratio_score = float(np.clip(1.0 - abs(mask_ratio - 0.02) / 0.03, 0.0, 1.0))

    top_values = sample_line_values(value_channel, ordered[0], ordered[2])
    bottom_values = sample_line_values(value_channel, ordered[1], ordered[3])
    top_saturation = sample_line_values(saturation_channel, ordered[0], ordered[2])
    bottom_saturation = sample_line_values(saturation_channel, ordered[1], ordered[3])
    top_gray = sample_line_values(gray, ordered[0], ordered[2])
    bottom_gray = sample_line_values(gray, ordered[1], ordered[3])

    brightness_score = float(
        np.clip(((top_values.mean() + bottom_values.mean()) * 0.5 - 120.0) / 100.0, 0.0, 1.0)
    )
    desaturation_score = float(
        np.clip(1.0 - ((top_saturation.mean() + bottom_saturation.mean()) * 0.5) / 140.0, 0.0, 1.0)
    )
    band_consistency = float(
        np.clip(1.0 - abs(top_gray.mean() - bottom_gray.mean()) / 80.0, 0.0, 1.0)
    )

    score = (
        0.38 * coverage_score +
        0.17 * ratio_score +
        0.20 * brightness_score +
        0.15 * desaturation_score +
        0.10 * band_consistency
    )
    return float(np.clip(score, 0.0, 1.0))


def sample_white_band_points(frame, start, end, search_radius, expected_half_thickness, sample_count=48):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2].astype(np.float32)
    saturation_channel = hsv[:, :, 1].astype(np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vertical_gradient = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    points = []
    height, width = frame.shape[:2]
    x_margin = 2

    for alpha in np.linspace(0.05, 0.95, sample_count):
        expected = interpolate_edge_point(start, end, alpha)
        x_coord = int(round(expected[0]))
        y_coord = int(round(expected[1]))
        x0 = max(0, x_coord - x_margin)
        x1 = min(width, x_coord + x_margin + 1)
        y0 = max(0, y_coord - search_radius)
        y1 = min(height, y_coord + search_radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        local_value = value_channel[y0:y1, x0:x1].mean(axis=1)
        local_saturation = saturation_channel[y0:y1, x0:x1].mean(axis=1)
        local_gradient = np.abs(vertical_gradient[y0:y1, x0:x1]).mean(axis=1)
        position_penalty = np.abs(np.arange(y0, y1, dtype=np.float32) - y_coord) / max(float(search_radius), 1.0)
        score = (
            1.35 * local_value
            - 0.85 * local_saturation
            + 0.35 * local_gradient
            - 18.0 * position_penalty
        )
        best_idx = int(np.argmax(score))
        best_y = float(y0 + best_idx)

        if abs(best_y - expected[1]) > search_radius:
            continue

        if expected_half_thickness > 0:
            best_y = float(np.clip(best_y, expected[1] - search_radius, expected[1] + search_radius))

        points.append([float(x_coord), best_y])

    return np.asarray(points, dtype=np.float32) if points else None


def refine_points_with_frame(frame, rough_points):
    if rough_points is None:
        return None

    ordered = order_pole_points(rough_points)
    avg_height = 0.5 * (
        float(np.linalg.norm(ordered[1] - ordered[0])) +
        float(np.linalg.norm(ordered[3] - ordered[2]))
    )
    search_radius = max(6, int(round(avg_height * 0.35)))
    expected_half_thickness = max(2.0, avg_height * 0.5)

    top_band_points = sample_white_band_points(
        frame,
        ordered[0],
        ordered[2],
        search_radius=search_radius,
        expected_half_thickness=expected_half_thickness,
    )
    bottom_band_points = sample_white_band_points(
        frame,
        ordered[1],
        ordered[3],
        search_radius=search_radius,
        expected_half_thickness=expected_half_thickness,
    )
    if top_band_points is None or bottom_band_points is None:
        return ordered

    top_line = fit_line_from_points(top_band_points)
    bottom_line = fit_line_from_points(bottom_band_points)
    if any(line is None for line in (top_line, bottom_line)):
        return ordered

    left_x = float(np.mean([ordered[0][0], ordered[1][0]]))
    right_x = float(np.mean([ordered[2][0], ordered[3][0]]))
    refined = build_points_from_band_lines(top_line, bottom_line, left_x, right_x, frame.shape)
    if refined is None:
        return ordered
    if is_plausible_net(refined, frame.shape):
        return refined
    return ordered


def fit_edge_line(points):
    if len(points) < 10:
        return None

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    point = np.array([x0.item(), y0.item()], dtype=np.float32)
    direction = np.array([vx.item(), vy.item()], dtype=np.float32)
    return point, direction


def refine_poles_with_edges(contour_points, rough_points, image_shape):
    ordered = order_pole_points(rough_points)
    edge_segments = [
        (ordered[0], ordered[1]),
        (ordered[0], ordered[2]),
        (ordered[2], ordered[3]),
        (ordered[1], ordered[3]),
    ]

    assigned_points = [[] for _ in range(4)]
    max_distance = max(image_shape[:2]) * 0.04

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
    for points_on_edge in assigned_points:
        line = fit_edge_line(points_on_edge)
        if line is None:
            return ordered
        fitted_lines.append(line)

    refined_points = []
    line_pairs = [(0, 1), (0, 3), (1, 2), (2, 3)]
    for fallback_idx, (line_a, line_b) in enumerate(line_pairs):
        intersection = line_intersection(fitted_lines[line_a], fitted_lines[line_b])
        if intersection is None:
            refined_points.append(ordered[fallback_idx])
            continue
        x_coord = np.clip(intersection[0], 0, image_shape[1] - 1)
        y_coord = np.clip(intersection[1], 0, image_shape[0] - 1)
        refined_points.append([x_coord, y_coord])

    return order_pole_points(np.asarray(refined_points, dtype=np.float32))


def extract_side_points(contour_points):
    x_coords = contour_points[:, 0]
    min_x = int(np.min(x_coords))
    max_x = int(np.max(x_coords))
    width = max(1, max_x - min_x)
    x_margin = max(4, int(width * 0.08))

    left_points = contour_points[x_coords <= min_x + x_margin]
    right_points = contour_points[x_coords >= max_x - x_margin]
    return left_points, right_points


def robust_pole_endpoints(side_points, fallback_x):
    if len(side_points) < 6:
        return None

    ordered = side_points[np.argsort(side_points[:, 1])]
    cluster_size = max(3, min(len(ordered) // 5, 20))
    top_cluster = ordered[:cluster_size]
    bottom_cluster = ordered[-cluster_size:]

    top_x = float(np.median(top_cluster[:, 0]))
    top_y = float(np.mean(top_cluster[:, 1]))
    bottom_x = float(np.median(bottom_cluster[:, 0]))
    bottom_y = float(np.mean(bottom_cluster[:, 1]))

    if bottom_y <= top_y:
        return None

    return np.asarray(
        [
            [top_x if np.isfinite(top_x) else fallback_x, top_y],
            [bottom_x if np.isfinite(bottom_x) else fallback_x, bottom_y],
        ],
        dtype=np.float32,
    )


def extract_poles_from_side_clusters(contour_points):
    left_points, right_points = extract_side_points(contour_points)
    if len(left_points) < 6 or len(right_points) < 6:
        return None

    left_center_x = float(np.median(left_points[:, 0]))
    right_center_x = float(np.median(right_points[:, 0]))
    left_endpoints = robust_pole_endpoints(left_points, left_center_x)
    right_endpoints = robust_pole_endpoints(right_points, right_center_x)
    if left_endpoints is None or right_endpoints is None:
        return None

    return np.asarray(
        [
            left_endpoints[0],
            left_endpoints[1],
            right_endpoints[0],
            right_endpoints[1],
        ],
        dtype=np.float32,
    )


def select_band_points(contour_points, axis: int, target: str, margin_ratio: float):
    values = contour_points[:, axis]
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = max(1.0, maximum - minimum)
    margin = max(4.0, span * margin_ratio)

    if target == "min":
        mask = values <= minimum + margin
    else:
        mask = values >= maximum - margin
    return contour_points[mask]


def fit_line_from_points(points: np.ndarray):
    if len(points) < 2:
        return None
    if len(points) == 2:
        p0 = np.asarray(points[0], dtype=np.float32)
        p1 = np.asarray(points[1], dtype=np.float32)
        direction = p1 - p0
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        return p0, direction / norm
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    return (
        np.array([x0.item(), y0.item()], dtype=np.float32),
        np.array([vx.item(), vy.item()], dtype=np.float32),
    )


def build_points_from_band_lines(top_line, bottom_line, left_x, right_x, image_shape):
    top_left = point_on_line_at_x(top_line, left_x)
    top_right = point_on_line_at_x(top_line, right_x)
    bottom_left = point_on_line_at_x(bottom_line, left_x)
    bottom_right = point_on_line_at_x(bottom_line, right_x)
    if any(point is None for point in (top_left, top_right, bottom_left, bottom_right)):
        return None

    points = order_pole_points(
        np.asarray([top_left, bottom_left, top_right, bottom_right], dtype=np.float32)
    )
    points[:, 0] = np.clip(points[:, 0], 0, image_shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, image_shape[0] - 1)
    return points


def extract_poles_from_mask_profile(mask):
    xs = np.where(mask.any(axis=0))[0]
    if len(xs) < 20:
        return None

    top_points = []
    bottom_points = []
    for x_coord in xs:
        ys = np.where(mask[:, x_coord] > 0)[0]
        if len(ys) == 0:
            continue
        top_points.append([float(x_coord), float(ys[0])])
        bottom_points.append([float(x_coord), float(ys[-1])])

    if len(top_points) < 20 or len(bottom_points) < 20:
        return None

    top_points = np.asarray(top_points, dtype=np.float32)
    bottom_points = np.asarray(bottom_points, dtype=np.float32)
    band_width = max(6, int(len(xs) * 0.08))
    left_x = float(np.median(xs[:band_width]))
    right_x = float(np.median(xs[-band_width:]))

    top_line = fit_line_from_points(top_points)
    bottom_line = fit_line_from_points(bottom_points)
    if any(line is None for line in (top_line, bottom_line)):
        return None

    return build_points_from_band_lines(top_line, bottom_line, left_x, right_x, mask.shape)


def extract_poles_from_edge_lines(contour_points, image_shape):
    left_points = select_band_points(contour_points, axis=0, target="min", margin_ratio=0.10)
    right_points = select_band_points(contour_points, axis=0, target="max", margin_ratio=0.10)
    top_points = select_band_points(contour_points, axis=1, target="min", margin_ratio=0.10)
    bottom_points = select_band_points(contour_points, axis=1, target="max", margin_ratio=0.10)

    left_line = fit_line_from_points(left_points)
    right_line = fit_line_from_points(right_points)
    top_line = fit_line_from_points(top_points)
    bottom_line = fit_line_from_points(bottom_points)
    if any(line is None for line in (left_line, right_line, top_line, bottom_line)):
        return None

    intersections = []
    for line_a, line_b in (
        (left_line, top_line),
        (left_line, bottom_line),
        (right_line, top_line),
        (right_line, bottom_line),
    ):
        point = line_intersection(line_a, line_b)
        if point is None:
            return None
        intersections.append(point)

    points = order_pole_points(np.asarray(intersections, dtype=np.float32))
    points[:, 0] = np.clip(points[:, 0], 0, image_shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, image_shape[0] - 1)
    return points


def extract_net_poles(mask, frame=None):
    cleaned_mask = clean_mask(mask)
    profile_points = extract_poles_from_mask_profile(cleaned_mask)
    if is_plausible_net(profile_points, cleaned_mask.shape):
        if frame is not None:
            profile_points = refine_points_with_frame(frame, profile_points)
        return profile_points, cleaned_mask

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, cleaned_mask

    contour = max(contours, key=cv2.contourArea)
    min_area = cleaned_mask.shape[0] * cleaned_mask.shape[1] * 0.0005
    if cv2.contourArea(contour) < min_area:
        return None, cleaned_mask

    contour_points = contour.reshape(-1, 2).astype(np.float32)
    edge_line_points = extract_poles_from_edge_lines(contour_points, cleaned_mask.shape)
    if is_plausible_net(edge_line_points, cleaned_mask.shape):
        if frame is not None:
            edge_line_points = refine_points_with_frame(frame, edge_line_points)
        return edge_line_points, cleaned_mask

    clustered_points = extract_poles_from_side_clusters(contour_points)
    if clustered_points is not None:
        clustered_points = order_pole_points(clustered_points)
        if is_plausible_net(clustered_points, cleaned_mask.shape):
            if frame is not None:
                clustered_points = refine_points_with_frame(frame, clustered_points)
            return clustered_points, cleaned_mask

    perimeter = cv2.arcLength(contour, True)
    candidates = []
    for epsilon_scale in np.linspace(0.005, 0.04, 12):
        approx = cv2.approxPolyDP(contour, epsilon_scale * perimeter, True)
        if len(approx) == 4:
            ordered = order_pole_points(approx.reshape(-1, 2))
            area = cv2.contourArea(np.asarray([ordered[0], ordered[2], ordered[3], ordered[1]], dtype=np.float32))
            candidates.append((area, ordered))

    if candidates:
        rough_points = max(candidates, key=lambda item: item[0])[1]
    else:
        rect = cv2.minAreaRect(contour)
        rough_points = order_pole_points(cv2.boxPoints(rect))

    refined_points = refine_poles_with_edges(contour_points, rough_points, cleaned_mask.shape)
    if is_plausible_net(refined_points, cleaned_mask.shape):
        if frame is not None:
            refined_points = refine_points_with_frame(frame, refined_points)
        return refined_points, cleaned_mask
    return None, cleaned_mask


def smooth_points(previous_points, current_points, alpha=0.25):
    if previous_points is None or current_points is None:
        return current_points
    return ((1.0 - alpha) * previous_points + alpha * current_points).astype(np.float32)


def build_refined_net_mask(image_shape, points):
    if points is None:
        return None

    points = order_pole_points(np.asarray(points, dtype=np.float32))
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    left_height = float(np.linalg.norm(points[1] - points[0]))
    right_height = float(np.linalg.norm(points[3] - points[2]))
    avg_height = max(2.0, 0.5 * (left_height + right_height))

    top_thickness = max(2, int(round(avg_height * 0.14)))
    bottom_thickness = max(2, int(round(avg_height * 0.10)))
    pole_thickness = max(2, int(round(avg_height * 0.12)))

    points_int = points.astype(np.int32)
    cv2.line(mask, tuple(points_int[0]), tuple(points_int[2]), 255, top_thickness, cv2.LINE_AA)
    cv2.line(mask, tuple(points_int[1]), tuple(points_int[3]), 255, bottom_thickness, cv2.LINE_AA)
    cv2.line(mask, tuple(points_int[0]), tuple(points_int[1]), 255, pole_thickness, cv2.LINE_AA)
    cv2.line(mask, tuple(points_int[2]), tuple(points_int[3]), 255, pole_thickness, cv2.LINE_AA)
    return mask


def build_display_mask(raw_mask, points):
    if raw_mask is None:
        return None
    if points is None:
        return raw_mask

    refined_mask = build_refined_net_mask((raw_mask.shape[0], raw_mask.shape[1], 3), points)
    if refined_mask is None:
        return raw_mask

    tightened_raw = cv2.erode(raw_mask, np.ones((3, 3), np.uint8), iterations=1)
    combined = cv2.bitwise_and(tightened_raw, refined_mask)
    if np.count_nonzero(combined) == 0:
        combined = refined_mask
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def draw_poles(frame, points):
    if points is None:
        return frame

    output = frame.copy()
    points_int = points.astype(np.int32)
    polygon = np.asarray(
        [points_int[0], points_int[2], points_int[3], points_int[1]],
        dtype=np.int32,
    )
    cv2.polylines(output, [polygon], True, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(output, tuple(points_int[0]), tuple(points_int[1]), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(output, tuple(points_int[2]), tuple(points_int[3]), (255, 255, 255), 3, cv2.LINE_AA)

    for point, label, color in zip(points_int, POINT_LABELS, POINT_COLORS):
        x_coord, y_coord = int(point[0]), int(point[1])
        cv2.circle(output, (x_coord, y_coord), 8, color, -1, cv2.LINE_AA)
        cv2.putText(
            output,
            label,
            (x_coord + 10, y_coord - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def main():
    args = parse_args()
    metadata = load_metadata(args.metadata_path)
    encoder_name = args.encoder_name or metadata.get("encoder_name", DEFAULT_ENCODER_NAME)
    image_size = args.image_size or int(metadata.get("image_size", DEFAULT_INFERENCE_SIZE))
    threshold = args.threshold
    if threshold is None:
        threshold = float(metadata.get("recommended_threshold", DEFAULT_THRESHOLD))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device, encoder_name, args.model_path)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video_path}")

    cv2.namedWindow("Tennis Net Detection", cv2.WINDOW_NORMAL)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_frame_time = args.source_frame_stride / max(video_fps, 1.0)

    previous_points = None
    previous_mask = None
    candidate_points = None
    candidate_mask = None
    hit_streak = 0
    missed_detections = 0
    frame_index = 0

    while True:
        frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        run_inference = previous_mask is None or frame_index % args.inference_frame_stride == 0
        if run_inference:
            tensor = preprocess_frame(frame, device, image_size)
            with torch.inference_mode():
                pred = model(tensor)
                pred = torch.sigmoid(pred).squeeze().detach().cpu().numpy().astype(np.float32)

            pred = cv2.resize(pred, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            pred = (pred > threshold).astype(np.uint8) * 255
            pred = filter_mask_components(pred, args.min_mask_ratio, args.max_mask_ratio)

            if pred is not None:
                detected_points, cleaned_mask = extract_net_poles(pred, frame=frame)
            else:
                detected_points, cleaned_mask = None, None

            if detected_points is not None and cleaned_mask is not None:
                detection_score = compute_detection_score(frame, cleaned_mask, detected_points)
                if detection_score >= args.detection_score_threshold:
                    hit_streak += 1
                    missed_detections = 0
                    candidate_points = smooth_points(candidate_points, detected_points, alpha=0.4)
                    candidate_mask = cleaned_mask

                    if previous_points is None:
                        if hit_streak >= args.confirm_frames:
                            previous_points = candidate_points
                            previous_mask = candidate_mask
                    else:
                        previous_points = smooth_points(previous_points, detected_points, alpha=0.3)
                        previous_mask = cleaned_mask
                else:
                    hit_streak = 0
                    candidate_points = None
                    candidate_mask = None
                    missed_detections += 1
                    if missed_detections >= args.stale_frame_limit:
                        previous_points = None
                        previous_mask = None
            else:
                hit_streak = 0
                candidate_points = None
                candidate_mask = None
                missed_detections += 1
                if missed_detections >= args.stale_frame_limit:
                    previous_points = None
                    previous_mask = None

        output = frame.copy()
        if previous_mask is not None:
            colored_mask = np.zeros_like(frame)
            display_mask = build_display_mask(previous_mask, previous_points)
            colored_mask[:, :, 1] = display_mask
            output = cv2.addWeighted(output, 1.0, colored_mask, 0.5, 0)

        output = draw_poles(output, previous_points)
        cv2.imshow("Tennis Net Detection", resize_for_display(output))

        elapsed = time.perf_counter() - frame_start
        wait_ms = max(1, int((target_frame_time - elapsed) * 1000))
        if cv2.waitKey(wait_ms) & 0xFF == 27:
            break

        for _ in range(args.source_frame_stride - 1):
            if not cap.grab():
                ret = False
                break
        if not ret:
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
