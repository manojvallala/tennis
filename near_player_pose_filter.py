from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch


PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class PersonDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int = PERSON_CLASS_ID

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return 0.5 * (self.bbox[0] + self.bbox[2])

    @property
    def foot_point(self) -> tuple[float, float]:
        return self.center_x, float(self.bbox[3])


def build_court_model(device, encoder_name: str, model_path: str):
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


def preprocess_court_frame(frame: np.ndarray, device, image_size: int) -> torch.Tensor:
    img = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.ascontiguousarray(np.expand_dims(img, axis=0))
    return torch.from_numpy(img).to(device)


def order_court_corners(corners: np.ndarray | list[list[float]]) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape != (4, 2):
        raise ValueError("Court corners must have shape (4, 2)")

    y_sorted = corners[np.argsort(corners[:, 1])]
    top = y_sorted[:2][np.argsort(y_sorted[:2, 0])]
    bottom = y_sorted[2:][np.argsort(y_sorted[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def _interpolate_edge(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return ((1.0 - alpha) * start + alpha * end).astype(np.float32)


def clean_court_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def line_intersection(
    line1: tuple[np.ndarray, np.ndarray],
    line2: tuple[np.ndarray, np.ndarray],
) -> np.ndarray | None:
    point1, direction1 = line1
    point2, direction2 = line2
    cross = direction1[0] * direction2[1] - direction1[1] * direction2[0]
    if abs(float(cross)) < 1e-6:
        return None

    delta = point2 - point1
    scale = (delta[0] * direction2[1] - delta[1] * direction2[0]) / cross
    return point1 + scale * direction1


def fit_edge_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 20:
        return None

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    point = np.array([x0.item(), y0.item()], dtype=np.float32)
    direction = np.array([vx.item(), vy.item()], dtype=np.float32)
    return point, direction


def refine_corners_with_edges(
    contour_points: np.ndarray,
    rough_corners: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    rough_corners = order_court_corners(rough_corners)
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
        line = fit_edge_line(np.asarray(points, dtype=np.float32))
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

    return order_court_corners(np.asarray(refined_corners, dtype=np.float32))


def extract_court_corners(mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    cleaned_mask = clean_court_mask(mask)
    contours, _ = cv2.findContours(
        cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None, cleaned_mask

    contour = max(contours, key=cv2.contourArea)
    min_area = cleaned_mask.shape[0] * cleaned_mask.shape[1] * 0.05
    if cv2.contourArea(contour) < min_area:
        return None, cleaned_mask

    perimeter = cv2.arcLength(contour, True)
    candidates = []
    for epsilon_scale in np.linspace(0.008, 0.06, 16):
        approx = cv2.approxPolyDP(contour, epsilon_scale * perimeter, True)
        if len(approx) == 4:
            ordered = order_court_corners(approx.reshape(-1, 2))
            area = cv2.contourArea(ordered.astype(np.float32))
            candidates.append((area, ordered))

    if not candidates:
        return None, cleaned_mask

    rough_corners = max(candidates, key=lambda item: item[0])[1]
    refined_corners = refine_corners_with_edges(
        contour.reshape(-1, 2),
        rough_corners,
        cleaned_mask.shape,
    )
    return refined_corners, cleaned_mask


def smooth_corners(
    previous_corners: np.ndarray | None,
    current_corners: np.ndarray | None,
    alpha: float = 0.25,
) -> np.ndarray | None:
    if previous_corners is None or current_corners is None:
        return current_corners
    return ((1.0 - alpha) * previous_corners + alpha * current_corners).astype(
        np.float32
    )


def detect_court_corners(
    frame: np.ndarray,
    model,
    device,
    image_size: int,
    threshold: float = 0.5,
) -> tuple[np.ndarray | None, np.ndarray]:
    tensor = preprocess_court_frame(frame, device, image_size)
    with torch.inference_mode():
        pred = model(tensor)
        pred = torch.sigmoid(pred).squeeze().detach().cpu().numpy().astype(np.float32)

    pred = cv2.resize(
        pred, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    pred = (pred > threshold).astype(np.uint8) * 255
    return extract_court_corners(pred)


class CourtDetector:
    def __init__(
        self,
        model_path: str,
        encoder_name: str = "mobilenet_v2",
        image_size: int = 192,
        threshold: float = 0.5,
        device=None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.image_size = image_size
        self.threshold = threshold
        self.model = build_court_model(self.device, encoder_name, model_path)

    def predict(self, frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        return detect_court_corners(
            frame=frame,
            model=self.model,
            device=self.device,
            image_size=self.image_size,
            threshold=self.threshold,
        )


def build_near_court_polygon(
    court_corners: np.ndarray | list[list[float]],
    depth_start: float = 0.52,
) -> np.ndarray:
    ordered = order_court_corners(court_corners)
    top_left, top_right, bottom_right, bottom_left = ordered

    near_left = _interpolate_edge(top_left, bottom_left, depth_start)
    near_right = _interpolate_edge(top_right, bottom_right, depth_start)
    return np.array(
        [near_left, near_right, bottom_right, bottom_left],
        dtype=np.float32,
    )


def _clip_bbox(
    bbox: tuple[float, float, float, float], frame_shape: tuple[int, int, int]
) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(round(x1), 0, frame_w - 1))
    y1 = int(np.clip(round(y1), 0, frame_h - 1))
    x2 = int(np.clip(round(x2), x1 + 1, frame_w))
    y2 = int(np.clip(round(y2), y1 + 1, frame_h))
    return x1, y1, x2, y2


def normalize_person_detections(
    detections: Iterable[PersonDetection | dict | tuple | list],
    frame_shape: tuple[int, int, int],
) -> list[PersonDetection]:
    normalized: list[PersonDetection] = []
    for detection in detections:
        if isinstance(detection, PersonDetection):
            bbox = _clip_bbox(detection.bbox, frame_shape)
            normalized.append(
                PersonDetection(
                    bbox=bbox,
                    confidence=float(detection.confidence),
                    class_id=int(detection.class_id),
                )
            )
            continue

        if isinstance(detection, dict):
            bbox = detection.get("bbox") or detection.get("xyxy")
            confidence = detection.get("confidence", detection.get("score", 0.0))
            class_id = detection.get("class_id", detection.get("cls", PERSON_CLASS_ID))
        else:
            if len(detection) < 5:
                raise ValueError("Each detection must contain bbox + confidence")
            bbox = detection[:4]
            confidence = detection[4]
            class_id = detection[5] if len(detection) > 5 else PERSON_CLASS_ID

        normalized.append(
            PersonDetection(
                bbox=_clip_bbox(tuple(float(v) for v in bbox), frame_shape),
                confidence=float(confidence),
                class_id=int(class_id),
            )
        )
    return normalized


def score_near_camera_player(
    detection: PersonDetection,
    frame_shape: tuple[int, int, int],
    court_corners: np.ndarray | list[list[float]] | None = None,
    near_court_polygon: np.ndarray | None = None,
) -> float:
    frame_h, frame_w = frame_shape[:2]
    if detection.class_id != PERSON_CLASS_ID or detection.area <= 0:
        return float("-inf")

    foot_x, foot_y = detection.foot_point
    frame_area = float(frame_h * frame_w)
    area_score = np.clip(detection.area / max(frame_area * 0.18, 1.0), 0.0, 1.0)
    foot_y_score = np.clip(foot_y / max(frame_h, 1), 0.0, 1.0)

    if near_court_polygon is None and court_corners is not None:
        near_court_polygon = build_near_court_polygon(court_corners)

    in_near_court = False
    court_center_x_score = 0.5
    if near_court_polygon is not None:
        in_near_court = cv2.pointPolygonTest(
            near_court_polygon.astype(np.float32),
            (float(foot_x), float(foot_y)),
            False,
        ) >= 0
        court_center_x = float(np.mean(near_court_polygon[:, 0]))
        half_width = max(float(np.ptp(near_court_polygon[:, 0])) * 0.5, 1.0)
        court_center_x_score = np.clip(
            1.0 - abs(float(foot_x) - court_center_x) / half_width,
            0.0,
            1.0,
        )

    return float(
        0.40 * area_score
        + 0.30 * foot_y_score
        + 0.15 * float(detection.confidence)
        + 0.15 * court_center_x_score
        + (0.75 if in_near_court else -0.75)
    )


def select_near_camera_player(
    detections: Iterable[PersonDetection | dict | tuple | list],
    frame_shape: tuple[int, int, int],
    court_corners: np.ndarray | list[list[float]] | None = None,
    min_confidence: float = 0.25,
    min_area_ratio: float = 0.01,
) -> PersonDetection | None:
    candidates = normalize_person_detections(detections, frame_shape)
    if not candidates:
        return None

    frame_area = float(frame_shape[0] * frame_shape[1])
    near_court_polygon = (
        build_near_court_polygon(court_corners) if court_corners is not None else None
    )

    best_detection = None
    best_score = float("-inf")
    for detection in candidates:
        if detection.class_id != PERSON_CLASS_ID:
            continue
        if detection.confidence < min_confidence:
            continue
        if detection.area < frame_area * min_area_ratio:
            continue

        score = score_near_camera_player(
            detection=detection,
            frame_shape=frame_shape,
            court_corners=court_corners,
            near_court_polygon=near_court_polygon,
        )
        if score > best_score:
            best_score = score
            best_detection = detection

    return best_detection


def expand_bbox(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
    x_padding_ratio: float = 0.18,
    y_padding_ratio: float = 0.12,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    expanded = (
        x1 - width * x_padding_ratio,
        y1 - height * y_padding_ratio,
        x2 + width * x_padding_ratio,
        y2 + height * y_padding_ratio,
    )
    return _clip_bbox(expanded, frame_shape)


def crop_pose_input(
    frame: np.ndarray,
    detection: PersonDetection,
    x_padding_ratio: float = 0.18,
    y_padding_ratio: float = 0.12,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    crop_box = expand_bbox(
        detection.bbox,
        frame.shape,
        x_padding_ratio=x_padding_ratio,
        y_padding_ratio=y_padding_ratio,
    )
    x1, y1, x2, y2 = crop_box
    return frame[y1:y2, x1:x2].copy(), crop_box


def build_pose_target_from_yolo(result) -> list[PersonDetection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(np.int32)
    return [
        PersonDetection(
            bbox=tuple(int(v) for v in box),
            confidence=float(confidence),
            class_id=int(class_id),
        )
        for box, confidence, class_id in zip(xyxy, confidences, class_ids)
    ]
