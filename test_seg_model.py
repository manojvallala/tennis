import argparse
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import time


DEFAULT_MODEL_PATH = "experiments/checkpoints/unet_mobilenet_v2_192.pth"
DEFAULT_VIDEO_PATH = "The Longest Grand Slam Rally Ever_ _ Australian Open 2013.mp4"
DEFAULT_ENCODER_NAME = "mobilenet_v2"
DEFAULT_INFERENCE_SIZE = 192
DEFAULT_INFERENCE_FRAME_STRIDE = 4
DEFAULT_SOURCE_FRAME_STRIDE = 2
CORNER_LABELS = ["TL", "TR", "BR", "BL"]
CORNER_COLORS = [
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 128, 0),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run long-tennis court segmentation inference on a video.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--video-path", default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER_NAME)
    parser.add_argument("--image-size", type=int, default=DEFAULT_INFERENCE_SIZE)
    parser.add_argument("--inference-frame-stride", type=int, default=DEFAULT_INFERENCE_FRAME_STRIDE)
    parser.add_argument("--source-frame-stride", type=int, default=DEFAULT_SOURCE_FRAME_STRIDE)
    return parser.parse_args()


def resize_for_display(image, max_width=1280, max_height=720):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)

    if scale == 1.0:
        return image

    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


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

        x = np.clip(intersection[0], 0, image_shape[1] - 1)
        y = np.clip(intersection[1], 0, image_shape[0] - 1)
        refined_corners.append([x, y])

    return order_corners(np.asarray(refined_corners, dtype=np.float32))


def extract_court_corners(mask):
    cleaned_mask = clean_mask(mask)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
            ordered = order_corners(approx.reshape(-1, 2))
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


def smooth_corners(previous_corners, current_corners, alpha=0.25):
    if previous_corners is None or current_corners is None:
        return current_corners

    return ((1.0 - alpha) * previous_corners + alpha * current_corners).astype(np.float32)


def draw_corners(frame, corners):
    if corners is None:
        return frame

    output = frame.copy()
    points = corners.astype(np.int32)

    cv2.polylines(output, [points], True, (0, 0, 255), 4, cv2.LINE_AA)

    for point, label, color in zip(points, CORNER_LABELS, CORNER_COLORS):
        x, y = int(point[0]), int(point[1])
        cv2.circle(output, (x, y), 10, color, -1, cv2.LINE_AA)
        cv2.putText(
            output,
            label,
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        device=device,
        encoder_name=args.encoder_name,
        model_path=args.model_path,
    )

    cap = cv2.VideoCapture(args.video_path)
    cv2.namedWindow("Long Tennis Court Detection", cv2.WINDOW_NORMAL)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_frame_time = args.source_frame_stride / max(video_fps, 1.0)

    previous_corners = None
    previous_mask = None
    frame_index = 0

    while True:
        frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        run_inference = previous_mask is None or frame_index % args.inference_frame_stride == 0

        if run_inference:
            img = preprocess_frame(frame, device, args.image_size)

            with torch.inference_mode():
                pred = model(img)
                pred = torch.sigmoid(pred).squeeze().detach().cpu().numpy()

            pred = (pred > 0.5).astype(np.uint8) * 255
            pred = cv2.resize(pred, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            detected_corners, cleaned_mask = extract_court_corners(pred)
            if detected_corners is not None:
                previous_corners = smooth_corners(previous_corners, detected_corners)
            previous_mask = cleaned_mask

        cleaned_mask = previous_mask
        corners = previous_corners

        colored_mask = np.zeros_like(frame)
        colored_mask[:, :, 1] = cleaned_mask

        output = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)
        output = draw_corners(output, corners)
        display_frame = resize_for_display(output)

        cv2.imshow("Long Tennis Court Detection", display_frame)

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
