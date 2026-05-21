from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


def face_bbox(
    landmarks_px: List[Tuple[int, int]],
    w: int,
    h: int,
    margin: float = 0.10,
) -> Optional[Tuple[int, int, int, int]]:
    if not landmarks_px:
        return None
    xs = [p[0] for p in landmarks_px]
    ys = [p[1] for p in landmarks_px]
    x1, x2 = max(0, min(xs)), min(w - 1, max(xs))
    y1, y2 = max(0, min(ys)), min(h - 1, max(ys))
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w - 1, x2 + dx)
    y2 = min(h - 1, y2 + dy)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def anchor_point(
    landmarks_px: List[Tuple[int, int]],
    indices: List[int],
) -> Optional[Tuple[int, int]]:
    pts = [landmarks_px[i] for i in indices if i < len(landmarks_px)]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


def put_text_bg(
    image_bgr: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.6,
    thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.55,
) -> Tuple[int, int]:
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 4
    x1 = max(0, x - pad)
    y1 = max(0, y - th - pad)
    x2 = min(image_bgr.shape[1] - 1, x + tw + pad)
    y2 = min(image_bgr.shape[0] - 1, y + baseline + pad)

    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0, image_bgr)
    cv2.putText(image_bgr, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return x2, y2


def draw_face_hud(
    frame_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    lines: List[Tuple[str, Tuple[int, int, int]]],
) -> None:
    x1, y1, x2, y2 = bbox
    # Draw bbox
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)
    # Place HUD inside top-left of face (or slightly above if space)
    hud_x = x1 + 6
    hud_y = y1 - 10
    if hud_y < 30:
        hud_y = y1 + 24
    y = hud_y
    for text, color in lines:
        put_text_bg(frame_bgr, text, (hud_x, y), font_scale=0.65, thickness=2, text_color=color)
        y += 26


def draw_skeleton(
    frame_bgr: np.ndarray,
    landmarks_px: List[Tuple[int, int]],
    connections: List[Tuple[int, int]],
    point_color: Tuple[int, int, int] = (0, 0, 255),
    line_color: Tuple[int, int, int] = (255, 255, 255),
    point_radius: int = 4,
    line_thickness: int = 2,
) -> None:
    h, w = frame_bgr.shape[:2]
    for a, b in connections:
        if a >= len(landmarks_px) or b >= len(landmarks_px):
            continue
        x1, y1 = landmarks_px[a]
        x2, y2 = landmarks_px[b]
        if x1 <= 0 and y1 <= 0:
            continue
        if x2 <= 0 and y2 <= 0:
            continue
        cv2.line(frame_bgr, (x1, y1), (x2, y2), line_color, line_thickness, cv2.LINE_AA)
    for x, y in landmarks_px:
        if x <= 0 and y <= 0:
            continue
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        cv2.circle(frame_bgr, (x, y), point_radius, point_color, -1, cv2.LINE_AA)


def draw_polyline(
    frame_bgr: np.ndarray,
    landmarks_px: List[Tuple[int, int]],
    indices: List[int],
    closed: bool,
    line_color: Tuple[int, int, int],
    thickness: int,
) -> None:
    pts = [(landmarks_px[i][0], landmarks_px[i][1]) for i in indices if i < len(landmarks_px)]
    if len(pts) < 2:
        return
    for i in range(len(pts) - 1):
        cv2.line(frame_bgr, pts[i], pts[i + 1], line_color, thickness, cv2.LINE_AA)
    if closed:
        cv2.line(frame_bgr, pts[-1], pts[0], line_color, thickness, cv2.LINE_AA)


def draw_face_pain_skeleton(
    frame_bgr: np.ndarray,
    landmarks_px: List[Tuple[int, int]],
    point_color: Tuple[int, int, int] = (0, 0, 255),
    line_color: Tuple[int, int, int] = (255, 255, 255),
    point_radius: int = 2,
    line_thickness: int = 1,
) -> None:
    """
    Draw a sparse face skeleton focused on pain-related regions:
    eyebrows, eyes, nose ridge, mouth outline. Styled like pose overlays.
    """
    # These are simplified contours (not full tessellation) to stay readable.
    left_eye = [33, 160, 158, 133, 153, 144, 33]
    right_eye = [263, 387, 385, 362, 380, 373, 263]
    left_brow = [70, 63, 105, 66, 107]
    right_brow = [336, 296, 334, 293, 300]
    nose_ridge = [168, 6, 197, 195, 5, 4, 1, 2]
    mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

    draw_polyline(frame_bgr, landmarks_px, left_eye, closed=False, line_color=line_color, thickness=line_thickness)
    draw_polyline(frame_bgr, landmarks_px, right_eye, closed=False, line_color=line_color, thickness=line_thickness)
    draw_polyline(frame_bgr, landmarks_px, left_brow, closed=False, line_color=line_color, thickness=line_thickness)
    draw_polyline(frame_bgr, landmarks_px, right_brow, closed=False, line_color=line_color, thickness=line_thickness)
    draw_polyline(frame_bgr, landmarks_px, nose_ridge, closed=False, line_color=line_color, thickness=line_thickness)
    draw_polyline(frame_bgr, landmarks_px, mouth, closed=False, line_color=line_color, thickness=line_thickness)

    # Points only for indices used in polylines to reduce clutter
    used = set(left_eye + right_eye + left_brow + right_brow + nose_ridge + mouth)
    h, w = frame_bgr.shape[:2]
    for i in used:
        if i >= len(landmarks_px):
            continue
        x, y = landmarks_px[i]
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        cv2.circle(frame_bgr, (x, y), point_radius, point_color, -1, cv2.LINE_AA)


# MediaPipe Pose connections (subset) to look like typical pose skeleton overlays.
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    # Face
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (3, 7),
    (6, 8),
    (9, 10),
    # Upper body
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    # Hands (simple)
    (15, 17),
    (15, 19),
    (15, 21),
    (16, 18),
    (16, 20),
    (16, 22),
    # Lower body
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 29),
    (29, 31),
    (28, 30),
    (30, 32),
]
