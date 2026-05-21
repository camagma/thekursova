from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class PainComponents:
    brow_down: float
    eye_squint: float
    nose_sneer: float
    mouth_frown: float
    jaw_open: float


@dataclass
class PainResult:
    pain_percent: float
    components: PainComponents
    occluded_regions: List[str]


@dataclass
class BaselineFeatures:
    brow_eye: float
    eye_ear: float
    nose_lip: float
    mouth_frown: float
    mouth_open: float


def _bs(blendshapes: Dict[str, float], name: str) -> float:
    return float(blendshapes.get(name, 0.0))


def _mean(*vals: float) -> float:
    arr = [v for v in vals if v is not None]
    return float(sum(arr) / max(len(arr), 1))


def _bbox_from_indices(
    landmarks_px: List[Tuple[int, int]],
    indices: List[int],
    w: int,
    h: int,
    margin: float = 0.20,
) -> Tuple[int, int, int, int]:
    xs = [landmarks_px[i][0] for i in indices if i < len(landmarks_px)]
    ys = [landmarks_px[i][1] for i in indices if i < len(landmarks_px)]
    if not xs or not ys:
        return 0, 0, 0, 0
    x1, x2 = max(0, min(xs)), min(w - 1, max(xs))
    y1, y2 = max(0, min(ys)), min(h - 1, max(ys))
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    return max(0, x1 - dx), max(0, y1 - dy), min(w - 1, x2 + dx), min(h - 1, y2 + dy)


def _skin_ratio_ycrcb(region_bgr: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    skin_pixels = float(cv2.countNonZero(skin_mask))
    total = region_bgr.shape[0] * region_bgr.shape[1]
    return skin_pixels / max(total, 1)


def _dark_ratio(region_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    dark = (v < 60).astype(np.uint8)
    return float(dark.mean())


def _region_occluded(region_bgr: np.ndarray, min_skin: float, min_dark: float) -> bool:
    if region_bgr.size == 0:
        return False
    skin = _skin_ratio_ycrcb(region_bgr)
    dark = _dark_ratio(region_bgr)
    # If it's very dark and not skin -> likely hair/occlusion.
    return (skin < min_skin) and (dark > min_dark)


def compute_pain_from_blendshapes(
    frame_bgr: np.ndarray,
    landmarks_px: List[Tuple[int, int]],
    blendshapes: Dict[str, float],
) -> PainResult:
    """
    Rule-based pain estimation from MediaPipe Tasks blendshapes.
    Also detects hair occlusion over specific regions and ignores those components.
    """
    h, w = frame_bgr.shape[:2]

    # FaceMesh landmark index sets (approx) for regions
    left_brow = [70, 63, 105, 66, 107]
    right_brow = [336, 296, 334, 293, 300]
    nose = [1, 2, 98, 327, 168, 197, 5]
    mouth = [61, 291, 0, 17, 13, 14, 78, 308]

    bx1, by1, bx2, by2 = _bbox_from_indices(landmarks_px, left_brow + right_brow, w, h, margin=0.30)
    nx1, ny1, nx2, ny2 = _bbox_from_indices(landmarks_px, nose, w, h, margin=0.30)
    mx1, my1, mx2, my2 = _bbox_from_indices(landmarks_px, mouth, w, h, margin=0.25)

    brow_region = frame_bgr[by1:by2, bx1:bx2]
    nose_region = frame_bgr[ny1:ny2, nx1:nx2]
    mouth_region = frame_bgr[my1:my2, mx1:mx2]

    occluded = []
    brow_occ = _region_occluded(brow_region, min_skin=0.18, min_dark=0.55)
    nose_occ = _region_occluded(nose_region, min_skin=0.20, min_dark=0.60)
    mouth_occ = _region_occluded(mouth_region, min_skin=0.18, min_dark=0.60)
    if brow_occ:
        occluded.append("brows")
    if nose_occ:
        occluded.append("nose")
    if mouth_occ:
        occluded.append("mouth")

    brow_down = _mean(_bs(blendshapes, "browDownLeft"), _bs(blendshapes, "browDownRight"))
    eye_squint = _mean(_bs(blendshapes, "eyeSquintLeft"), _bs(blendshapes, "eyeSquintRight"))
    nose_sneer = _mean(_bs(blendshapes, "noseSneerLeft"), _bs(blendshapes, "noseSneerRight"))
    mouth_frown = _mean(_bs(blendshapes, "mouthFrownLeft"), _bs(blendshapes, "mouthFrownRight"))
    jaw_open = _bs(blendshapes, "jawOpen")

    if brow_occ:
        brow_down = 0.0
    if nose_occ:
        nose_sneer = 0.0
    if mouth_occ:
        mouth_frown = 0.0
        jaw_open = 0.0

    components = PainComponents(
        brow_down=brow_down,
        eye_squint=eye_squint,
        nose_sneer=nose_sneer,
        mouth_frown=mouth_frown,
        jaw_open=jaw_open,
    )

    # Weighted sum; tuned for stability (not medical-grade).
    raw = (
        1.20 * brow_down
        + 1.35 * eye_squint
        + 0.90 * nose_sneer
        + 0.70 * mouth_frown
        + 0.25 * jaw_open
    )

    # Map to 0..1 with logistic; bias roughly where neutral faces land.
    k = 4.0
    bias = 0.55
    prob = 1.0 / (1.0 + float(np.exp(-k * (raw - bias))))
    pain_percent = float(np.clip(prob * 100.0, 0.0, 100.0))

    return PainResult(pain_percent=pain_percent, components=components, occluded_regions=occluded)


def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _safe_div(a: float, b: float, eps: float = 1e-6) -> float:
    return float(a / (b if abs(b) > eps else eps))


def compute_baseline_features(landmarks_px: List[Tuple[int, int]]) -> BaselineFeatures:
    left_eye_outer, right_eye_outer = 33, 263
    left_eye = [33, 160, 158, 133, 153, 144]
    right_eye = [362, 385, 387, 263, 373, 380]
    left_brow = [70, 63, 105, 66, 107]
    right_brow = [336, 296, 334, 293, 300]
    nose_tip = 1
    upper_lip = 13
    lower_lip = 14
    mouth_left, mouth_right = 61, 291

    lx, ly = landmarks_px[left_eye_outer]
    rx, ry = landmarks_px[right_eye_outer]
    cx, cy = (lx + rx) / 2.0, (ly + ry) / 2.0
    ang = float(np.arctan2(ry - ly, rx - lx))
    ca, sa = float(np.cos(-ang)), float(np.sin(-ang))

    def _pt(i: int) -> Tuple[float, float]:
        x, y = landmarks_px[i]
        x -= cx
        y -= cy
        xr = ca * x - sa * y
        yr = sa * x + ca * y
        return (xr + cx, yr + cy)

    scale = _dist(_pt(left_eye_outer), _pt(right_eye_outer))
    scale = max(scale, 1.0)

    brow_y = _mean(*(_pt(i)[1] for i in (left_brow + right_brow)))
    eye_top_y = _mean(_pt(159)[1], _pt(386)[1])
    brow_eye = _safe_div((eye_top_y - brow_y), scale) 

    def ear(pts: List[int]) -> float:
        p1, p2, p3, p4, p5, p6 = pts
        width = _dist(_pt(p1), _pt(p4))
        height = _dist(_pt(p2), _pt(p6)) + _dist(_pt(p3), _pt(p5))
        return _safe_div(height, 2.0 * width)

    eye_ear = _mean(ear(left_eye), ear(right_eye))

    nose_lip = _safe_div(_dist(_pt(nose_tip), _pt(upper_lip)), scale)

    mouth_center_y = _mean(_pt(upper_lip)[1], _pt(lower_lip)[1])
    mouth_corner_y = _mean(_pt(mouth_left)[1], _pt(mouth_right)[1])
    mouth_frown = _safe_div((mouth_corner_y - mouth_center_y), scale)

    mouth_open = _safe_div(_dist(_pt(upper_lip), _pt(lower_lip)), scale)

    return BaselineFeatures(
        brow_eye=float(brow_eye),
        eye_ear=float(eye_ear),
        nose_lip=float(nose_lip),
        mouth_frown=float(mouth_frown),
        mouth_open=float(mouth_open),
    )


def diff_features(baseline: BaselineFeatures, current: BaselineFeatures) -> dict:
    """
    Signed deltas (current - baseline).
    Interpretation:
    - brow_eye: negative => brow lowered (pain-like)
    - eye_ear: negative => eyes more closed (pain-like)
    - nose_lip: negative => upper lip closer to nose (pain-like)
    - mouth_frown: positive => corners lower than center (pain-like)
    - mouth_open: positive => mouth more open
    """
    return {
        "brow_eye": float(current.brow_eye - baseline.brow_eye),
        "eye_ear": float(current.eye_ear - baseline.eye_ear),
        "nose_lip": float(current.nose_lip - baseline.nose_lip),
        "mouth_frown": float(current.mouth_frown - baseline.mouth_frown),
        "mouth_open": float(current.mouth_open - baseline.mouth_open),
    }


def compute_pain_from_landmarks(
    frame_bgr: np.ndarray,
    landmarks_px: List[Tuple[int, int]],
    baseline: BaselineFeatures,
) -> PainResult:
    """
    Rule-based pain estimation from MediaPipe landmarks geometry.
    - Brow lowering: brow-to-eye vertical relation vs baseline
    - Eye squint: eye aspect ratio (EAR) decrease vs baseline
    - Nose/upper lip: nose-to-upper-lip distance decrease vs baseline
    - Mouth frown: corners lower than center vs baseline
    - Jaw open: mouth opening increase vs baseline
    Also ignores regions likely occluded by hair (dark + low-skin) separately for brows/nose/mouth.
    """
    h, w = frame_bgr.shape[:2]

    left_brow = [70, 63, 105, 66, 107]
    right_brow = [336, 296, 334, 293, 300]
    nose = [1, 2, 98, 327, 168, 197, 5]
    mouth = [61, 291, 0, 17, 13, 14, 78, 308]

    bx1, by1, bx2, by2 = _bbox_from_indices(landmarks_px, left_brow + right_brow, w, h, margin=0.30)
    nx1, ny1, nx2, ny2 = _bbox_from_indices(landmarks_px, nose, w, h, margin=0.30)
    mx1, my1, mx2, my2 = _bbox_from_indices(landmarks_px, mouth, w, h, margin=0.25)

    brow_region = frame_bgr[by1:by2, bx1:bx2]
    nose_region = frame_bgr[ny1:ny2, nx1:nx2]
    mouth_region = frame_bgr[my1:my2, mx1:mx2]

    occluded = []
    brow_occ = _region_occluded(brow_region, min_skin=0.18, min_dark=0.55)
    nose_occ = _region_occluded(nose_region, min_skin=0.20, min_dark=0.60)
    mouth_occ = _region_occluded(mouth_region, min_skin=0.18, min_dark=0.60)
    if brow_occ:
        occluded.append("brows")
    if nose_occ:
        occluded.append("nose")
    if mouth_occ:
        occluded.append("mouth")

    curr = compute_baseline_features(landmarks_px)
    brow_down = float(np.clip((baseline.brow_eye - curr.brow_eye) * 5.0, 0.0, 1.0))
    eye_squint = float(np.clip((baseline.eye_ear - curr.eye_ear) * 6.0, 0.0, 1.0))
    nose_sneer = float(np.clip((baseline.nose_lip - curr.nose_lip) * 6.0, 0.0, 1.0))
    mouth_frown = float(np.clip((curr.mouth_frown - baseline.mouth_frown) * 6.0, 0.0, 1.0))
    jaw_open = float(np.clip((curr.mouth_open - baseline.mouth_open) * 5.0, 0.0, 1.0))

    if brow_occ:
        brow_down = 0.0
    if nose_occ:
        nose_sneer = 0.0
    if mouth_occ:
        mouth_frown = 0.0
        jaw_open = 0.0

    components = PainComponents(
        brow_down=brow_down,
        eye_squint=eye_squint,
        nose_sneer=nose_sneer,
        mouth_frown=mouth_frown,
        jaw_open=jaw_open,
    )

    raw = (
        1.20 * brow_down
        + 1.40 * eye_squint
        + 0.80 * nose_sneer
        + 0.70 * mouth_frown
        + 0.20 * jaw_open
    )
    k = 2.0
    bias = 1.6
    prob = 1.0 / (1.0 + float(np.exp(-k * (raw - bias))))
    pain_percent = float(np.clip(prob * 100.0, 0.0, 100.0))

    return PainResult(pain_percent=pain_percent, components=components, occluded_regions=occluded)
