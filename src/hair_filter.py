from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


class HairOcclusionChecker:
    """
    Heuristic hair occlusion detector.
    Uses face landmarks to estimate top/forehead region; if skin coverage is very low there,
    assume hair is occluding brows/forehead and skip inference.
    """

    def __init__(self, skin_threshold: float = 0.25, top_ratio: float = 0.25):
        self.skin_threshold = skin_threshold
        self.top_ratio = top_ratio

    @staticmethod
    def _skin_ratio(image_bgr: np.ndarray) -> float:
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        skin_pixels = float(cv2.countNonZero(skin_mask))
        total = image_bgr.shape[0] * image_bgr.shape[1]
        return skin_pixels / max(total, 1)

    def is_occluded(self, image_bgr: np.ndarray, landmarks: Sequence[Tuple[float, float, Optional[float]]]) -> bool:
        if not landmarks:
            return False
        h, w = image_bgr.shape[:2]
        xs = [int(p[0] * w) for p in landmarks]
        ys = [int(p[1] * h) for p in landmarks]
        x_min, x_max = max(0, min(xs)), min(w - 1, max(xs))
        y_min, y_max = max(0, min(ys)), min(h - 1, max(ys))
        face_height = max(1, y_max - y_min)
        top_h = int(face_height * self.top_ratio)
        top_y2 = max(y_min + top_h, y_min + 1)
        top_region = image_bgr[y_min:top_y2, x_min:x_max]
        if top_region.size == 0:
            return False
        skin_ratio = self._skin_ratio(top_region)
        return skin_ratio < self.skin_threshold
