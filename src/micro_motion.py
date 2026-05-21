from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _l2(a: Tuple[float, float]) -> float:
    return float(np.hypot(a[0], a[1]))


def _safe_div(a: float, b: float, eps: float = 1e-6) -> float:
    return float(a / (b if abs(b) > eps else eps))


@dataclass
class MotionSignals:
    per_region: Dict[str, float]
    overall: float
    per_region_amp: Dict[str, float]
    overall_amp: float


class LandmarkMicroMotion:

    def __init__(self, ema_alpha: float = 0.1):
        self.ema_alpha = ema_alpha
        self._trend: Dict[str, np.ndarray] = {}
        self._prev_residual: Dict[str, np.ndarray] = {}

    def update(
        self,
        regions: Dict[str, List[Tuple[int, int]]],
        scale: float,
        dt: float = 1.0,
    ) -> MotionSignals:
        per_region_speed: Dict[str, float] = {}
        per_region_amp: Dict[str, float] = {}
        dt = float(max(dt, 1e-3))
        for region_name, pts in regions.items():
            if not pts:
                per_region_speed[region_name] = 0.0
                per_region_amp[region_name] = 0.0
                continue
            arr = np.array(pts, dtype=np.float32)
            key = region_name
            if key not in self._trend:
                self._trend[key] = arr.copy()
                self._prev_residual[key] = np.zeros_like(arr)
                per_region_speed[region_name] = 0.0
                per_region_amp[region_name] = 0.0
                continue
            trend = self._trend[key]
            trend = (1.0 - self.ema_alpha) * trend + self.ema_alpha * arr
            self._trend[key] = trend
            residual = arr - trend
            amp = float(np.mean(np.linalg.norm(residual, axis=1)))
            amp_n = float(_safe_div(amp, max(scale, 1.0)))
            per_region_amp[region_name] = amp_n

            prev = self._prev_residual.get(key)
            if prev is None or prev.shape != residual.shape:
                self._prev_residual[key] = residual.copy()
                per_region_speed[region_name] = 0.0
            else:
                delta = residual - prev
                self._prev_residual[key] = residual.copy()
                speed = float(np.mean(np.linalg.norm(delta, axis=1)))
                speed_n = float(_safe_div(speed, max(scale, 1.0) * dt))
                per_region_speed[region_name] = speed_n

        overall_speed = float(np.mean(list(per_region_speed.values()))) if per_region_speed else 0.0
        overall_amp = float(np.mean(list(per_region_amp.values()))) if per_region_amp else 0.0
        return MotionSignals(
            per_region=per_region_speed,
            overall=overall_speed,
            per_region_amp=per_region_amp,
            overall_amp=overall_amp,
        )


def face_regions_from_landmarks(face_lm: List[Tuple[int, int]]) -> Dict[str, List[Tuple[int, int]]]:
    idx = {
        "brows": [70, 63, 105, 66, 107, 336, 296, 334, 293, 300],
        "eyes": [33, 133, 159, 145, 362, 263, 386, 374],
        "nose": [1, 2, 5, 4, 195, 197],
        "mouth": [61, 291, 13, 14, 0, 17, 78, 308],
    }
    out: Dict[str, List[Tuple[int, int]]] = {}
    for name, indices in idx.items():
        out[name] = [face_lm[i] for i in indices if i < len(face_lm)]
    return out


def pose_regions_from_landmarks(pose_lm: List[Tuple[int, int]]) -> Dict[str, List[Tuple[int, int]]]:
   
    idx = {
        "head": [0, 7, 8],  
        "shoulders": [11, 12],
        "elbows": [13, 14],
        "wrists": [15, 16],
        "hips": [23, 24],
        "knees": [25, 26],
        "ankles": [27, 28],
        "torso": [11, 12, 23, 24],
    }
    out: Dict[str, List[Tuple[int, int]]] = {}
    for name, indices in idx.items():
        out[name] = [pose_lm[i] for i in indices if i < len(pose_lm)]
    return out


def face_scale(face_lm: List[Tuple[int, int]]) -> float:
    if len(face_lm) <= 263:
        return 1.0
    x1, y1 = face_lm[33]
    x2, y2 = face_lm[263]
    return float(np.hypot(x2 - x1, y2 - y1))


def pose_scale(pose_lm: List[Tuple[int, int]]) -> float:
    if len(pose_lm) <= 12:
        return 1.0
    x1, y1 = pose_lm[11]
    x2, y2 = pose_lm[12]
    return float(np.hypot(x2 - x1, y2 - y1))
