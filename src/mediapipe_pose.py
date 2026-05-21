from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import requests
from tqdm import tqdm


@dataclass
class PoseAnalysis:
    landmarks_px: List[Tuple[int, int]]
    visibility: List[float]
    presence: List[float]
    quality: float
    is_body: bool


class MediaPipePoseAnalyzer:
    """
    MediaPipe Tasks Pose Landmarker wrapper (33 landmarks).
    """

    def __init__(
        self,
        model_path: Path = Path("data/mediapipe/pose_landmarker.task"),
        min_pose_detection_confidence: float = 0.3,
    ):
        self._ensure_model(model_path)
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            min_pose_detection_confidence=float(min_pose_detection_confidence),
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def close(self) -> None:
        if getattr(self, "_landmarker", None) is not None:
            try:
                self._landmarker.close()
            finally:
                self._landmarker = None

    def __enter__(self) -> "MediaPipePoseAnalyzer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _ensure_model(model_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if model_path.exists() and model_path.stat().st_size > 500_000:
            return
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with open(model_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=model_path.name) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    def analyze(self, frame_bgr) -> Optional[PoseAnalysis]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect(mp_img)
        if not res.pose_landmarks:
            return None
        landmarks = res.pose_landmarks[0]
        landmarks_px = [(int(p.x * w), int(p.y * h)) for p in landmarks]
        visibility = [float(getattr(p, "visibility", 0.0) or 0.0) for p in landmarks]
        presence = [float(getattr(p, "presence", 0.0) or 0.0) for p in landmarks]

        # Quality gating:
        # - Must work for "upper-body only" (laptop camera).
        # - Must not treat a close-up face as a full body.
        # Use confidence (visibility/presence) + size/geometry heuristics.
        def conf(i: int) -> float:
            if i >= len(visibility) or i >= len(presence):
                return 0.0
            return float(min(visibility[i], presence[i]))

        c_sh = min(conf(11), conf(12))  # shoulders
        c_el = min(conf(13), conf(14))  # elbows
        c_wr = min(conf(15), conf(16))  # wrists
        c_hip = min(conf(23), conf(24))  # hips

        shoulder_span = 0.0
        if all(i < len(landmarks_px) for i in (11, 12)):
            x1, y1 = landmarks_px[11]
            x2, y2 = landmarks_px[12]
            shoulder_span = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5) / max(w, 1)

        torso_span = 0.0
        if all(i < len(landmarks_px) for i in (11, 12, 23, 24)):
            sh_y = (landmarks_px[11][1] + landmarks_px[12][1]) / 2.0
            hip_y = (landmarks_px[23][1] + landmarks_px[24][1]) / 2.0
            torso_span = max(0.0, hip_y - sh_y) / max(h, 1)

        upper_span = 0.0
        if all(i < len(landmarks_px) for i in (11, 12, 13, 14)):
            sh_y = (landmarks_px[11][1] + landmarks_px[12][1]) / 2.0
            el_y = (landmarks_px[13][1] + landmarks_px[14][1]) / 2.0
            upper_span = max(0.0, el_y - sh_y) / max(h, 1)

        base_conf = max(min(c_sh, c_hip), min(c_sh, c_el), min(c_sh, c_wr))
        size_term = max(
            min(1.0, torso_span / 0.20) if torso_span > 0 else 0.0,
            min(1.0, upper_span / 0.18) if upper_span > 0 else 0.0,
            min(1.0, shoulder_span / 0.20) if shoulder_span > 0 else 0.0,
        )
        quality = float(0.6 * base_conf + 0.4 * size_term)

        full_body = bool(c_sh >= 0.5 and c_hip >= 0.5 and torso_span >= 0.10)
        upper_body = bool(c_sh >= 0.6 and c_el >= 0.5 and upper_span >= 0.08 and shoulder_span >= 0.08)
        is_body = bool(full_body or upper_body)

        return PoseAnalysis(
            landmarks_px=landmarks_px,
            visibility=visibility,
            presence=presence,
            quality=quality,
            is_body=is_body,
        )
