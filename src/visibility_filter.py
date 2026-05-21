from typing import Iterable, Sequence

import cv2
import os
from pathlib import Path
from typing import Optional

import mediapipe
import numpy as np


class FaceVisibilityChecker:
    """
    Estimate landmark visibility; if key regions (eyes/nose/mouth) are not visible, skip inference.
    Uses MediaPipe Face Mesh when available; otherwise falls back to MediaPipe Tasks face landmarker.
    If neither backend is available, the checker degrades gracefully (always returns True).
    """

    def __init__(self, static_image_mode: bool = False, min_detection_confidence: float = 0.5):
        self.backend = None
        self.key_indices = [
            1, 4, 5, 33, 133, 362, 263, 168, 2, 0, 13, 14, 61, 291
        ]
        # Prefer classic solutions if present
        if hasattr(mediapipe, "solutions"):
            from mediapipe import solutions as mp_solutions  # type: ignore

            self.mesh = mp_solutions.face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5,
            )
            self.backend = "solutions"
        elif hasattr(mediapipe, "tasks"):
            # Use Tasks FaceLandmarker as fallback
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = self._ensure_tasks_model()
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            self.backend = "tasks"
        else:  # pragma: no cover - unexpected environment
            print("Warning: mediapipe backend not available; visibility filter disabled.")
            self.backend = None

    def _ensure_tasks_model(self) -> Path:
        """Download face_landmarker.task if missing."""
        # This is the official MediaPipe Tasks model bundle for face landmarker.
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        target = Path("data/mediapipe/face_landmarker.task")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return target
        import requests
        from tqdm import tqdm

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with open(target, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=target.name) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return target

    def is_visible(
        self,
        image_bgr: np.ndarray,
        min_visibility: float = 0.5,
        min_ratio: float = 0.6,
        min_skin: float = 0.2,
    ) -> bool:
        # Fast heuristic: require enough skin pixels
        if self.skin_coverage(image_bgr) < min_skin:
            return False
        if self.backend is None:
            return True

        if self.backend == "solutions":
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            res = self.mesh.process(rgb)
            if not res.multi_face_landmarks:
                return False
            landmarks = res.multi_face_landmarks[0].landmark
            vis_values = []
            for idx in self.key_indices:
                if idx < len(landmarks):
                    vis_values.append(landmarks[idx].visibility)
            if not vis_values:
                return False
            good = [v for v in vis_values if v >= min_visibility]
            return len(good) / len(vis_values) >= min_ratio

        if self.backend == "tasks":
            from mediapipe import Image as MpImage
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_img = MpImage(image_format=mediapipe.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect(mp_img)
            if not res.face_landmarks:
                return False
            landmarks = res.face_landmarks[0]
            # tasks landmarks are x,y,z; no visibility field, so approximate via presence of enough points
            return len(landmarks) >= 150

        return True

    @staticmethod
    def skin_coverage(image_bgr: np.ndarray) -> float:
        """Estimate skin pixel ratio using YCrCb thresholding."""
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        # Simple skin range; may be tuned
        skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        skin_pixels = float(cv2.countNonZero(skin_mask))
        total = image_bgr.shape[0] * image_bgr.shape[1]
        return skin_pixels / max(total, 1)

    def visibility_score(
        self,
        image_bgr: np.ndarray,
        min_visibility: float = 0.5,
        min_ratio: float = 0.6,
    ) -> float:
        """Return a score 0-1 combining skin coverage and landmark presence."""
        skin = self.skin_coverage(image_bgr)
        if self.backend is None:
            return skin
        if self.backend == "solutions":
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            res = self.mesh.process(rgb)
            if not res.multi_face_landmarks:
                return 0.0
            landmarks = res.multi_face_landmarks[0].landmark
            vis_values = []
            for idx in self.key_indices:
                if idx < len(landmarks):
                    vis_values.append(landmarks[idx].visibility)
            if not vis_values:
                return 0.0
            good = [v for v in vis_values if v >= min_visibility]
            ratio = len(good) / len(vis_values)
            return min(1.0, max(0.0, (skin + ratio) / 2.0))
        if self.backend == "tasks":
            from mediapipe import Image as MpImage
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_img = MpImage(image_format=mediapipe.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect(mp_img)
            if not res.face_landmarks:
                return 0.0
            ratio = min(1.0, len(res.face_landmarks[0]) / 468.0)
            return min(1.0, max(0.0, (skin + ratio) / 2.0))
        return skin

    def get_landmarks(self, image_bgr: np.ndarray):
        """Return list of landmarks [(x,y,visibility_or_None), ...] or empty list."""
        if self.backend == "solutions":
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            res = self.mesh.process(rgb)
            if not res.multi_face_landmarks:
                return []
            lm = res.multi_face_landmarks[0].landmark
            return [(p.x, p.y, p.visibility if hasattr(p, "visibility") else None) for p in lm]
        if self.backend == "tasks":
            from mediapipe import Image as MpImage
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_img = MpImage(image_format=mediapipe.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect(mp_img)
            if not res.face_landmarks:
                return []
            return [(p.x, p.y, None) for p in res.face_landmarks[0]]
        return []
