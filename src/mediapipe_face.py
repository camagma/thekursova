from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import requests
from tqdm import tqdm


@dataclass
class FaceAnalysis:
    landmarks_px: List[Tuple[int, int]]
    blendshapes: Dict[str, float]


class MediaPipeFaceAnalyzer:
    """
    MediaPipe Tasks Face Landmarker wrapper that returns:
    - 468 face mesh landmarks in pixel coordinates
    - blendshape scores (browDown, eyeSquint, noseSneer, etc.)
    """

    def __init__(
        self,
        model_path: Path = Path("data/mediapipe/face_landmarker.task"),
        use_blendshapes: bool = False,
    ):
        self._ensure_model(model_path)
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        try:
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=use_blendshapes,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except ValueError as e:
            # Some environments/models don't support BLENDSHAPES; fall back to landmarks-only.
            if "BLENDSHAPES" in str(e).upper():
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                )
                self._landmarker = vision.FaceLandmarker.create_from_options(options)
            else:
                raise
        self._vision = vision
        self._use_blendshapes = use_blendshapes

    @staticmethod
    def _ensure_model(model_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # If an older/smaller asset exists, refresh to the official model bundle.
        if model_path.exists() and model_path.stat().st_size > 2_000_000:
            return
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with open(model_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=model_path.name) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    def close(self) -> None:
        # Avoid mediapipe tasks destructor issues by closing explicitly.
        if getattr(self, "_landmarker", None) is not None:
            try:
                self._landmarker.close()
            finally:
                self._landmarker = None

    def __enter__(self) -> "MediaPipeFaceAnalyzer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def analyze(self, frame_bgr) -> Optional[FaceAnalysis]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect(mp_img)
        if not res.face_landmarks:
            return None

        landmarks = res.face_landmarks[0]
        landmarks_px = [(int(p.x * w), int(p.y * h)) for p in landmarks]

        blendshapes: Dict[str, float] = {}
        if res.face_blendshapes:
            for c in res.face_blendshapes[0]:
                blendshapes[c.category_name] = float(c.score)

        return FaceAnalysis(landmarks_px=landmarks_px, blendshapes=blendshapes)
