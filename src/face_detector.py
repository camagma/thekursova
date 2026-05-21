import hashlib
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import requests
from tqdm import tqdm


FACE_PROTO_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
]
# Multiple mirrors for the same face detector weights; try in order.
FACE_MODEL_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel",
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel",
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
    "https://huggingface.co/Durraiya/res10_300x300_ssd_iter_140000_fp16.caffemodel/resolve/main/res10_300x300_ssd_iter_140000_fp16.caffemodel",
]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def _download_first_available(urls, dest: Path) -> None:
    last_err = None
    for url in urls:
        try:
            _download(url, dest)
            return
        except Exception as e:  # pragma: no cover - network dependent
            last_err = e
            continue
    raise RuntimeError(f"Failed to download {dest.name} from all mirrors: {urls}") from last_err


def load_face_detector(det_dir: Path = Path("data/face_detector")) -> cv2.dnn_Net:
    proto = det_dir / "deploy.prototxt"
    model = det_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    _download_first_available(FACE_PROTO_URLS, proto)
    _download_first_available(FACE_MODEL_URLS, model)
    net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
    return net


def detect_faces(
    image_bgr: np.ndarray,
    net: cv2.dnn_Net,
    conf_threshold: float = 0.5,
) -> List[Tuple[int, int, int, int]]:
    """Return list of (x1, y1, x2, y2) boxes."""
    (h, w) = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(image_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < conf_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        boxes.append(
            (
                max(0, x1),
                max(0, y1),
                min(w - 1, x2),
                min(h - 1, y2),
            )
        )
    return boxes


def crop_to_face(
    image_bgr: np.ndarray,
    net: cv2.dnn_Net,
    conf_threshold: float = 0.5,
    margin: float = 0.2,
) -> np.ndarray:
    """Crop largest detected face with a small margin; fallback to original image."""
    boxes = detect_faces(image_bgr, net, conf_threshold)
    if not boxes:
        return image_bgr
    # take largest face
    areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes]
    x1, y1, x2, y2 = boxes[int(np.argmax(areas))]
    w = image_bgr.shape[1]
    h = image_bgr.shape[0]
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w - 1, x2 + dx)
    y2 = min(h - 1, y2 + dy)
    return image_bgr[y1:y2, x1:x2]


def crop_to_face_or_none(
    image_bgr: np.ndarray,
    net: cv2.dnn_Net,
    conf_threshold: float = 0.5,
    margin: float = 0.2,
) -> Tuple[np.ndarray, bool]:
    """Return (crop, found). If no face, returns (None, False)."""
    boxes = detect_faces(image_bgr, net, conf_threshold)
    if not boxes:
        return None, False
    # take largest face
    areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes]
    x1, y1, x2, y2 = boxes[int(np.argmax(areas))]
    w = image_bgr.shape[1]
    h = image_bgr.shape[0]
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w - 1, x2 + dx)
    y2 = min(h - 1, y2 + dy)
    return image_bgr[y1:y2, x1:x2], True
