import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import requests
from tqdm import tqdm

from src.face_detector import crop_to_face


# URLs exposed by the Borealis (Dataverse) API
DATA_FILES = [
    {
        "name": "SynPain_Part1",
        "zip_name": "SynPain_Part1.zip",
        "url": "https://borealisdata.ca/api/access/datafile/955581",
    },
    {
        "name": "SynPain_Part2",
        "zip_name": "SynPain_Part2.zip",
        "url": "https://borealisdata.ca/api/access/datafile/955580",
    },
]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


def _extract(zip_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def ensure_synpain_data(data_dir: Path = Path("data")) -> None:
    """Download and extract both SynPain parts if they are missing."""
    for item in DATA_FILES:
        part_dir = data_dir / item["name"]
        if part_dir.exists():
            continue
        zip_path = data_dir / item["zip_name"]
        if not zip_path.exists():
            print(f"Downloading {zip_path.name} ...")
            _download(item["url"], zip_path)
        print(f"Extracting {zip_path.name} ...")
        _extract(zip_path, data_dir)


def _parse_label_from_name(path: Path) -> int:
    # filenames follow: [ID]_[expression]_[gender]_[age].jpg
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected filename format: {path.name}")
    return 1 if parts[1].lower() == "pain" else 0


def collect_image_paths(data_dir: Path = Path("data")) -> List[Tuple[Path, int]]:
    """Return list of (path, label) across all SynPain parts."""
    ensure_synpain_data(data_dir)
    image_dirs = sorted(data_dir.glob("SynPain_Part*/Images_*"))
    items: List[Tuple[Path, int]] = []
    for img_dir in image_dirs:
        for img_path in img_dir.glob("*.jpg"):
            label = _parse_label_from_name(img_path)
            items.append((img_path, label))
    if not items:
        raise RuntimeError(f"No images found under {data_dir}")
    return items


class SynPainDataset:
    """Lightweight dataset wrapper using OpenCV for image loading."""

    def __init__(self, samples: Iterable[Tuple[Path, int]], transform=None, face_net=None, face_conf: float = 0.5):
        self.samples: List[Tuple[Path, int]] = list(samples)
        self.transform = transform
        self.face_net = face_net
        self.face_conf = face_conf

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        if self.face_net is not None:
            img = crop_to_face(img, self.face_net, conf_threshold=self.face_conf)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label
