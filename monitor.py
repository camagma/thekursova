import argparse
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from src.face_detector import crop_to_face_or_none, detect_faces, load_face_detector
from src.visibility_filter import FaceVisibilityChecker
from src.hair_filter import HairOcclusionChecker


def load_model(weights_path: Path, device: torch.device):
    checkpoint = torch.load(weights_path, map_location=device)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("std", [0.229, 0.224, 0.225])
    class_to_idx = checkpoint.get("class_to_idx", {"NoPain": 0, "Pain": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    pain_idx = next(i for i, cls in idx_to_class.items() if cls.lower() == "pain")
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return model, transform, idx_to_class, pain_idx


def predict_pain_prob(
    model: nn.Module, transform, device: torch.device, image_bgr: np.ndarray, pain_idx: int
) -> float:
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return float(probs[pain_idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Timed pain monitoring with plotting.")
    parser.add_argument("--weights", type=Path, default=Path("models/pain_resnet18.pt"))
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in seconds.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between samples.")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Optional directory with images to loop over instead of a camera (useful for testing).",
    )
    parser.add_argument("--output", type=Path, default=Path("pain_monitor.png"), help="Output plot path.")
    parser.add_argument("--no-display", action="store_true", help="Disable live window overlay.")
    parser.add_argument("--no-face", action="store_true", help="Disable face detection crop.")
    parser.add_argument("--face-conf", type=float, default=0.5, help="Face detector confidence threshold.")
    parser.add_argument(
        "--ema",
        type=float,
        default=0.6,
        help="Exponential smoothing factor for probabilities (0 disables smoothing).",
    )
    parser.add_argument("--no-vis-filter", action="store_true", help="Disable landmark visibility filter.")
    parser.add_argument("--vis-threshold", type=float, default=0.35, help="Visibility threshold to allow inference.")
    parser.add_argument("--vis-scale", action="store_true", help="Scale probability by visibility score.")
    parser.add_argument("--no-hair-filter", action="store_true", help="Disable hair occlusion filter.")
    return parser.parse_args()


def collect_frames_from_dir(img_dir: Path) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for img_path in sorted(img_dir.glob("*")):
        img = cv2.imread(str(img_path))
        if img is not None:
            frames.append(img)
    if not frames:
        raise RuntimeError(f"No readable images in {img_dir}")
    return frames


def monitor(args: argparse.Namespace) -> Tuple[List[float], List[float]]:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model, transform, idx_to_class, pain_idx = load_model(args.weights, device)
    face_net = None if args.no_face else load_face_detector()
    vis_checker = None if args.no_vis_filter else FaceVisibilityChecker(static_image_mode=False)
    hair_checker = None if args.no_hair_filter else HairOcclusionChecker()

    use_dir = args.image_dir is not None
    if use_dir:
        frames = collect_frames_from_dir(args.image_dir)
    else:
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {args.camera_id}")

    times: List[float] = []
    pain_probs: List[float] = []
    start = time.time()

    try:
        frame_idx = 0
        smoothed_prob = None
        while True:
            elapsed = time.time() - start
            if elapsed >= args.duration:
                break

            if use_dir:
                frame = frames[frame_idx % len(frames)]
                frame_idx += 1
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Warning: failed to read frame; stopping.")
                    break

            if face_net and not args.no_face:
                frame_for_pred, found = crop_to_face_or_none(frame, face_net, conf_threshold=args.face_conf)
                if not found:
                    if not args.no_display:
                        cv2.putText(
                            frame,
                            "No face detected - pausing",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 165, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow("Pain monitor", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("Stopped by user.")
                            break
                    time.sleep(max(0.0, args.interval))
                    continue
            else:
                frame_for_pred = frame
            if vis_checker and not args.no_vis_filter:
                score = vis_checker.visibility_score(frame_for_pred)
                if score < args.vis_threshold:
                    if not args.no_display:
                        cv2.putText(
                            frame,
                            "Low visibility/skin - pausing",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 165, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow("Pain monitor", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("Stopped by user.")
                            break
                    time.sleep(max(0.0, args.interval))
                    continue
            else:
                score = 1.0

            if hair_checker and not args.no_hair_filter:
                lms = vis_checker.get_landmarks(frame_for_pred) if vis_checker else []
                if hair_checker.is_occluded(frame_for_pred, lms):
                    if not args.no_display:
                        cv2.putText(
                            frame,
                            "Hair occlusion - pausing",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 165, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow("Pain monitor", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("Stopped by user.")
                            break
                    time.sleep(max(0.0, args.interval))
                    continue

            pain_prob = predict_pain_prob(model, transform, device, frame_for_pred, pain_idx)
            if args.vis_scale:
                pain_prob *= score

            if args.ema > 0:
                smoothed_prob = pain_prob if smoothed_prob is None else args.ema * pain_prob + (1 - args.ema) * smoothed_prob
                use_prob = smoothed_prob
            else:
                use_prob = pain_prob

            times.append(elapsed)
            pain_probs.append(use_prob)

            if not args.no_display:
                label = f"Pain prob: {use_prob*100:.1f}%"
                color = (0, 0, 255) if use_prob >= 0.5 else (0, 255, 0)
                if face_net is not None and not args.no_face:
                    boxes = detect_faces(frame, face_net, conf_threshold=args.face_conf)
                    if boxes:
                        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.imshow("Pain monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break

            time.sleep(max(0.0, args.interval))
    finally:
        if not use_dir:
            cap.release()
        cv2.destroyAllWindows()

    return times, pain_probs


def plot_results(times: List[float], pain_probs: List[float], output: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(times, np.array(pain_probs) * 100.0, label="Pain probability")
    plt.axhline(50, color="red", linestyle="--", linewidth=1, label="50% threshold")
    plt.xlabel("Time (s)")
    plt.ylabel("Pain probability (%)")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close()


def main() -> None:
    args = parse_args()
    times, probs = monitor(args)
    if not probs:
        print("No data collected.")
        return

    mean_prob = float(np.mean(probs))
    pain_percent = float(np.mean(np.array(probs) >= 0.5) * 100.0)
    print(f"Collected {len(probs)} samples over {times[-1]:.1f}s")
    print(f"Average pain probability: {mean_prob*100:.1f}%")
    print(f"Percent frames flagged as pain (>=50%): {pain_percent:.1f}%")

    plot_results(times, probs, args.output)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
