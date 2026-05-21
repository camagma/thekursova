import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from src.face_detector import crop_to_face_or_none, detect_faces, load_face_detector
from src.visibility_filter import FaceVisibilityChecker
from src.hair_filter import HairOcclusionChecker


def load_model(weights_path: Path, device: torch.device) -> Tuple[nn.Module, dict]:
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
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return model, transform, idx_to_class


def predict_image(model: nn.Module, transform, device: torch.device, image_bgr: np.ndarray) -> Tuple[int, float]:
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, float(probs[pred_idx])


def run_image(
    path: Path,
    model: nn.Module,
    transform,
    device: torch.device,
    idx_to_class: dict,
    face_net,
    use_face: bool,
    face_conf: float,
    vis_checker: FaceVisibilityChecker,
    use_vis_filter: bool,
    vis_threshold: float,
    vis_scale: bool,
    hair_checker: HairOcclusionChecker,
    use_hair_filter: bool,
) -> None:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    if use_face:
        image, found = crop_to_face_or_none(image, face_net, conf_threshold=face_conf)
        if not found:
            print("No face detected in image; skipping prediction.")
            return
    score = 1.0
    if use_vis_filter:
        score = vis_checker.visibility_score(image)
        if score < vis_threshold:
            print("Low visibility/skin; skipping prediction.")
            return
    pred_idx, prob = predict_image(model, transform, device, image)
    if use_hair_filter:
        lms = vis_checker.get_landmarks(image) if use_vis_filter else []
        if hair_checker.is_occluded(image, lms):
            print("Hair occlusion detected; skipping prediction.")
            return
    if vis_scale:
        prob = prob * score
    print(f"{path}: {idx_to_class[pred_idx]} ({prob:.2%} confidence, vis_score={score:.2f})")


def run_camera(
    model: nn.Module,
    transform,
    device: torch.device,
    idx_to_class: dict,
    face_net,
    use_face: bool,
    face_conf: float,
    vis_checker: FaceVisibilityChecker,
    use_vis_filter: bool,
    vis_threshold: float,
    vis_scale: bool,
    hair_checker: HairOcclusionChecker,
    use_hair_filter: bool,
    camera: int = 0,
) -> None:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if use_face:
                frame_for_pred, found = crop_to_face_or_none(frame, face_net, conf_threshold=face_conf)
                if not found:
                    cv2.putText(frame, "No face detected - pausing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Pain detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
            else:
                frame_for_pred = frame
            score = 1.0
            if use_vis_filter:
                score = vis_checker.visibility_score(frame_for_pred)
                if score < vis_threshold:
                    cv2.putText(frame, "Low visibility/skin - pausing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Pain detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
            if use_hair_filter:
                lms = vis_checker.get_landmarks(frame_for_pred) if use_vis_filter else []
                if hair_checker.is_occluded(frame_for_pred, lms):
                    cv2.putText(frame, "Hair occlusion - pausing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Pain detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
            if face_net is not None and use_face:
                boxes = detect_faces(frame, face_net, conf_threshold=face_conf)
                if boxes:
                    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            pred_idx, prob = predict_image(model, transform, device, frame_for_pred)
            if vis_scale:
                prob = prob * score
            label = f"{idx_to_class[pred_idx]} {prob:.1%}"
            color = (0, 255, 0) if idx_to_class[pred_idx] == "NoPain" else (0, 0, 255)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.imshow("Pain detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pain/no-pain inference.")
    parser.add_argument("--weights", type=Path, default=Path("models/pain_resnet18.pt"))
    parser.add_argument("--image", type=Path, help="Path to image for inference.")
    parser.add_argument("--camera", action="store_true", help="Use webcam instead of image file.")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--no-face", action="store_true", help="Disable face detection crop.")
    parser.add_argument("--face-conf", type=float, default=0.5, help="Face detector confidence threshold.")
    parser.add_argument("--no-vis-filter", action="store_true", help="Disable landmark visibility filter.")
    parser.add_argument("--vis-threshold", type=float, default=0.35, help="Visibility threshold to allow inference.")
    parser.add_argument("--vis-scale", action="store_true", help="Scale probability by visibility score.")
    parser.add_argument("--no-hair-filter", action="store_true", help="Disable hair occlusion filter.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model, transform, idx_to_class = load_model(args.weights, device)
    face_net = None if args.no_face else load_face_detector()
    vis_checker = FaceVisibilityChecker(static_image_mode=bool(args.image))
    hair_checker = HairOcclusionChecker()
    if args.camera:
        run_camera(
            model,
            transform,
            device,
            idx_to_class,
            face_net,
            use_face=not args.no_face,
            face_conf=args.face_conf,
            vis_checker=vis_checker,
            use_vis_filter=not args.no_vis_filter,
            vis_threshold=args.vis_threshold,
            vis_scale=args.vis_scale,
            hair_checker=hair_checker,
            use_hair_filter=not args.no_hair_filter,
            camera=args.camera_id,
        )
    elif args.image:
        run_image(
            args.image,
            model,
            transform,
            device,
            idx_to_class,
            face_net,
            use_face=not args.no_face,
            face_conf=args.face_conf,
            vis_checker=vis_checker,
            use_vis_filter=not args.no_vis_filter,
            vis_threshold=args.vis_threshold,
            vis_scale=args.vis_scale,
            hair_checker=hair_checker,
            use_hair_filter=not args.no_hair_filter,
        )
    else:
        raise SystemExit("Provide --image <path> or --camera to run inference.")


if __name__ == "__main__":
    main()
