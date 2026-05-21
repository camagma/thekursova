import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.mediapipe_face import MediaPipeFaceAnalyzer
from src.mediapipe_pose import MediaPipePoseAnalyzer
from src.micro_motion import (
    LandmarkMicroMotion,
    MotionSignals,
    face_regions_from_landmarks,
    face_scale,
    pose_regions_from_landmarks,
    pose_scale,
)
from src.pain_rules import BaselineFeatures, compute_baseline_features, compute_pain_from_landmarks, diff_features
from src.overlay import POSE_CONNECTIONS, anchor_point, draw_face_pain_skeleton, draw_skeleton, face_bbox, put_text_bg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multimodal pain monitor: face pain rules + face/body micro-movements.")
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--interval", type=float, default=0.05)
    p.add_argument("--ema", type=float, default=0.6)
    p.add_argument("--calibrate", type=float, default=3.0, help="Seconds to capture neutral baseline.")
    p.add_argument("--output", type=Path, default=Path("pain_multimodal.png"))
    p.add_argument("--csv-out", type=Path, default=Path("pain_multimodal.csv"))
    p.add_argument("--npz-out", type=Path, default=Path("pain_multimodal.npz"))
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--threshold", type=float, default=60.0)
    p.add_argument("--pose-min-quality", type=float, default=0.55, help="Minimum pose quality to treat as body.")
    p.add_argument("--pose-detect-conf", type=float, default=0.3, help="Pose detection confidence (MediaPipe).")
    p.add_argument("--show-deltas", action="store_true", help="Overlay MediaPipe face feature deltas and pose quality.")
    p.add_argument("--draw-pose", action="store_true", help="Draw pose skeleton lines/points like typical pose overlays.")
    p.add_argument(
        "--draw-pose-raw",
        action="store_true",
        help="Draw pose overlay even when pose is ignored (debug).",
    )
    p.add_argument("--draw-face-lines", action="store_true", help="Draw face lines/points like a skeleton overlay.")
    p.add_argument("--w-face", type=float, default=0.7, help="Weight for face pain (when available).")
    p.add_argument("--w-body", type=float, default=0.3, help="Weight for body micro-motion (when available).")
    return p.parse_args()


def _clamp_interval(interval: float) -> float:
    return max(0.03, float(interval))


def _motion_to_percent(m: float, gain: float = 800.0) -> float:
    return float(np.clip(100.0 * (1.0 - np.exp(-gain * m)), 0.0, 100.0))


def fuse_scores(face_score: Optional[float], body_score: Optional[float], w_face: float, w_body: float) -> float:
    parts = []
    weights = []
    if face_score is not None:
        parts.append(face_score)
        weights.append(w_face)
    if body_score is not None:
        parts.append(body_score)
        weights.append(w_body)
    if not parts:
        return float("nan")
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()
    return float(np.dot(np.array(parts, dtype=np.float32), weights))


def plot(times: List[float], total: List[float], face: List[float], body: List[float], threshold: float, out: Path) -> None:
    plt.figure(figsize=(11, 4))
    plt.plot(times, total, label="Total pain (%)", linewidth=2)
    plt.plot(times, face, label="Face pain (%)", alpha=0.7)
    plt.plot(times, body, label="Body micro-motion (%)", alpha=0.7)
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1, label=f"Threshold {threshold:.0f}%")
    plt.ylim(0, 100)
    plt.xlabel("Time (s)")
    plt.ylabel("Pain (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()


def main() -> None:
    args = parse_args()
    args.interval = _clamp_interval(args.interval)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera_id}")

    times: List[float] = []
    total_series: List[float] = []
    face_series: List[float] = []
    body_series: List[float] = []


    face_micro_hist: Dict[str, List[float]] = {}
    pose_micro_hist: Dict[str, List[float]] = {}
    face_micro_amp_hist: Dict[str, List[float]] = {}
    pose_micro_amp_hist: Dict[str, List[float]] = {}
    pose_quality_series: List[float] = []

    smoothed_total = None

    baseline: BaselineFeatures | None = None
    calib_features: List[BaselineFeatures] = []
    calib_face_micro: List[float] = []
    calib_pose_micro: List[float] = []
    face_micro0: float | None = None
    pose_micro0: float | None = None

    face_motion = LandmarkMicroMotion(ema_alpha=0.08)
    pose_motion = LandmarkMicroMotion(ema_alpha=0.08)

    start = time.time()
    last_sample = 0.0

    with MediaPipeFaceAnalyzer(use_blendshapes=False) as face_analyzer, MediaPipePoseAnalyzer(
        min_pose_detection_confidence=args.pose_detect_conf
    ) as pose_analyzer:
        while True:
            now = time.time()
            elapsed = now - start
            if args.duration > 0 and elapsed >= args.duration:
                break

            ret, frame = cap.read()
            if not ret:
                break
            
            if (now - last_sample) < args.interval:
                if not args.no_display:
                    cv2.imshow("Pain multimodal", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue
            dt_sample = float(max(now - last_sample, args.interval))
            last_sample = now

            face_res = face_analyzer.analyze(frame)
            pose_res = pose_analyzer.analyze(frame)

            face_score: Optional[float] = None
            body_score: Optional[float] = None

            if face_res is not None:
                if baseline is None and args.calibrate > 0 and elapsed < args.calibrate:
                    calib_features.append(compute_baseline_features(face_res.landmarks_px))
                    f_regions = face_regions_from_landmarks(face_res.landmarks_px)
                    f_scale = face_scale(face_res.landmarks_px)
                    f_motion: MotionSignals = face_motion.update(f_regions, scale=f_scale, dt=dt_sample)
                    calib_face_micro.append(float(f_motion.overall))

                    if pose_res is not None and pose_res.is_body and pose_res.quality >= args.pose_min_quality:
                        p_regions = pose_regions_from_landmarks(pose_res.landmarks_px)
                        p_scale = pose_scale(pose_res.landmarks_px)
                        p_motion: MotionSignals = pose_motion.update(p_regions, scale=p_scale, dt=dt_sample)
                        calib_pose_micro.append(float(p_motion.overall))
                    if not args.no_display:
                        cv2.putText(
                            frame,
                            f"Calibrating face baseline... {elapsed:.1f}/{args.calibrate:.1f}s",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 165, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow("Pain multimodal", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue

                if baseline is None and args.calibrate > 0 and elapsed >= args.calibrate:
                    if calib_features:
                        arr = np.array(
                            [[f.brow_eye, f.eye_ear, f.nose_lip, f.mouth_frown, f.mouth_open] for f in calib_features]
                        )
                        med = np.median(arr, axis=0)
                        baseline = BaselineFeatures(
                            brow_eye=float(med[0]),
                            eye_ear=float(med[1]),
                            nose_lip=float(med[2]),
                            mouth_frown=float(med[3]),
                            mouth_open=float(med[4]),
                        )
                        if face_micro0 is None and calib_face_micro:
                            face_micro0 = float(np.median(np.array(calib_face_micro, dtype=np.float32)))
                        if pose_micro0 is None and calib_pose_micro:
                            pose_micro0 = float(np.median(np.array(calib_pose_micro, dtype=np.float32)))
                    else:
                        baseline = compute_baseline_features(face_res.landmarks_px)

                if baseline is None:
                    baseline = compute_baseline_features(face_res.landmarks_px)
                pain_face = compute_pain_from_landmarks(frame, face_res.landmarks_px, baseline).pain_percent
                f_regions = face_regions_from_landmarks(face_res.landmarks_px)
                f_scale = face_scale(face_res.landmarks_px)
                f_motion: MotionSignals = face_motion.update(f_regions, scale=f_scale, dt=dt_sample)
                for k, v in f_motion.per_region.items():
                    face_micro_hist.setdefault(k, []).append(v)
                face_micro_hist.setdefault("overall", []).append(f_motion.overall)
                for k, v in f_motion.per_region_amp.items():
                    face_micro_amp_hist.setdefault(k, []).append(v)
                face_micro_amp_hist.setdefault("overall", []).append(f_motion.overall_amp)

                face_floor = float(face_micro0 or 0.0)
                face_micro_percent = _motion_to_percent(max(0.0, f_motion.overall - face_floor), gain=12.0)
                face_score = float(np.clip(0.85 * pain_face + 0.15 * face_micro_percent, 0.0, 100.0))

            if pose_res is not None:
                pose_quality_series.append(pose_res.quality)
            else:
                pose_quality_series.append(float("nan"))

            if pose_res is not None and pose_res.is_body and pose_res.quality >= args.pose_min_quality:
                p_regions = pose_regions_from_landmarks(pose_res.landmarks_px)
                p_scale = pose_scale(pose_res.landmarks_px)
                p_motion: MotionSignals = pose_motion.update(p_regions, scale=p_scale, dt=dt_sample)
                for k, v in p_motion.per_region.items():
                    pose_micro_hist.setdefault(k, []).append(v)
                pose_micro_hist.setdefault("overall", []).append(p_motion.overall)
                for k, v in p_motion.per_region_amp.items():
                    pose_micro_amp_hist.setdefault(k, []).append(v)
                pose_micro_amp_hist.setdefault("overall", []).append(p_motion.overall_amp)
                pose_floor = float(pose_micro0 or 0.0)
                body_score = _motion_to_percent(max(0.0, p_motion.overall - pose_floor), gain=8.0)
            elif pose_res is not None:
                body_score = None

            total = fuse_scores(face_score, body_score, args.w_face, args.w_body)
            if np.isnan(total):
                if not args.no_display:
                    cv2.putText(
                        frame,
                        "No face/body detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Pain multimodal", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            if args.ema > 0:
                smoothed_total = total if smoothed_total is None else args.ema * total + (1 - args.ema) * smoothed_total
                total = float(smoothed_total)

            times.append(elapsed)
            total_series.append(total)
            face_series.append(face_score if face_score is not None else float("nan"))
            body_series.append(body_score if body_score is not None else float("nan"))

            if not args.no_display:
                if args.draw_pose and pose_res is not None:
                    allowed = bool(pose_res.is_body and pose_res.quality >= args.pose_min_quality)
                    if allowed or args.draw_pose_raw:
                        draw_skeleton(
                            frame,
                            pose_res.landmarks_px,
                            POSE_CONNECTIONS,
                            point_color=(0, 0, 255) if allowed else (0, 165, 255),
                            line_color=(255, 255, 255) if allowed else (140, 140, 140),
                            point_radius=4,
                            line_thickness=2,
                        )
                if args.draw_face_lines and face_res is not None:
                    draw_face_pain_skeleton(
                        frame,
                        face_res.landmarks_px,
                        point_color=(0, 0, 255),
                        line_color=(255, 255, 255),
                        point_radius=2,
                        line_thickness=1,
                    )

                bbox = face_bbox(face_res.landmarks_px, frame.shape[1], frame.shape[0]) if face_res is not None else None
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                # Fixed panel (top-left): pain percentages and general status.
                color = (0, 0, 255) if total >= args.threshold else (0, 255, 0)
                y = 30
                put_text_bg(frame, f"Total: {total:5.1f}%", (10, y), text_color=color)
                y += 28
                put_text_bg(
                    frame,
                    f"Face: {face_score if face_score is not None else float('nan'):.1f}%  Body: {body_score if body_score is not None else float('nan'):.1f}%",
                    (10, y),
                    font_scale=0.6,
                    text_color=(255, 255, 255),
                )
                y += 26
                if pose_res is not None:
                    put_text_bg(
                        frame,
                        f"Pose quality: {pose_res.quality:.2f} body={int(pose_res.is_body)}",
                        (10, y),
                        font_scale=0.55,
                        text_color=(255, 255, 255),
                    )
                    y += 26
                if pose_res is not None and pose_res.is_body and pose_res.quality >= args.pose_min_quality:
                    last_vals = {k: pose_micro_hist[k][-1] for k in pose_micro_hist if k != "overall" and pose_micro_hist[k]}
                    if last_vals:
                        top = max(last_vals.items(), key=lambda kv: kv[1])
                        put_text_bg(
                            frame,
                            f"Top motion: {top[0]} ({_motion_to_percent(top[1]):.0f}%)",
                            (10, y),
                            text_color=(0, 165, 255),
                        )
                elif pose_res is not None:
                    put_text_bg(frame, f"Pose ignored (quality {pose_res.quality:.2f})", (10, y), text_color=(0, 165, 255))
                if args.show_deltas and face_res is not None and baseline is not None:
                    cur_feat = compute_baseline_features(face_res.landmarks_px)
                    d = diff_features(baseline, cur_feat)
                    brow_pt = anchor_point(face_res.landmarks_px, [70, 63, 105, 66, 107, 336, 296, 334, 293, 300])
                    eye_pt = anchor_point(face_res.landmarks_px, [33, 263, 159, 386])
                    nose_pt = anchor_point(face_res.landmarks_px, [1, 2, 5, 4])
                    mouth_pt = anchor_point(face_res.landmarks_px, [13, 14, 61, 291])
                    if brow_pt:
                        put_text_bg(frame, f"d_brow {d['brow_eye']:+.3f}", (brow_pt[0], brow_pt[1] - 10), font_scale=0.5)
                    if eye_pt:
                        put_text_bg(frame, f"d_eye {d['eye_ear']:+.3f}", (eye_pt[0], eye_pt[1] - 10), font_scale=0.5)
                    if nose_pt:
                        put_text_bg(frame, f"d_nose {d['nose_lip']:+.3f}", (nose_pt[0], nose_pt[1] + 20), font_scale=0.5)
                    if mouth_pt:
                        put_text_bg(
                            frame,
                            f"d_frown {d['mouth_frown']:+.3f} d_open {d['mouth_open']:+.3f}",
                            (mouth_pt[0], mouth_pt[1] + 25),
                            font_scale=0.5,
                        )
                cv2.imshow("Pain multimodal", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()

    if not total_series:
        print("No samples collected.")
        return

    avg_total = float(np.nanmean(np.array(total_series)))
    pct_above = float(np.mean(np.array(total_series) >= args.threshold) * 100.0)
    print(f"Samples: {len(total_series)}; Avg total pain: {avg_total:.1f}% ; >= {args.threshold:.0f}%: {pct_above:.1f}%")

    plot(times, total_series, face_series, body_series, args.threshold, args.output)
    print(f"Saved plot: {args.output}")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "total_pain", "face_pain", "body_motion"])
        for t, tot, fa, bo in zip(times, total_series, face_series, body_series):
            w.writerow([f"{t:.3f}", f"{tot:.3f}", f"{fa:.3f}", f"{bo:.3f}"])
    print(f"Saved CSV: {args.csv_out}")
    arrays = {
        "t_sec": np.array(times, dtype=np.float32),
        "total_pain": np.array(total_series, dtype=np.float32),
        "face_pain": np.array(face_series, dtype=np.float32),
        "body_motion": np.array(body_series, dtype=np.float32),
        "pose_quality": np.array(pose_quality_series[: len(times)], dtype=np.float32),
    }
    for k, v in face_micro_hist.items():
        arrays[f"face_micro_{k}"] = np.array(v, dtype=np.float32)
    for k, v in face_micro_amp_hist.items():
        arrays[f"face_micro_amp_{k}"] = np.array(v, dtype=np.float32)
    for k, v in pose_micro_hist.items():
        arrays[f"pose_micro_{k}"] = np.array(v, dtype=np.float32)
    for k, v in pose_micro_amp_hist.items():
        arrays[f"pose_micro_amp_{k}"] = np.array(v, dtype=np.float32)
    args.npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.npz_out, **arrays)
    print(f"Saved micro-motion NPZ: {args.npz_out}")


if __name__ == "__main__":
    main()
