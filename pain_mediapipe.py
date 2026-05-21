import argparse
import csv
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.mediapipe_face import MediaPipeFaceAnalyzer
from src.pain_rules import BaselineFeatures, compute_baseline_features, compute_pain_from_landmarks, diff_features
from src.overlay import anchor_point, draw_face_pain_skeleton, face_bbox, put_text_bg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rule-based pain estimation from MediaPipe (brows/eyes/nose/mouth).")
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--duration", type=float, default=0.0, help="Seconds to run (0 = until q).")
    p.add_argument("--interval", type=float, default=0.05, help="Seconds between samples (>=0.03 recommended).")
    p.add_argument("--ema", type=float, default=0.6, help="EMA smoothing (0 disables).")
    p.add_argument("--output", type=Path, default=Path("pain_mediapipe.png"))
    p.add_argument("--csv-out", type=Path, help="Optional CSV export of per-sample values.")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--calibrate", type=float, default=3.0, help="Seconds to capture neutral baseline.")
    p.add_argument("--threshold", type=float, default=60.0, help="Pain threshold (%%) for summary/plot line.")
    p.add_argument("--show-deltas", action="store_true", help="Overlay MediaPipe feature deltas (current-baseline).")
    p.add_argument("--draw-face-lines", action="store_true", help="Draw face lines/points like a skeleton overlay.")
    return p.parse_args()


def plot_results(
    times: List[float],
    pain: List[float],
    occluded: List[bool],
    output: Path,
    threshold: float,
    avg: float,
    pct_above: float,
) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(times, pain, label="Pain (%)")
    if occluded:
        occ_t = [t for t, o in zip(times, occluded) if o]
        occ_p = [p for p, o in zip(pain, occluded) if o]
        if occ_t:
            plt.scatter(occ_t, occ_p, s=12, c="orange", label="Occluded regions")
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1, label=f"Threshold {threshold:.0f}%")
    plt.xlabel("Time (s)")
    plt.ylabel("Pain (%)")
    plt.ylim(0, 100)
    plt.title(f"Avg {avg:.1f}% | >=thr: {pct_above:.1f}% of samples")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close()

def write_csv(
    path: Path,
    times: List[float],
    pain: List[float],
    components_rows: List[Tuple[float, float, float, float, float]],
    occluded_regions: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "pain_percent", "brow", "eye", "nose", "mouth", "jaw", "occluded_regions"])
        for t, p, comp, occ in zip(times, pain, components_rows, occluded_regions):
            brow, eye, nose, mouth, jaw = comp
            w.writerow([f"{t:.3f}", f"{p:.3f}", f"{brow:.4f}", f"{eye:.4f}", f"{nose:.4f}", f"{mouth:.4f}", f"{jaw:.4f}", occ])


def main() -> None:
    args = parse_args()
    if args.interval < 0.03:
        print(f"Warning: --interval {args.interval} is very small; clamping to 0.03 for stability.")
        args.interval = 0.03
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera_id}")

    times: List[float] = []
    pains: List[float] = []
    smoothed = None
    baseline: BaselineFeatures | None = None
    calib_features: List[BaselineFeatures] = []
    calib_landmarks: List[List[Tuple[int, int]]] = []
    baseline_points: dict[int, Tuple[int, int]] | None = None
    occluded_flags: List[bool] = []
    occluded_text: List[str] = []
    components_rows: List[Tuple[float, float, float, float, float]] = []

    start = time.time()
    last_sample = 0.0

    with MediaPipeFaceAnalyzer(use_blendshapes=False) as analyzer:
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
                    cv2.imshow("Pain (MediaPipe rules)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue
            last_sample = now

            analysis = analyzer.analyze(frame)
            if analysis is None:
                if not args.no_display:
                    cv2.putText(
                        frame,
                        "No face detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Pain (MediaPipe rules)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            # Calibration phase (neutral face)
            if baseline is None and args.calibrate > 0 and elapsed < args.calibrate:
                calib_features.append(compute_baseline_features(analysis.landmarks_px))
                calib_landmarks.append(analysis.landmarks_px)
                if not args.no_display:
                    cv2.putText(
                        frame,
                        f"Calibrating... {elapsed:.1f}/{args.calibrate:.1f}s",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Pain (MediaPipe rules)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue
            if baseline is None and args.calibrate > 0 and elapsed >= args.calibrate:
                if calib_features:
                    # Robust baseline: median of calibration samples
                    arr = np.array([[f.brow_eye, f.eye_ear, f.nose_lip, f.mouth_frown, f.mouth_open] for f in calib_features])
                    med = np.median(arr, axis=0)
                    baseline = BaselineFeatures(
                        brow_eye=float(med[0]),
                        eye_ear=float(med[1]),
                        nose_lip=float(med[2]),
                        mouth_frown=float(med[3]),
                        mouth_open=float(med[4]),
                    )
                    # Median baseline pixel positions for a few key points to visualize "deviation"
                    key_idx = [33, 263, 70, 300, 1, 13, 14, 61, 291]
                    pts = []
                    for lm in calib_landmarks:
                        if all(i < len(lm) for i in key_idx):
                            pts.append([lm[i] for i in key_idx])
                    if pts:
                        arr_xy = np.array(pts, dtype=np.float32)  # [N,K,2]
                        med_xy = np.median(arr_xy, axis=0).astype(int)
                        baseline_points = {i: (int(x), int(y)) for i, (x, y) in zip(key_idx, med_xy)}
                else:
                    baseline = compute_baseline_features(analysis.landmarks_px)

            if baseline is None:
                baseline = compute_baseline_features(analysis.landmarks_px)

            res = compute_pain_from_landmarks(frame, analysis.landmarks_px, baseline)
            pain = float(res.pain_percent)
            if args.ema > 0:
                smoothed = pain if smoothed is None else args.ema * pain + (1 - args.ema) * smoothed
                pain = float(smoothed)

            times.append(elapsed)
            pains.append(pain)
            occluded_flags.append(bool(res.occluded_regions))
            occluded_text.append(",".join(res.occluded_regions))
            c = res.components
            components_rows.append((c.brow_down, c.eye_squint, c.nose_sneer, c.mouth_frown, c.jaw_open))

            if not args.no_display:
                if args.draw_face_lines:
                    draw_face_pain_skeleton(
                        frame,
                        analysis.landmarks_px,
                        point_color=(0, 0, 255),
                        line_color=(255, 255, 255),
                        point_radius=2,
                        line_thickness=1,
                    )
                details = f"brow:{c.brow_down:.2f} eye:{c.eye_squint:.2f} nose:{c.nose_sneer:.2f} mouth:{c.mouth_frown:.2f} jaw:{c.jaw_open:.2f}"
                bbox = face_bbox(analysis.landmarks_px, frame.shape[1], frame.shape[0])
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                # Fixed panel (top-left): pain + components + occlusion
                y = 30
                color = (0, 0, 255) if pain >= args.threshold else (0, 255, 0)
                put_text_bg(frame, f"Pain: {pain:5.1f}%", (10, y), text_color=color)
                y += 28
                put_text_bg(frame, details, (10, y), font_scale=0.55, text_color=(255, 255, 255))
                y += 26
                if res.occluded_regions:
                    put_text_bg(
                        frame,
                        "Occluded: " + ",".join(res.occluded_regions),
                        (10, y),
                        font_scale=0.55,
                        text_color=(0, 165, 255),
                    )

                if args.show_deltas and baseline is not None:
                    cur_feat = compute_baseline_features(analysis.landmarks_px)
                    d = diff_features(baseline, cur_feat)
                    # Place deltas directly on-face near relevant regions
                    brow_pt = anchor_point(analysis.landmarks_px, [70, 63, 105, 66, 107, 336, 296, 334, 293, 300])
                    eye_pt = anchor_point(analysis.landmarks_px, [33, 263, 159, 386])
                    nose_pt = anchor_point(analysis.landmarks_px, [1, 2, 5, 4])
                    mouth_pt = anchor_point(analysis.landmarks_px, [13, 14, 61, 291])
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

                    if baseline_points:
                        for idx, (bx, by) in baseline_points.items():
                            if idx < len(analysis.landmarks_px):
                                cx, cy = analysis.landmarks_px[idx]
                                cv2.circle(frame, (bx, by), 3, (255, 0, 0), -1)  # baseline
                                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)  # current
                if res.occluded_regions:
                    # already included in HUD
                    pass
                cv2.imshow("Pain (MediaPipe rules)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()

    if pains:
        avg = float(np.mean(pains))
        pct_above = float(np.mean(np.array(pains) >= args.threshold) * 100.0)
        print(f"Samples: {len(pains)}; Average pain: {avg:.1f}% ; >= {args.threshold:.0f}%: {pct_above:.1f}% of samples")
        plot_results(times, pains, occluded_flags, args.output, args.threshold, avg, pct_above)
        print(f"Saved plot: {args.output}")
        if args.csv_out:
            write_csv(args.csv_out, times, pains, components_rows, occluded_text)
            print(f"Saved CSV: {args.csv_out}")
    else:
        print("No samples collected.")


if __name__ == "__main__":
    main()
