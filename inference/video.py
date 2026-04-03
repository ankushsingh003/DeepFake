"""
inference/video.py
Batch inference on a video file (not live webcam).
Outputs:
  - Per-frame fake probability
  - Annotated output video with overlays
  - Summary stats (overall verdict, peak fake score, timeline)

Usage:
    python inference/video.py --input path/to/video.mp4 --checkpoint data/snapshots/best.ckpt
    python inference/video.py --input video.mp4 --checkpoint best.ckpt --output result.mp4 --heatmap
"""

import cv2
import torch
import argparse
import numpy as np
import collections
from pathlib import Path
from retinaface import RetinaFace

from models import DeepfakeDetector, get_combined_heatmap
from data.augment import VAL_SPATIAL
from train.train import DeepfakeLightning


def load_model(checkpoint: str) -> DeepfakeDetector:
    if checkpoint.endswith(".ckpt"):
        lit = DeepfakeLightning.load_from_checkpoint(checkpoint)
        return lit.model.eval()
    model = DeepfakeDetector()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval()


def preprocess(face_bgr: np.ndarray) -> torch.Tensor:
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return VAL_SPATIAL(face_rgb).unsqueeze(0)


def analyze_video(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(args.checkpoint).to(device)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {args.input}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_buffer  = collections.deque(maxlen=args.clip_len)
    frame_scores  = []   # (frame_idx, prob)
    frame_idx     = 0
    processed     = 0

    print(f"Analyzing {total} frames at {fps:.1f} fps...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every N frames for inference speed
        if frame_idx % args.frame_skip != 0:
            if writer:
                writer.write(frame)
            frame_idx += 1
            continue

        # Face detection
        try:
            faces = RetinaFace.detect_faces(frame)
        except Exception:
            faces = {}

        prob = None
        if faces:
            face_info = max(
                faces.values(),
                key=lambda f: (
                    (f["facial_area"][2] - f["facial_area"][0]) *
                    (f["facial_area"][3] - f["facial_area"][1])
                )
            )
            x1, y1, x2, y2 = face_info["facial_area"]
            x1 = max(0, x1 - 15); y1 = max(0, y1 - 15)
            x2 = min(width, x2 + 15); y2 = min(height, y2 + 15)
            face_crop = frame[y1:y2, x1:x2]

            spatial_t = preprocess(face_crop).to(device)
            frame_buffer.append(spatial_t.cpu())

            if len(frame_buffer) == args.clip_len:
                temporal_t = torch.stack(
                    [t.squeeze(0) for t in frame_buffer], dim=1
                ).unsqueeze(0).to(device)

                with torch.no_grad():
                    prob = torch.sigmoid(model(spatial_t, temporal_t)).item()

                frame_scores.append((frame_idx, prob))
                processed += 1

                # Annotate frame
                is_fake = prob > args.threshold
                color   = (50, 50, 220) if is_fake else (30, 180, 80)

                if args.heatmap and is_fake:
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    hm = get_combined_heatmap(model, spatial_t.cpu(), temporal_t.cpu(), face_rgb)
                    hm_bgr = cv2.cvtColor(hm, cv2.COLOR_RGB2BGR)
                    fh, fw = hm_bgr.shape[:2]
                    frame[y1:y1+fh, x1:x1+fw] = hm_bgr

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{'FAKE' if is_fake else 'REAL'} {prob:.2f}"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Progress bar overlay
        bar_filled = int(width * frame_idx / max(total, 1))
        cv2.rectangle(frame, (0, height - 5), (width, height), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, height - 5), (bar_filled, height), (100, 180, 100), -1)
        ts = f"{frame_idx / fps:.1f}s"
        cv2.putText(frame, ts, (4, height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        if writer:
            writer.write(frame)

        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total} frames  ({processed} with detections)")

    cap.release()
    if writer:
        writer.release()

    # ── Summary ──────────────────────────────────────────────────────────────
    if not frame_scores:
        print("No faces detected in video.")
        return

    probs = [p for _, p in frame_scores]
    avg_prob  = np.mean(probs)
    peak_prob = np.max(probs)
    fake_pct  = np.mean([p > args.threshold for p in probs]) * 100

    print("\n── Analysis complete ────────────────────────────────────")
    print(f"  Frames analyzed  : {len(frame_scores)}")
    print(f"  Average fake prob: {avg_prob:.4f}")
    print(f"  Peak fake prob   : {peak_prob:.4f}")
    print(f"  Frames > threshold: {fake_pct:.1f}%")
    print(f"  Overall verdict  : {'FAKE' if avg_prob > args.threshold else 'REAL'}")
    if args.output:
        print(f"  Annotated video  : {args.output}")

    # Save score timeline as .npy (raw numbers, no CSV)
    if args.save_scores:
        out = Path(args.input).stem + "_scores.npy"
        np.save(out, np.array(frame_scores))
        print(f"  Score timeline   : {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True,  help="Input .mp4 path")
    parser.add_argument("--checkpoint",  required=True,  help=".ckpt or .pth model path")
    parser.add_argument("--output",      default=None,   help="Annotated output .mp4 path")
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--clip-len",    type=int,   default=16)
    parser.add_argument("--frame-skip",  type=int,   default=1,
                        help="Run inference every N frames (1 = every frame)")
    parser.add_argument("--heatmap",     action="store_true")
    parser.add_argument("--save-scores", action="store_true",
                        help="Save per-frame scores as .npy")
    args = parser.parse_args()
    analyze_video(args)