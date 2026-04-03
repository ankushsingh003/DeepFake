"""
inference/realtime.py
Real-time deepfake detection from webcam.
Runs RetinaFace face detection + DeepfakeDetector inference on each frame.
Draws bounding box, verdict badge, confidence bar, and optional Grad-CAM heatmap.

Usage:
    python inference/realtime.py --checkpoint data/snapshots/best.ckpt
    python inference/realtime.py --checkpoint data/snapshots/best.ckpt --heatmap
    python inference/realtime.py --checkpoint data/snapshots/best.ckpt --threshold 0.6
"""

import cv2
import torch
import argparse
import numpy as np
import collections
from retinaface import RetinaFace

from models import DeepfakeDetector, get_combined_heatmap
from data.augment import VAL_SPATIAL
from train.train import DeepfakeLightning


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> DeepfakeDetector:
    if checkpoint_path.endswith(".ckpt"):
        lit = DeepfakeLightning.load_from_checkpoint(checkpoint_path)
        model = lit.model
    else:
        model = DeepfakeDetector()
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def preprocess_spatial(face_bgr: np.ndarray) -> torch.Tensor:
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return VAL_SPATIAL(face_rgb).unsqueeze(0)           # (1, 3, 224, 224)


def build_temporal(buffer: list) -> torch.Tensor:
    """Stack buffered face tensors into (1, 3, T, 224, 224)."""
    clip = torch.stack([t.squeeze(0) for t in buffer], dim=1)  # (3, T, H, W)
    return clip.unsqueeze(0)                                    # (1, 3, T, H, W)


def draw_verdict(frame, x1, y1, x2, y2, verdict: str, prob: float, threshold: float):
    is_fake = prob > threshold
    color   = (50, 50, 220) if is_fake else (30, 180, 80)   # BGR

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Badge background
    label   = f"{'FAKE' if is_fake else 'REAL'}  {prob:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Confidence bar at bottom of box
    bar_w = x2 - x1
    filled = int(bar_w * prob)
    cv2.rectangle(frame, (x1, y2 + 2), (x2, y2 + 8), (60, 60, 60), -1)
    cv2.rectangle(frame, (x1, y2 + 2), (x1 + filled, y2 + 8), color, -1)


def draw_hud(frame, fps: float, buffered: int, clip_len: int):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}   Buffer: {buffered}/{clip_len}",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint).to(device)
    frame_buffer = collections.deque(maxlen=args.clip_len)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_counter = collections.deque(maxlen=30)
    import time
    prev_time = time.time()

    print("Running — press Q to quit, H to toggle heatmap.")
    show_heatmap = args.heatmap

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS
        now = time.time()
        fps_counter.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps = sum(fps_counter) / len(fps_counter)

        # Face detection (every frame)
        try:
            faces = RetinaFace.detect_faces(frame)
        except Exception:
            faces = {}

        if faces:
            face_info = max(
                faces.values(),
                key=lambda f: (
                    (f["facial_area"][2] - f["facial_area"][0]) *
                    (f["facial_area"][3] - f["facial_area"][1])
                )
            )
            x1, y1, x2, y2 = face_info["facial_area"]
            pad = 20
            x1 = max(0, x1 - pad);  y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)
            face_crop = frame[y1:y2, x1:x2]

            spatial_t = preprocess_spatial(face_crop).to(device)
            frame_buffer.append(spatial_t.cpu())

            if len(frame_buffer) == args.clip_len:
                temporal_t = build_temporal(list(frame_buffer)).to(device)

                with torch.no_grad():
                    prob = torch.sigmoid(model(spatial_t, temporal_t)).item()

                # Optional heatmap overlay
                if show_heatmap:
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    heatmap  = get_combined_heatmap(
                        model,
                        spatial_t.cpu(),
                        temporal_t.cpu(),
                        face_rgb,
                    )
                    face_h   = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                    fh, fw   = face_h.shape[:2]
                    frame[y1:y1+fh, x1:x1+fw] = face_h

                draw_verdict(frame, x1, y1, x2, y2, "FAKE" if prob > args.threshold else "REAL",
                             prob, args.threshold)
            else:
                # Still buffering
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 0), 1)
                cv2.putText(frame, f"Buffering {len(frame_buffer)}/{args.clip_len}",
                            (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

        draw_hud(frame, fps, len(frame_buffer), args.clip_len)
        cv2.imshow("Deepfake Detector — live", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("h"):
            show_heatmap = not show_heatmap
            print(f"Heatmap: {'ON' if show_heatmap else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt or .pth model file")
    parser.add_argument("--camera",     type=int,   default=0,    help="Camera index")
    parser.add_argument("--threshold",  type=float, default=0.5,  help="Fake decision threshold")
    parser.add_argument("--clip-len",   type=int,   default=16,   help="Temporal buffer size")
    parser.add_argument("--heatmap",    action="store_true",       help="Show Grad-CAM heatmap")
    args = parser.parse_args()
    run(args)