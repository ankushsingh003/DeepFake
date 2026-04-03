"""
inference/server.py
FastAPI WebSocket server — receives base64 frames from the browser UI,
runs DeepfakeDetector, returns verdict + Grad-CAM heatmap.

Usage:
    pip install fastapi uvicorn websockets
    uvicorn inference.server:app --host 0.0.0.0 --port 8000 --reload

Then open the browser UI at http://localhost:3000
"""

import base64
import collections
import json

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from retinaface import RetinaFace

from models import DeepfakeDetector, get_combined_heatmap
from data.augment import VAL_SPATIAL
from train.train import DeepfakeLightning

app = FastAPI(title="Deepfake Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model (loaded once at startup) ─────────────────────────────────────

_model: DeepfakeDetector = None
_device: torch.device = None


@app.on_event("startup")
async def load_model():
    global _model, _device
    import os
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = os.environ.get("DEEPFAKE_CHECKPOINT", "data/snapshots/best.ckpt")
    if ckpt.endswith(".ckpt"):
        lit = DeepfakeLightning.load_from_checkpoint(ckpt, map_location=_device)
        _model = lit.model.eval().to(_device)
    else:
        _model = DeepfakeDetector().eval().to(_device)
        _model.load_state_dict(torch.load(ckpt, map_location=_device))

    print(f"Model loaded on {_device}")


# ── Per-connection state ───────────────────────────────────────────────────────

class ConnectionState:
    def __init__(self, clip_len: int = 16):
        self.frame_buffer: collections.deque = collections.deque(maxlen=clip_len)
        self.clip_len = clip_len


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws/detect")
async def detect(ws: WebSocket):
    await ws.accept()
    state = ConnectionState()

    try:
        while True:
            raw = await ws.receive_text()
            payload = json.loads(raw)

            threshold   = float(payload.get("threshold", 0.5))
            want_heatmap = bool(payload.get("heatmap", False))

            # Decode base64 JPEG frame
            img_bytes = base64.b64decode(payload["frame"])
            arr   = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                await ws.send_text(json.dumps({"error": "bad frame"}))
                continue

            # Face detection
            try:
                faces = RetinaFace.detect_faces(frame)
            except Exception:
                faces = {}

            if not faces:
                await ws.send_text(json.dumps({"verdict": None, "reason": "no_face"}))
                continue

            face_info = max(
                faces.values(),
                key=lambda f: (
                    (f["facial_area"][2] - f["facial_area"][0]) *
                    (f["facial_area"][3] - f["facial_area"][1])
                )
            )
            x1, y1, x2, y2 = face_info["facial_area"]
            h, w = frame.shape[:2]
            x1 = max(0, x1 - 15); y1 = max(0, y1 - 15)
            x2 = min(w, x2 + 15); y2 = min(h, y2 + 15)
            face_crop = frame[y1:y2, x1:x2]

            face_rgb  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            spatial_t = VAL_SPATIAL(face_rgb).unsqueeze(0).to(_device)
            state.frame_buffer.append(spatial_t.cpu())

            if len(state.frame_buffer) < state.clip_len:
                await ws.send_text(json.dumps({
                    "verdict": "buffering",
                    "frames_buffered": len(state.frame_buffer),
                    "clip_len": state.clip_len,
                }))
                continue

            temporal_t = torch.stack(
                [t.squeeze(0) for t in state.frame_buffer], dim=1
            ).unsqueeze(0).to(_device)

            with torch.no_grad():
                prob = torch.sigmoid(_model(spatial_t, temporal_t)).item()

            is_fake = prob > threshold
            spatial_score  = _model.last_spatial
            temporal_score = _model.last_temporal

            # Grad-CAM heatmap (only if requested + fake)
            heatmap_b64 = None
            if want_heatmap and is_fake:
                hm = get_combined_heatmap(
                    _model,
                    spatial_t.cpu(),
                    temporal_t.cpu(),
                    face_rgb,
                )
                hm_bgr = cv2.cvtColor(hm, cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".jpg", hm_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                heatmap_b64 = base64.b64encode(buf).decode()

            await ws.send_text(json.dumps({
                "verdict":  "fake" if is_fake else "real",
                "score":    round(prob, 4),
                "spatial":  round(spatial_score, 4),
                "temporal": round(temporal_score, 4),
                "bbox":     [x1, y1, x2, y2],
                "heatmap":  heatmap_b64,
            }))

    except WebSocketDisconnect:
        pass


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(_device)}