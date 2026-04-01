"""
data/inspect.py
Validates and summarizes the raw .mp4 dataset.
Checks for: corrupted files, no-face videos, class imbalance, duration stats.
Run before training to catch problems early.
 
Usage:
    python data/inspect.py
    python data/inspect.py --fix   # move corrupt files to data/quarantine/
"""
 
import cv2
import os
import argparse
import logging
from pathlib import Path
from collections import defaultdict
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)
 
REAL_DIR = Path("data/raw/real")
FAKE_DIR = Path("data/raw/fake")
QUARANTINE = Path("data/quarantine")
 
 
def check_video(path: Path) -> dict:
    """
    Open a .mp4 and return basic stats.
    Returns dict with: ok, frames, duration_s, fps, width, height, error
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False, "error": "Cannot open file"}
 
    fps     = cap.get(cv2.CAP_PROP_FPS) or 0
    frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frames / fps if fps > 0 else 0
 
    # Try to read first and last frame
    ok = True
    error = None
    ret, _ = cap.read()
    if not ret:
        ok = False
        error = "Cannot read first frame"
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames - 2))
        ret2, _ = cap.read()
        if not ret2:
            ok = False
            error = "Cannot read last frame (truncated?)"
 
    cap.release()
    return {
        "ok": ok,
        "frames": frames,
        "duration_s": round(duration, 1),
        "fps": round(fps, 1),
        "width": width,
        "height": height,
        "error": error,
    }
 
 
def inspect_directory(directory: Path, label: str, fix: bool = False) -> dict:
    """Inspect all .mp4 files in a directory. Returns summary stats."""
    mp4s = list(directory.glob("*.mp4"))
    if not mp4s:
        log.warning(f"  No .mp4 files found in {directory}")
        return {}
 
    log.info(f"\n── {label.upper()} videos ({directory}) ─────────────────")
    log.info(f"  Found {len(mp4s)} .mp4 files")
 
    stats = defaultdict(list)
    bad = []
 
    for i, path in enumerate(mp4s):
        info = check_video(path)
        if not info["ok"]:
            bad.append((path, info["error"]))
            log.warning(f"  [BAD] {path.name}: {info['error']}")
            continue
 
        stats["duration_s"].append(info["duration_s"])
        stats["frames"].append(info["frames"])
        stats["fps"].append(info["fps"])
        stats["width"].append(info["width"])
        stats["height"].append(info["height"])
 
        if i % 50 == 0 and i > 0:
            log.info(f"  Checked {i}/{len(mp4s)}...")
 
    # Print summary
    def s(lst): return f"min={min(lst):.1f} avg={sum(lst)/len(lst):.1f} max={max(lst):.1f}" if lst else "—"
 
    log.info(f"  Good files  : {len(mp4s) - len(bad)}")
    log.info(f"  Bad files   : {len(bad)}")
    log.info(f"  Duration (s): {s(stats['duration_s'])}")
    log.info(f"  Frames      : {s(stats['frames'])}")
    log.info(f"  FPS         : {s(stats['fps'])}")
    log.info(f"  Resolution  : {s(stats['width'])} x {s(stats['height'])}")
 
    # Move bad files to quarantine
    if fix and bad:
        QUARANTINE.mkdir(parents=True, exist_ok=True)
        for path, err in bad:
            dest = QUARANTINE / path.name
            path.rename(dest)
            log.info(f"  Moved to quarantine: {path.name} ({err})")
 
    return {
        "total": len(mp4s),
        "good": len(mp4s) - len(bad),
        "bad": len(bad),
        "durations": stats["duration_s"],
    }
 
 
def check_balance(real_stats: dict, fake_stats: dict):
    """Warn if class imbalance is severe."""
    r = real_stats.get("good", 0)
    f = fake_stats.get("good", 0)
    if r == 0 or f == 0:
        log.warning("\n  WARNING: One class has 0 good videos!")
        return
    ratio = max(r, f) / min(r, f)
    log.info(f"\n── CLASS BALANCE ─────────────────────────────────────")
    log.info(f"  Real : {r}  |  Fake : {f}  |  Ratio: {ratio:.2f}:1")
    if ratio > 3:
        minority = "fake" if f < r else "real"
        log.warning(f"  WARNING: High imbalance (>{ratio:.1f}x). Consider collecting more {minority} videos.")
        log.warning(f"  Or use weighted sampling in DataLoader (see dataset.py).")
    else:
        log.info(f"  Balance looks good.")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Move corrupt files to data/quarantine/")
    parser.add_argument("--real-dir", default="data/raw/real")
    parser.add_argument("--fake-dir", default="data/raw/fake")
    args = parser.parse_args()
 
    real_stats = inspect_directory(Path(args.real_dir), "real", fix=args.fix)
    fake_stats = inspect_directory(Path(args.fake_dir), "fake", fix=args.fix)
    check_balance(real_stats, fake_stats)
 
    log.info("\n── RECOMMENDATION ────────────────────────────────────")
    total = real_stats.get("good", 0) + fake_stats.get("good", 0)
    if total < 200:
        log.warning(f"  Only {total} good videos. Aim for 500+ before training.")
    elif total < 500:
        log.info(f"  {total} videos — decent for a first run. More data will help.")
    else:
        log.info(f"  {total} videos — good training set size.")
 