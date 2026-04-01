
"""
data/download.py
Downloads raw .mp4 files from:
  - yt-dlp  → data/raw/real/
  - FaceForensics++ script → data/raw/fake/
  - Kaggle DFDC  → data/raw/fake/
No CSVs. Labels are folder-based.
"""
 
import os
import subprocess
import argparse
import logging
from pathlib import Path
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/logs/download.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)
 
REAL_DIR = Path("data/raw/real")
FAKE_DIR = Path("data/raw/fake")
 
# ── yt-dlp search queries for REAL talking-head videos ────────────────────────
REAL_QUERIES = [
    "ytsearch50:interview talking head camera 2023",
    "ytsearch50:news anchor broadcast closeup",
    "ytsearch50:ted talk speaker face",
    "ytsearch50:podcast video face cam 2023",
    "ytsearch50:vlog face closeup natural lighting",
    "ytsearch50:documentary interview subject",
    "ytsearch50:university lecture professor face",
    "ytsearch50:press conference speaker podium",
]
 
 
def download_real_videos(max_per_query: int = 50, max_duration: int = 300):
    """
    Scrape real talking-head .mp4s via yt-dlp.
    max_per_query : videos per search query
    max_duration  : skip videos longer than N seconds
    """
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading REAL videos → {REAL_DIR}")
 
    for query in REAL_QUERIES:
        log.info(f"  Query: {query}")
        cmd = [
            "yt-dlp",
            query,
            "--format", "bestvideo[ext=mp4][height<=720]+bestaudio/best[ext=mp4]",
            "--output", str(REAL_DIR / "%(id)s.%(ext)s"),
            "--max-downloads", str(max_per_query),
            "--no-playlist",
            "--match-filter", f"duration < {max_duration}",
            "--ignore-errors",
            "--quiet",
            "--no-warnings",
        ]
        subprocess.run(cmd, check=False)
 
    count = len(list(REAL_DIR.glob("*.mp4")))
    log.info(f"  Done. {count} real .mp4 files in {REAL_DIR}")
 
 
def download_faceforensics(data_root: str, ff_script: str = "download_FaceForensics.py"):
    """
    Download FaceForensics++ fake videos.
 
    Steps:
      1. git clone https://github.com/ondyari/FaceForensics
      2. Fill the access request form linked in the repo to get a password
      3. Run this function (it calls their official download script)
 
    Args:
        data_root  : where to store FF++ videos
        ff_script  : path to their download_FaceForensics.py
    """
    if not os.path.exists(ff_script):
        log.error(
            "FaceForensics download script not found.\n"
            "Run: git clone https://github.com/ondyari/FaceForensics\n"
            "Then request access at: https://github.com/ondyari/FaceForensics#access"
        )
        return
 
    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading FaceForensics++ fakes → {FAKE_DIR}")
 
    # Downloads all 4 manipulation methods at c23 (light) compression
    cmd = [
        "python", ff_script,
        data_root,
        "-d", "all",        # DeepFakes, Face2Face, FaceSwap, NeuralTextures
        "-c", "c23",        # c23 = light compression, c40 = heavy
        "-t", "videos",     # raw .mp4 files (not images)
    ]
    subprocess.run(cmd, check=True)
    log.info("FaceForensics++ download complete.")
 
 
def download_dfdc(subset: int = 0):
    """
    Download DFDC (Deepfake Detection Challenge) videos from Kaggle.
 
    Setup:
      pip install kaggle
      Place API key at ~/.kaggle/kaggle.json
      Accept competition rules at:
        https://www.kaggle.com/competitions/deepfake-detection-challenge
 
    Args:
        subset: which part to download (0–49). Each part ~10GB.
    """
    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading DFDC part {subset} → {FAKE_DIR}")
 
    fname = f"dfdc_train_part_{subset}.zip"
    cmd = [
        "kaggle", "competitions", "download",
        "-c", "deepfake-detection-challenge",
        "-f", fname,
        "-p", str(FAKE_DIR),
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.error(
            "Kaggle download failed.\n"
            "Make sure ~/.kaggle/kaggle.json exists and you've accepted\n"
            "competition rules at kaggle.com/competitions/deepfake-detection-challenge"
        )
        return
 
    # Unzip in place
    subprocess.run(["unzip", "-q", str(FAKE_DIR / fname), "-d", str(FAKE_DIR)])
    os.remove(FAKE_DIR / fname)
    log.info(f"DFDC part {subset} extracted to {FAKE_DIR}")
 
 
def record_webcam(output_name: str = "webcam_001.mp4", duration: int = 60):
    """
    Record a real video directly from the webcam.
    Saves to data/raw/real/
    """
    import cv2
    out_path = REAL_DIR / output_name
    REAL_DIR.mkdir(parents=True, exist_ok=True)
 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open webcam.")
        return
 
    fps = 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
 
    log.info(f"Recording webcam for {duration}s → {out_path}")
    total = fps * duration
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        if i % fps == 0:
            log.info(f"  {i // fps}s / {duration}s")
 
    cap.release()
    writer.release()
    log.info(f"Webcam recording saved: {out_path}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download deepfake detector data")
    parser.add_argument("--real",       action="store_true", help="Download real videos via yt-dlp")
    parser.add_argument("--ff",         action="store_true", help="Download FaceForensics++")
    parser.add_argument("--dfdc",       type=int, default=-1, metavar="PART", help="Download DFDC part N (0-49)")
    parser.add_argument("--webcam",     action="store_true", help="Record from webcam")
    parser.add_argument("--all",        action="store_true", help="Run all downloads")
    parser.add_argument("--max-videos", type=int, default=50, help="Max videos per yt-dlp query")
    parser.add_argument("--duration",   type=int, default=300, help="Max video duration (seconds)")
    parser.add_argument("--ff-script",  type=str, default="FaceForensics/download_FaceForensics.py")
    parser.add_argument("--ff-root",    type=str, default="data/raw/fake/ff++")
    args = parser.parse_args()
 
    os.makedirs("data/logs", exist_ok=True)
 
    if args.all or args.real:
        download_real_videos(max_per_query=args.max_videos, max_duration=args.duration)
    if args.all or args.ff:
        download_faceforensics(args.ff_root, args.ff_script)
    if args.dfdc >= 0:
        download_dfdc(subset=args.dfdc)
    if args.all or args.webcam:
        record_webcam()
 
