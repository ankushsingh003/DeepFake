"""
data/dataset.py
PyTorch Dataset that reads raw .mp4 files directly.
Labels come from folder names (real/ fake/) — no CSV needed.
 
Directory layout expected:
  data/raw/real/*.mp4   →  label 0
  data/raw/fake/*.mp4   →  label 1
"""
 
import os
import cv2
import torch
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from retinaface import RetinaFace
 
log = logging.getLogger(__name__)
 
# ── Normalization (ImageNet stats — matches ViT pretrain) ─────────────────────
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
 
SPATIAL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE,
])
 
 
class RawVideoDataset(Dataset):
    """
    Reads raw .mp4 files. No CSV. Labels = folder name.
      real/ → 0
      fake/ → 1
 
    Args:
        real_dir   : path to folder of real .mp4 files
        fake_dir   : path to folder of fake .mp4 files
        clip_len   : number of frames per clip for temporal branch
        frame_skip : sample every N-th frame (reduces redundancy)
        face_detect: whether to run RetinaFace crop (slower but accurate)
        cache_dir  : if set, cache preprocessed tensors as .pt files
    """
 
    def __init__(
        self,
        real_dir: str = "data/raw/real",
        fake_dir: str = "data/raw/fake",
        clip_len: int = 16,
        frame_skip: int = 3,
        face_detect: bool = True,
        cache_dir: str = "data/cache",
    ):
        self.clip_len = clip_len
        self.frame_skip = frame_skip
        self.face_detect = face_detect
        self.cache_dir = Path(cache_dir) if cache_dir else None
 
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
 
        # Build file list — labels from directory, NOT from a CSV
        self.samples = []
        for path in sorted(Path(real_dir).glob("*.mp4")):
            self.samples.append((str(path), 0))
        for path in sorted(Path(fake_dir).glob("*.mp4")):
            self.samples.append((str(path), 1))
 
        real_count = sum(1 for _, l in self.samples if l == 0)
        fake_count = sum(1 for _, l in self.samples if l == 1)
        log.info(f"Dataset: {real_count} real | {fake_count} fake | total {len(self.samples)}")
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        path, label = self.samples[idx]
 
        # Check cache first
        cache_key = Path(path).stem + f"_cl{self.clip_len}_fs{self.frame_skip}"
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pt"
            if cache_file.exists():
                data = torch.load(cache_file)
                return data["spatial"], data["temporal"], torch.tensor(label, dtype=torch.float32)
 
        # Read raw .mp4
        frames_raw = self._extract_frames(path)
 
        # Face detection and crop
        if self.face_detect:
            frames_raw = self._crop_faces(frames_raw)
 
        # Build spatial tensor (single representative frame)
        mid = len(frames_raw) // 2
        spatial = SPATIAL_TRANSFORM(frames_raw[mid])  # (3, 224, 224)
 
        # Build temporal tensor (clip_len frames)
        temporal = self._build_temporal(frames_raw)   # (3, T, 224, 224)
 
        # Cache to disk
        if self.cache_dir:
            torch.save({"spatial": spatial, "temporal": temporal}, cache_file)
 
        return spatial, temporal, torch.tensor(label, dtype=torch.float32)
 
    # ── Internal helpers ───────────────────────────────────────────────────────
 
    def _extract_frames(self, video_path: str) -> list:
        """Read every frame_skip-th frame from .mp4 as RGB numpy arrays."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.warning(f"Cannot open video: {video_path}")
            return [np.zeros((224, 224, 3), dtype=np.uint8)]
 
        frames = []
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % self.frame_skip == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            i += 1
        cap.release()
 
        if not frames:
            log.warning(f"No frames extracted from: {video_path}")
            return [np.zeros((224, 224, 3), dtype=np.uint8)]
 
        return frames
 
    def _crop_faces(self, frames: list) -> list:
        """
        Detect face in the middle frame, crop that region across all frames.
        Falls back to full frame if no face found.
        """
        mid = frames[len(frames) // 2]
        try:
            faces = RetinaFace.detect_faces(mid)
        except Exception:
            faces = {}
 
        if not faces:
            return frames  # no face found — use full frame
 
        # Pick the largest face
        best = max(
            faces.values(),
            key=lambda f: (
                (f["facial_area"][2] - f["facial_area"][0]) *
                (f["facial_area"][3] - f["facial_area"][1])
            )
        )
        x1, y1, x2, y2 = best["facial_area"]
 
        # Add 20% padding
        h, w = mid.shape[:2]
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
 
        return [f[y1:y2, x1:x2] for f in frames]
 
    def _build_temporal(self, frames: list) -> torch.Tensor:
        """
        Sample exactly clip_len frames uniformly from the video.
        Returns shape (3, T, H, W) for 3D-ResNet input.
        """
        total = len(frames)
 
        if total >= self.clip_len:
            indices = np.linspace(0, total - 1, self.clip_len, dtype=int)
        else:
            # Repeat last frame if video is short
            indices = list(range(total)) + [total - 1] * (self.clip_len - total)
 
        clip_frames = []
        for i in indices:
            frame = frames[i]
            frame = cv2.resize(frame, (224, 224))
            t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            t = NORMALIZE(t)
            clip_frames.append(t)
 
        # Stack → (T, 3, H, W) → permute → (3, T, H, W)
        return torch.stack(clip_frames).permute(1, 0, 2, 3)
 
 
# ── Convenience function ───────────────────────────────────────────────────────
 
def get_dataloaders(
    real_dir: str = "data/raw/real",
    fake_dir: str = "data/raw/fake",
    batch_size: int = 8,
    clip_len: int = 16,
    frame_skip: int = 3,
    face_detect: bool = True,
    cache_dir: str = "data/cache",
    val_split: float = 0.15,
    test_split: float = 0.10,
    num_workers: int = 4,
    seed: int = 42,
):
    """
    Build train / val / test DataLoaders from raw .mp4 directories.
    No CSV. Splits are random but reproducible via seed.
    """
    full_dataset = RawVideoDataset(
        real_dir=real_dir,
        fake_dir=fake_dir,
        clip_len=clip_len,
        frame_skip=frame_skip,
        face_detect=face_detect,
        cache_dir=cache_dir,
    )
 
    n = len(full_dataset)
    n_val  = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test
 
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )
 
    log.info(f"Split → train: {n_train} | val: {n_val} | test: {n_test}")
 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
 
    return train_loader, val_loader, test_loader
 
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train, val, test = get_dataloaders(batch_size=4, face_detect=False)
    spatial, temporal, labels = next(iter(train))
    print("Spatial shape :", spatial.shape)   # (4, 3, 224, 224)
    print("Temporal shape:", temporal.shape)  # (4, 3, 16, 224, 224)
    print("Labels        :", labels)          # tensor([0., 1., 0., 1.])
 