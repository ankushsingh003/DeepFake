"""
data/augment.py
Augmentation strategies for deepfake detection.
Applied on raw frames BEFORE tensor conversion.
Temporal augmentations work on the full clip (list of frames).
"""
 
import cv2
import random
import numpy as np
import torch
from torchvision import transforms
 
 
# ── Spatial augmentations (applied per-frame) ─────────────────────────────────
 
TRAIN_SPATIAL = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
VAL_SPATIAL = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
 
# ── Temporal augmentations (applied on clip — list of np frames) ──────────────
 
def temporal_flip(frames: list) -> list:
    """Reverse the frame order (time flip)."""
    return frames[::-1]
 
 
def temporal_dropout(frames: list, drop_prob: float = 0.1) -> list:
    """Randomly replace frames with neighbours (simulates frame drops)."""
    result = frames.copy()
    for i in range(1, len(frames) - 1):
        if random.random() < drop_prob:
            result[i] = result[i - 1]
    return result
 
 
def add_compression_artifacts(frame: np.ndarray, quality: int = None) -> np.ndarray:
    """
    Re-compress frame as JPEG to simulate video compression.
    Deepfake artifacts are often visible at high compression.
    """
    if quality is None:
        quality = random.randint(50, 95)
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buf = cv2.imencode(".jpg", frame, encode_param)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
 
 
def add_gaussian_noise(frame: np.ndarray, std: float = None) -> np.ndarray:
    """Add gaussian noise to simulate low-light / sensor noise."""
    if std is None:
        std = random.uniform(0, 15)
    noise = np.random.normal(0, std, frame.shape).astype(np.int16)
    noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy
 
 
def random_blur(frame: np.ndarray) -> np.ndarray:
    """Light gaussian blur to simulate motion or focus issues."""
    k = random.choice([3, 5])
    return cv2.GaussianBlur(frame, (k, k), 0)
 
 
def apply_clip_augmentations(frames: list, is_train: bool = True) -> list:
    """
    Apply temporal + per-frame augmentations to a clip.
    frames: list of RGB numpy arrays
    Returns: augmented list of RGB numpy arrays
    """
    if not is_train:
        return frames
 
    # Temporal augmentations (whole clip)
    if random.random() < 0.3:
        frames = temporal_flip(frames)
    if random.random() < 0.2:
        frames = temporal_dropout(frames, drop_prob=0.1)
 
    # Consistent per-frame augmentation (same params for whole clip)
    apply_compress = random.random() < 0.4
    apply_noise    = random.random() < 0.3
    apply_blur     = random.random() < 0.2
    compress_q     = random.randint(40, 90) if apply_compress else 95
    noise_std      = random.uniform(2, 12) if apply_noise else 0
 
    augmented = []
    for frame in frames:
        f = frame.copy()
        if apply_compress:
            f = add_compression_artifacts(f, quality=compress_q)
        if apply_noise and noise_std > 0:
            f = add_gaussian_noise(f, std=noise_std)
        if apply_blur:
            f = random_blur(f)
        augmented.append(f)
 
    return augmented
 
 
# ── CutOut augmentation on tensors ────────────────────────────────────────────
 
class CutOut:
    """
    Randomly mask a square patch of the image to zero.
    Helps model not over-rely on specific face regions.
    """
    def __init__(self, n_holes: int = 1, length: int = 56):
        self.n_holes = n_holes
        self.length = length
 
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        mask = torch.ones_like(img)
        for _ in range(self.n_holes):
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            x1 = max(0, cx - self.length // 2)
            x2 = min(w, cx + self.length // 2)
            y1 = max(0, cy - self.length // 2)
            y2 = min(h, cy + self.length // 2)
            mask[:, y1:y2, x1:x2] = 0
        return img * mask
 
 
TRAIN_SPATIAL_WITH_CUTOUT = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    CutOut(n_holes=1, length=56),
])