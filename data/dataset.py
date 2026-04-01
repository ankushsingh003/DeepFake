import  torch 
import numpy as np 
from torch.utils.data import WeightedRandomSampler , DataLoader
from data.dataset import RawVideoDataset
from data.augment import TRAIN_SPATIAL, VAL_SPATIAL, apply_clip_augmentations
 

# weightage sampler

def make_weighted_sampler(dataset: RawVideoDataset) -> WeightedRandomSampler:
    """
    Compute per-sample weights so each class is sampled equally,
    regardless of how many videos exist per class.
    """
    labels = [label for _, label in dataset.samples]
    class_counts = [labels.count(0), labels.count(1)]
    weights_per_class = [1.0 / c if c > 0 else 0.0 for c in class_counts]
    sample_weights = [weights_per_class[l] for l in labels]
 
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

 
def mixup_batch(
    spatial: torch.Tensor,
    temporal: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.3,
):
    """
    MixUp regularisation on a batch.
    Blends pairs of samples and their labels.
    alpha: Beta distribution parameter (higher = more mixing)
    """
    lam = np.random.beta(alpha, alpha)
    B = spatial.size(0)
    idx = torch.randperm(B)
 
    mixed_spatial  = lam * spatial  + (1 - lam) * spatial[idx]
    mixed_temporal = lam * temporal + (1 - lam) * temporal[idx]
    mixed_labels   = lam * labels   + (1 - lam) * labels[idx]
 
    return mixed_spatial, mixed_temporal, mixed_labels
 


# augmented dataset wrapper 




class AugmentedVideoDataset(RawVideoDataset):
    """
    Wraps RawVideoDataset and applies clip-level augmentations on-the-fly.
    is_train=True  → TRAIN_SPATIAL + temporal augments
    is_train=False → VAL_SPATIAL   + no augments
    """
 
    def __init__(self, *args, is_train: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_train = is_train
        self.spatial_transform = TRAIN_SPATIAL if is_train else VAL_SPATIAL
 
    def __getitem__(self, idx):
        path, label = self.samples[idx]
 
        frames_raw = self._extract_frames(path)
        frames_raw = self._crop_faces(frames_raw)
 
        # Apply clip augmentations (temporal flip, noise, compression)
        frames_raw = apply_clip_augmentations(frames_raw, is_train=self.is_train)
 
        # Spatial: single representative frame
        mid = len(frames_raw) // 2
        spatial = self.spatial_transform(frames_raw[mid])
 
        # Temporal clip
        temporal = self._build_temporal(frames_raw)
 
        return spatial, temporal, torch.tensor(label, dtype=torch.float32)



# ── DataLoader factory with weighted sampling ──────────────────────────────────
 
def get_augmented_dataloaders(
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
    use_mixup: bool = False,
):
    """
    Returns train/val/test DataLoaders with:
    - WeightedRandomSampler on train (handles class imbalance)
    - AugmentedVideoDataset on train
    - Plain RawVideoDataset on val/test
    """
    from torch.utils.data import random_split
 
    full = AugmentedVideoDataset(
        real_dir=real_dir,
        fake_dir=fake_dir,
        clip_len=clip_len,
        frame_skip=frame_skip,
        face_detect=face_detect,
        cache_dir=cache_dir,
        is_train=True,
    )
 
    n = len(full)
    n_val   = int(n * val_split)
    n_test  = int(n * test_split)
    n_train = n - n_val - n_test
 
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=g)
 
    # Weighted sampler for training subset
    train_labels = [full.samples[i][1] for i in train_ds.indices]
    counts = [train_labels.count(0), train_labels.count(1)]
    w_per_class = [1.0 / c if c > 0 else 0.0 for c in counts]
    sample_w = [w_per_class[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
 
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
 
    return train_loader, val_loader, test_loader