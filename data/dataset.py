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
 