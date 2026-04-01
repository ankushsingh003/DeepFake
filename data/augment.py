import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

def get_spatial_transforms(image_size=224):
    """
    Returns spatial augmentation transforms.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.FancyPCA(p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def temporal_augmentation(frames):
    """
    Simulates temporal augmentation by skipping or repeating frames.
    """
    # Sample logic for temporal augmentation
    # ...
    return frames
