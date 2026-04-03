"""
models/gradcam.py
Grad-CAM heatmaps for the DeepfakeDetector.

Two targets:
  SpatialGradCAM  — runs on ViT last encoder block → shows forged face regions
  TemporalGradCAM — runs on 3D-ResNet last conv layer → shows temporal artifacts

Usage:
    from models.gradcam import SpatialGradCAM
    cam = SpatialGradCAM(model)
    heatmap = cam(spatial_tensor, temporal_tensor)  # np.ndarray (H, W, 3) uint8
"""

import cv2
import torch
import numpy as np
from typing import Optional


# ── Hook-based Grad-CAM base ───────────────────────────────────────────────────

class GradCAMBase:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients   = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, input, output):
            self._activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def _compute_cam(self) -> np.ndarray:
        raise NotImplementedError

    def _overlay(
        self,
        cam: np.ndarray,
        face_rgb: np.ndarray,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Blend CAM heatmap onto the face crop. Returns uint8 RGB."""
        h, w = face_rgb.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        heatmap = cv2.applyColorMap((cam_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        blended = (alpha * heatmap + (1 - alpha) * face_rgb).astype(np.uint8)
        return blended


# ── Spatial Grad-CAM (ViT) ────────────────────────────────────────────────────

class SpatialGradCAM(GradCAMBase):
    """
    Grad-CAM on the last ViT encoder block.
    Highlights which face patches contributed most to the fake score.
    """

    def __init__(self, model):
        target = model.vit.encoder.layer[-1]
        super().__init__(model, target)

    def __call__(
        self,
        spatial: torch.Tensor,    # (1, 3, 224, 224)
        temporal: torch.Tensor,   # (1, 3, T, 224, 224)
        face_rgb: Optional[np.ndarray] = None,  # original face crop for overlay
    ) -> np.ndarray:
        """
        Returns:
            If face_rgb is given → blended heatmap (H, W, 3) uint8
            Otherwise           → raw grayscale CAM (14, 14) float32
        """
        self.model.eval()
        spatial  = spatial.requires_grad_(True)
        temporal = temporal.detach()

        logits = self.model(spatial, temporal)
        self.model.zero_grad()
        logits.backward()

        cam = self._compute_cam()

        if face_rgb is not None:
            return self._overlay(cam, face_rgb)
        return cam

    def _compute_cam(self) -> np.ndarray:
        # activations: (1, num_patches+1, hidden)  → drop CLS token
        acts = self._activations[0, 1:, :]        # (196, 768)
        grads = self._gradients[0, 1:, :]          # (196, 768)

        # Weight each patch's activation by its gradient
        weights = grads.mean(dim=-1)               # (196,)
        cam = (weights * acts.mean(dim=-1)).relu() # (196,)

        # Reshape to 14×14 spatial grid (ViT-B/16 on 224px → 14×14 patches)
        side = int(cam.shape[0] ** 0.5)
        cam = cam.reshape(side, side).cpu().numpy()
        return cam


# ── Temporal Grad-CAM (3D-ResNet) ─────────────────────────────────────────────

class TemporalGradCAM(GradCAMBase):
    """
    Grad-CAM on the last 3D conv block of the temporal branch.
    Highlights which frames carry the most artifact signal.
    """

    def __init__(self, model):
        target = model.temporal.layer4[-1]
        super().__init__(model, target)

    def __call__(
        self,
        spatial: torch.Tensor,
        temporal: torch.Tensor,
        face_rgb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.model.eval()
        temporal = temporal.requires_grad_(True)
        spatial  = spatial.detach()

        logits = self.model(spatial, temporal)
        self.model.zero_grad()
        logits.backward()

        cam = self._compute_cam()

        if face_rgb is not None:
            return self._overlay(cam, face_rgb)
        return cam

    def _compute_cam(self) -> np.ndarray:
        # activations: (1, C, T, H, W)
        acts  = self._activations[0]   # (C, T, H, W)
        grads = self._gradients[0]     # (C, T, H, W)

        # Average over time → spatial map
        weights = grads.mean(dim=(1, 2, 3))   # (C,)
        cam = torch.einsum("c,cthw->hw", weights, acts).relu()
        cam = cam.cpu().numpy()
        return cam


# ── Combined overlay ───────────────────────────────────────────────────────────

def get_combined_heatmap(
    model,
    spatial: torch.Tensor,
    temporal: torch.Tensor,
    face_rgb: np.ndarray,
    spatial_weight: float = 0.6,
    temporal_weight: float = 0.4,
) -> np.ndarray:
    """
    Blend spatial + temporal Grad-CAM into one overlay.
    Gives a richer view of what the model is attending to.
    """
    s_cam = SpatialGradCAM(model)(spatial, temporal)
    t_cam = TemporalGradCAM(model)(spatial, temporal)

    h, w = face_rgb.shape[:2]
    s_cam = cv2.resize(s_cam, (w, h))
    t_cam = cv2.resize(t_cam, (w, h))

    # Normalise each map to [0, 1]
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    combined = spatial_weight * norm(s_cam) + temporal_weight * norm(t_cam)
    combined = norm(combined)

    heatmap = cv2.applyColorMap((combined * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (0.45 * heatmap + 0.55 * face_rgb).astype(np.uint8)
    return blended