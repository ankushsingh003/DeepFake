"""
models/detector.py
DeepfakeDetector — dual-branch architecture:
  Spatial  : ViT-B/16  (patch-level features from a single frame)
  Temporal : 3D-ResNet18 (motion patterns across a 16-frame clip)
  Fusion   : Cross-attention  →  MLP classifier head
"""

import torch
import torch.nn as nn
from transformers import ViTModel
from torchvision.models.video import r3d_18


class DeepfakeDetector(nn.Module):
    """
    Args:
        vit_name     : HuggingFace ViT checkpoint
        embed_dim    : shared fusion embedding dimension
        num_heads    : attention heads in cross-attention
        dropout      : dropout in classifier head
        freeze_vit   : freeze ViT weights (useful for fine-tuning on small data)
    """

    def __init__(
        self,
        vit_name: str = "google/vit-base-patch16-224",
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.4,
        freeze_vit: bool = False,
    ):
        super().__init__()

        # ── Spatial branch — Vision Transformer ───────────────────────────────
        self.vit = ViTModel.from_pretrained(vit_name)
        vit_hidden = self.vit.config.hidden_size  # 768 for ViT-B

        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.spatial_proj = nn.Sequential(
            nn.Linear(vit_hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ── Temporal branch — 3D ResNet ────────────────────────────────────────
        self.temporal = r3d_18(pretrained=True)
        temporal_hidden = self.temporal.fc.in_features  # 512
        self.temporal.fc = nn.Identity()

        self.temporal_proj = nn.Sequential(
            nn.Linear(temporal_hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ── Cross-attention fusion ─────────────────────────────────────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # ── Classifier head ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

        # Store intermediate features for Grad-CAM
        self.last_spatial_feat = None
        self.last_temporal_feat = None

    def forward(
        self,
        spatial: torch.Tensor,    # (B, 3, 224, 224)
        temporal: torch.Tensor,   # (B, 3, T, 224, 224)
    ) -> torch.Tensor:            # (B,) raw logits

        # Spatial: ViT CLS token
        vit_out = self.vit(pixel_values=spatial)
        s_feat = vit_out.last_hidden_state[:, 0, :]   # (B, 768)
        s_feat = self.spatial_proj(s_feat)             # (B, embed_dim)
        self.last_spatial_feat = s_feat.detach()

        # Temporal: 3D-ResNet global avg pool
        t_feat = self.temporal(temporal)               # (B, 512)
        t_feat = self.temporal_proj(t_feat)            # (B, embed_dim)
        self.last_temporal_feat = t_feat.detach()

        # Cross-attention: spatial queries temporal
        s = s_feat.unsqueeze(1)                        # (B, 1, embed_dim)
        t = t_feat.unsqueeze(1)                        # (B, 1, embed_dim)
        fused, _ = self.cross_attn(query=s, key=t, value=t)
        fused = self.attn_norm(fused.squeeze(1) + s_feat)  # residual

        logits = self.classifier(fused).squeeze(-1)    # (B,)
        return logits

    def predict_proba(
        self,
        spatial: torch.Tensor,
        temporal: torch.Tensor,
    ) -> torch.Tensor:
        """Returns fake probability in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(spatial, temporal)
        return torch.sigmoid(logits)

    @property
    def last_spatial(self) -> float:
        if self.last_spatial_feat is None:
            return 0.0
        return float(torch.sigmoid(self.last_spatial_feat.mean()).item())

    @property
    def last_temporal(self) -> float:
        if self.last_temporal_feat is None:
            return 0.0
        return float(torch.sigmoid(self.last_temporal_feat.mean()).item())