import torch
import torch.nn as nn
from torchvision.models import vit_b_16, resnet18, video

class CrossAttention(nn.Module):
    """
    Cross-attention mechanism to fuse spatial and temporal features.
    """
    def __init__(self, embed_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, spatial, temporal):
        # spatial: [batch_size, seq_len, embed_dim]
        # temporal: [batch_size, seq_len, embed_dim]
        
        # Cross-attention: Spatial queries attend to Temporal updates
        attn_out, _ = self.attn(spatial, temporal, temporal)
        spatial = self.norm1(spatial + attn_out)
        
        return spatial

class DeepfakeDetector(nn.Module):
    """
    Hybrid Deepfake Detector: ViT-B/16 (Spatial) + 3D-ResNet18 (Temporal) + Cross-Attention Fusion.
    """
    def __init__(self, num_classes=2):
        super(DeepfakeDetector, self).__init__()
        
        # Spatial: ViT-B/16
        self.vit = vit_b_16(weights='DEFAULT')
        self.vit.heads = nn.Identity() # Remove top classification head
        
        # Temporal: 3D-ResNet18
        self.resnet3d = video.r3d_18(weights='DEFAULT')
        self.resnet3d.fc = nn.Identity() # Remove top classification head
        
        # Fusion
        self.fusion_attention = CrossAttention(embed_dim=768) # 768 for ViT-B/16
        self.fc = nn.Linear(768 + 512, num_classes) # Final classifier
        
    def forward(self, x):
        # x: [batch_size, channels, frames, height, width]
        
        # Spatial processing (middle frame or average)
        # spatial_feat = self.vit(x[:, :, frame_idx, :, :])
        
        # Temporal processing
        # temporal_feat = self.resnet3d(x)
        
        # Placeholder for cross-attention and combined classification
        # fused = self.fusion_attention(spatial_feat, temporal_feat)
        # output = self.fc(fused)
        
        return torch.randn(x.size(0), 2) # Simulated output

if __name__ == "__main__":
    model = DeepfakeDetector()
    print(model)
