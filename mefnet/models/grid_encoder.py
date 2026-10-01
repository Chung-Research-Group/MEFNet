"""3D patch-embedding encoder for the electrostatic grid."""
import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """Split the grid into cubic patches and embed each patch."""
    def __init__(self, in_channels=4, embed_dim=128, patch_size=5):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SmallGridEncoder(nn.Module):

    """Patch embedding followed by a token MLP."""
    def __init__(self, in_channels=4, embed_dim=128, patch_size=5, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, grid):
        tokens = self.patch_embed(grid)
        tokens = self.norm(tokens)
        tokens = tokens + self.mlp(tokens)
        tokens = self.out_norm(tokens)
        pooled = tokens.mean(dim=1)
        return tokens, pooled
