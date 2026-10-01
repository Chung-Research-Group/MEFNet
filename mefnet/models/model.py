"""MEFNet: fuses the charged crystal graph, electrostatic grid and cell volume
for classification or regression."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_encoder import GraphEncoder
from .grid_encoder import SmallGridEncoder


def pool_by_crystal(atom_tokens: torch.Tensor, crystal_atom_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Mean-pool atom features per crystal."""
    d = atom_tokens.shape[-1]
    pooled = atom_tokens.new_zeros(batch_size, d)
    counts = atom_tokens.new_zeros(batch_size, 1)
    pooled.index_add_(0, crystal_atom_idx, atom_tokens)
    counts.index_add_(0, crystal_atom_idx, atom_tokens.new_ones(atom_tokens.size(0), 1))
    return pooled / counts.clamp_min(1.0)


def extract_patch_channel_means(grid: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Per-patch channel means (target of the patch auxiliary loss)."""
    b, c, s1, s2, s3 = grid.shape
    if not (s1 == s2 == s3):
        raise ValueError(f'grid must be cubic; got {grid.shape}')
    if s1 % patch_size != 0:
        raise ValueError(f'grid size {s1} must be divisible by patch size {patch_size}')
    n = s1 // patch_size
    x = grid.view(b, c, n, patch_size, n, patch_size, n, patch_size)
    x = x.mean(dim=(3, 5, 7))
    x = x.reshape(b, c, n**3).transpose(1, 2).contiguous()
    return x


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class MPDHead(nn.Module):

    def __init__(self, embed_dim: int, condition_dim: int = 0, out_dim: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + self.condition_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, out_dim),
        )

    def forward(self, latent, condition=None):
        if self.condition_dim > 0:
            if condition is None:
                raise ValueError('MPDHead was built with condition_dim>0 '
                                 'but no condition was given')
            latent = torch.cat([latent, condition], dim=-1)
        return self.net(latent)


class GraphElectrostaticModel(nn.Module):
    """MEFNet with gate / gcsa (GCCA) / none (mean) grid pooling."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_graph = bool(getattr(config, 'use_graph', True))
        self.use_grid = bool(config.use_grid)
        self.use_volume = bool(getattr(config, 'use_volume', True))
        if not (self.use_graph or self.use_grid):
            raise ValueError('use_graph and use_grid cannot both be False')
        self.fusion_type = config.fusion_type

        if self.use_graph:
            self.graph_encoder = GraphEncoder(
                atom_init_fea_len=config.atom_init_fea_len,
                atom_fea_len=config.atom_fea_len,
                nbr_fea_len=config.nbr_fea_len,
                n_conv=config.n_conv,
                graph_hidden_dim=config.graph_hidden_dim,
                use_charge_embedding=config.use_charge_embedding,
                charge_embed_dim=config.charge_embed_dim,
            )
        else:
            self.graph_encoder = None

        if self.use_grid:
            self.grid_encoder = SmallGridEncoder(
                in_channels=config.in_channels,
                embed_dim=config.grid_hidden_dim,
                patch_size=config.patch_size,
                dropout=config.dropout,
            )
            if (not self.use_graph) or self.fusion_type == 'none':
                self.graph_gate = None
                self.grid_attn = None
                self.query_proj = None
            elif self.fusion_type == 'gate':
                self.graph_gate = nn.Sequential(nn.Linear(config.graph_hidden_dim, config.grid_hidden_dim), nn.Sigmoid())
                self.grid_attn = None
                self.query_proj = None
            elif self.fusion_type == 'gcsa':
                self.graph_gate = None
                self.grid_attn = nn.MultiheadAttention(
                    embed_dim=config.grid_hidden_dim,
                    num_heads=config.n_attn_heads,
                    batch_first=True,
                    dropout=config.dropout,
                )
                self.query_proj = nn.Linear(config.graph_hidden_dim, config.grid_hidden_dim)
            else:
                raise ValueError(
                    f"Unknown fusion_type={self.fusion_type}; "
                    "expected 'gate', 'gcsa' or 'none'")
            self.patch_recon_head = nn.Linear(config.grid_hidden_dim, config.in_channels)
        else:
            self.grid_encoder = None
            self.graph_gate = None
            self.grid_attn = None
            self.query_proj = None
            self.patch_recon_head = None

        fusion_in = 0
        if self.use_graph:
            fusion_in += config.graph_hidden_dim
        if self.use_grid:
            fusion_in += config.grid_hidden_dim
        if self.use_volume:
            fusion_in += config.graph_hidden_dim

        self.volume_proj = nn.Sequential(
            nn.Linear(1, config.graph_hidden_dim),
            nn.LayerNorm(config.graph_hidden_dim),
            nn.GELU(),
        ) if self.use_volume else None
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, config.embed_dim),
        )
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.task = getattr(config, 'task', 'classification')

        if self.task == 'regression':
            n_out = getattr(config, 'n_outputs', None)
            if not n_out:
                raise ValueError(
                    'config.n_outputs must be set for task="regression"; '
                    'create_dataloaders() fills it from the labels file, so call '
                    'it before building the model')
            self.mpd_head = MPDHead(config.embed_dim,
                                    getattr(config, 'condition_dim', 0),
                                    n_out, config.dropout)
            self.mpd_softmax = bool(getattr(config, 'mpd_softmax', True))
            self.classifier = None
        elif self.task == 'classification':
            self.classifier = ClassificationHead(config.embed_dim, config.n_classes, config.dropout)
            self.mpd_head = None
            self.mpd_softmax = False
        else:
            raise ValueError(f"Unknown task={self.task}; "
                             "expected 'classification' or 'regression'")

    def _fuse_grid(self, graph_summary, grid_tokens):
        if (not self.use_graph) or self.fusion_type == 'none':
            return grid_tokens, grid_tokens.mean(dim=1), None
        if self.fusion_type == 'gate':
            gate = self.graph_gate(graph_summary).unsqueeze(1)
            conditioned_grid_tokens = grid_tokens * gate
            grid_summary = conditioned_grid_tokens.mean(dim=1)
            attn_weights = None
        else:
            query = self.query_proj(graph_summary).unsqueeze(1)
            grid_summary, attn_weights = self.grid_attn(
                query=query,
                key=grid_tokens,
                value=grid_tokens,
                need_weights=True,
                average_attn_weights=True,
            )
            conditioned_grid_tokens = grid_tokens
            grid_summary = grid_summary.squeeze(1)
        return conditioned_grid_tokens, grid_summary, attn_weights

    def encode(self, atom_fea, charges, nbr_fea, nbr_fea_idx, crystal_atom_idx, grid, volume):
        """Return the fused latent vector and intermediate outputs."""
        batch_size = int(crystal_atom_idx.max().item()) + 1 if crystal_atom_idx.numel() \
            else grid.shape[0]
        out = {}
        parts = []

        if self.use_graph:
            atom_tokens = self.graph_encoder(atom_fea, charges, nbr_fea, nbr_fea_idx)
            graph_summary = pool_by_crystal(atom_tokens, crystal_atom_idx, batch_size)
            out['atom_tokens'] = atom_tokens
            out['graph_summary'] = graph_summary
            parts.append(graph_summary)
        else:
            graph_summary = None

        if self.use_grid:
            grid_tokens, _ = self.grid_encoder(grid)
            conditioned_grid_tokens, grid_summary, attn_weights = self._fuse_grid(
                graph_summary, grid_tokens)
            out.update({'grid_tokens': grid_tokens,
                        'conditioned_grid_tokens': conditioned_grid_tokens,
                        'grid_summary': grid_summary})
            if attn_weights is not None:
                out['attn_weights'] = attn_weights
            parts.append(grid_summary)

        if self.use_volume:
            volume_summary = self.volume_proj(torch.log10(volume.clamp_min(1.0)))
            out['volume_summary'] = volume_summary
            parts.append(volume_summary)

        fused = torch.cat(parts, dim=-1)
        latent = self.final_norm(self.fusion(fused))
        out['latent'] = latent
        return out

    def forward(self, atom_fea, charges, nbr_fea, nbr_fea_idx, crystal_atom_idx, grid, volume, return_aux=False, return_attention=False):
        encoded = self.encode(atom_fea, charges, nbr_fea, nbr_fea_idx, crystal_atom_idx, grid, volume)
        if self.task == 'regression':
            logits = self.mpd_head(encoded['latent'], None)
        else:
            logits = self.classifier(encoded['latent'])
        if not return_aux and not return_attention:
            return logits
        aux = {'logits': logits}
        if self.use_grid:
            aux['patch_recon_pred'] = self.patch_recon_head(encoded['grid_tokens'])
        if return_attention:
            aux['latent'] = encoded['latent']
            aux['atom_tokens'] = encoded['atom_tokens']
            aux['graph_summary'] = encoded['graph_summary']
            if 'attn_weights' in encoded:
                aux['attn_weights'] = encoded['attn_weights']
        return aux
