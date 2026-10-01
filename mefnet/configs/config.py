"""Default MEFNet hyperparameters, ablation switches and data paths."""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    """All model, training and data settings."""
    embed_dim: int = 256
    graph_hidden_dim: int = 128
    grid_hidden_dim: int = 256
    fusion_hidden_dim: int = 128
    dropout: float = 0.20

    atom_init_fea_len: int = 92
    atom_fea_len: int = 128
    nbr_fea_len: int = 41
    n_conv: int = 3
    use_charge_embedding: bool = True
    charge_embed_dim: int = 16
    use_graph: bool = True
    use_volume: bool = True

    use_grid: bool = True
    in_channels: int = 4
    grid_size: int = 30
    patch_size: int = 5

    fusion_type: str = "none"
    n_attn_heads: int = 8

    task: str = "classification"
    n_classes: int = 2

    n_outputs: Optional[int] = None
    target_cols: Optional[List[str]] = None
    mpd_softmax: bool = True
    condition_dim: int = 0

    lr: float = 4.26e-5
    weight_decay: float = 2.09e-1
    max_epochs: int = 300
    batch_size: int = 32
    patience: int = 30
    grad_clip: float = 1.0
    random_seed: int = 42
    use_amp: bool = False

    cls_weight: float = 1.0
    patch_aux_weight: float = 0.20

    class_counts: Optional[List[int]] = None

    graph_dir: str = "data/graphs"
    grid_dir: str = "data/grids"
    label_file: str = "data/labels.csv"
    split_file: str = "data/split.json"
    group_file: Optional[str] = None
    result_dir: str = "results"
    num_workers: int = 2
    pin_memory: bool = True

    persistent_workers: bool = False
    prefetch_factor: int = 2

    @property
    def class_weights(self):
        """Inverse-frequency class weights computed from the training labels."""
        if not self.class_counts or min(self.class_counts) == 0:
            raise ValueError(f"every class needs at least one training structure; "
                             f"counts = {self.class_counts}")
        total = sum(self.class_counts)
        return [total / (len(self.class_counts) * n) for n in self.class_counts]

    @classmethod
    def from_dict(cls, d):
        """Build a Config from a saved config.json, ignoring unknown keys."""
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})
