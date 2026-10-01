"""Charge-aware crystal graph convolution encoder (CGCNN-style)."""
import torch
import torch.nn as nn


class GraphConvLayer(nn.Module):

    """Gated graph convolution with charge embedding."""
    def __init__(self, atom_fea_len: int, nbr_fea_len: int, charge_embed_dim: int = 16,
                 use_charge_embedding: bool = True):
        super().__init__()
        self.use_charge_embedding = use_charge_embedding
        msg_in = atom_fea_len * 2 + nbr_fea_len
        if use_charge_embedding:
            msg_in += charge_embed_dim * 2

        self.fc_full = nn.Linear(msg_in, atom_fea_len * 2)
        self.bn1 = nn.BatchNorm1d(atom_fea_len * 2)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.softplus2 = nn.Softplus()

    def forward(self, atom_fea, charge_emb, nbr_fea, nbr_fea_idx):
        n_atoms, max_nbr = nbr_fea_idx.shape
        nbr_atom_fea = atom_fea[nbr_fea_idx]
        self_atom = atom_fea.unsqueeze(1).expand(n_atoms, max_nbr, atom_fea.size(-1))

        parts = [self_atom, nbr_atom_fea, nbr_fea]
        if self.use_charge_embedding:
            nbr_charge = charge_emb[nbr_fea_idx]
            self_charge = charge_emb.unsqueeze(1).expand(n_atoms, max_nbr, charge_emb.size(-1))
            parts.extend([self_charge, nbr_charge])

        total_nbr_fea = torch.cat(parts, dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, atom_fea.size(-1) * 2)
        ).view(n_atoms, max_nbr, atom_fea.size(-1) * 2)

        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        return self.softplus2(atom_fea + nbr_sumed)


class GraphEncoder(nn.Module):

    """Stack of charge-aware graph convolutions."""
    def __init__(
        self,
        atom_init_fea_len: int = 92,
        atom_fea_len: int = 64,
        nbr_fea_len: int = 41,
        n_conv: int = 3,
        graph_hidden_dim: int = 256,
        use_charge_embedding: bool = True,
        charge_embed_dim: int = 16,
    ):
        super().__init__()
        self.use_charge_embedding = use_charge_embedding
        self.charge_embed_dim = charge_embed_dim

        self.embedding = nn.Linear(atom_init_fea_len, atom_fea_len)
        if use_charge_embedding:
            self.charge_proj = nn.Sequential(
                nn.Linear(1, charge_embed_dim),
                nn.GELU(),
                nn.Linear(charge_embed_dim, charge_embed_dim),
            )
        else:
            self.charge_proj = None

        self.convs = nn.ModuleList([
            GraphConvLayer(
                atom_fea_len=atom_fea_len,
                nbr_fea_len=nbr_fea_len,
                charge_embed_dim=charge_embed_dim,
                use_charge_embedding=use_charge_embedding,
            )
            for _ in range(n_conv)
        ])
        self.out_proj = nn.Linear(atom_fea_len, graph_hidden_dim)

    def forward(self, atom_fea, charges, nbr_fea, nbr_fea_idx):
        atom_fea = self.embedding(atom_fea)
        charge_emb = None
        if self.use_charge_embedding:
            charge_emb = self.charge_proj(charges.unsqueeze(-1))

        for conv in self.convs:
            atom_fea = conv(atom_fea, charge_emb, nbr_fea, nbr_fea_idx)

        atom_tokens = self.out_proj(atom_fea)
        return atom_tokens
