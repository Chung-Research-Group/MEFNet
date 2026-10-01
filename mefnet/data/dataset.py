"""Dataset, collate function and DataLoader construction for graph + grid + volume inputs."""
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class SplitData:
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]


ID_COLS = ("mof_id", "id", "name", "coreid")


def _find_id_col(fieldnames, what):
    cols = [c for c in fieldnames if c.lower() in ID_COLS]
    if not cols:
        raise ValueError(f"{what} needs an id column named one of {ID_COLS}; found {fieldnames}")
    return cols[0]


def load_labels(label_file: str, task: str = "classification"):
    """Read the label CSV: id + 'label' (classification) or id + one or more target columns (regression)."""
    labels = {}
    with open(label_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        key_col = _find_id_col(reader.fieldnames, "label file")
        if task == "regression":
            target_cols = [c for c in reader.fieldnames if c != key_col]
            if not target_cols:
                raise ValueError("label file has no target columns besides the id column")
            for row in reader:
                vals = [float(row[c]) if row.get(c, "") not in ("", None) else np.nan
                        for c in target_cols]
                labels[str(row[key_col])] = {"target": np.asarray(vals, dtype=np.float32),
                                             "label": 0}
            labels["__target_cols__"] = target_cols
            return labels
        label_cols = [c for c in reader.fieldnames if c.lower() in ("label", "target", "y")]
        if not label_cols:
            raise ValueError(f"label file needs a 'label' column; found {reader.fieldnames}")
        for row in reader:
            labels[str(row[key_col])] = {"label": int(row[label_cols[0]])}
    return labels


def load_groups(group_file: str) -> Dict[str, int]:
    """Read structure id -> group_id."""
    groups: Dict[str, int] = {}
    with open(group_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        id_col = _find_id_col(reader.fieldnames, "group file")
        g_cols = [c for c in reader.fieldnames if c.lower() in ('group_id', 'group')]
        if not g_cols:
            raise ValueError(f"group file needs a 'group_id' column; found {reader.fieldnames}")
        for row in reader:
            groups[str(row[id_col])] = int(row[g_cols[0]])
    return groups


def load_split(split_file: str) -> SplitData:
    """Read train/val/test ids from a split JSON."""
    with open(split_file, 'r', encoding='utf-8') as f:
        split = json.load(f)
    return SplitData(train_ids=list(split['train']), val_ids=list(split['val']), test_ids=list(split['test']))


class GraphElectrostaticDataset(Dataset):
    """Loads one structure: graph (.npz), grid (.npy), volume and label."""
    def __init__(self, mof_ids, labels, graph_dir, grid_dir, grid_size=30, groups=None):
        self.mof_ids = list(mof_ids)
        self.labels = labels
        self.groups = groups or {}
        self.graph_dir = graph_dir
        self.grid_dir = grid_dir
        self.grid_size = int(grid_size)

    def __len__(self):
        return len(self.mof_ids)

    def _load_graph(self, mof_id):
        path = os.path.join(self.graph_dir, f'{mof_id}.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        g = np.load(path, allow_pickle=False)
        required = ['atom_fea', 'charges', 'nbr_fea', 'nbr_fea_idx', 'volume']
        missing = [k for k in required if k not in g]
        if missing:
            raise KeyError(f'{path} missing keys: {missing}')
        if g['atom_fea'].shape[0] != g['charges'].shape[0]:
            raise ValueError(f'{mof_id}: atom count != charge count')
        return {
            'atom_fea': g['atom_fea'].astype(np.float32),
            'charges': g['charges'].astype(np.float32),
            'nbr_fea': g['nbr_fea'].astype(np.float32),
            'nbr_fea_idx': g['nbr_fea_idx'].astype(np.int64),
            'volume': np.array([float(g['volume'])], dtype=np.float32),
        }

    def _load_grid(self, mof_id):
        path = os.path.join(self.grid_dir, f'{mof_id}.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        grid = np.load(path).astype(np.float32)
        if grid.shape[-3:] == (self.grid_size, self.grid_size, self.grid_size):
            return grid
        x = torch.as_tensor(grid, dtype=torch.float32).unsqueeze(0)
        x = F.interpolate(x, size=(self.grid_size, self.grid_size, self.grid_size),
                          mode='trilinear', align_corners=False)
        return x.squeeze(0).cpu().numpy().astype(np.float32)

    def __getitem__(self, idx):
        mof_id = self.mof_ids[idx]
        graph = self._load_graph(mof_id)
        grid = self._load_grid(mof_id)
        return {
            'mof_id': mof_id,
            'atom_fea': graph['atom_fea'],
            'charges': graph['charges'],
            'nbr_fea': graph['nbr_fea'],
            'nbr_fea_idx': graph['nbr_fea_idx'],
            'volume': graph['volume'],
            'grid': grid,
            'label': int(self.labels[mof_id]['label']),
            'group_id': int(self.groups.get(mof_id, -1)),
            'target': self.labels[mof_id].get('target'),
        }


def collate_graph_electrostatic(batch):
    """Batch variable-size crystal graphs together with grids and labels."""
    atom_fea_list, charge_list, nbr_fea_list, nbr_idx_list = [], [], [], []
    crystal_atom_idx_list = []
    volume_list, grid_list, label_list = [], [], []
    group_list, target_list = [], []
    mof_ids = []
    atom_offset = 0
    for crystal_id, item in enumerate(batch):
        atom_fea = torch.as_tensor(item['atom_fea'], dtype=torch.float32)
        charges = torch.as_tensor(item['charges'], dtype=torch.float32)
        nbr_fea = torch.as_tensor(item['nbr_fea'], dtype=torch.float32)
        nbr_idx = torch.as_tensor(item['nbr_fea_idx'], dtype=torch.long)
        n_atoms = atom_fea.shape[0]
        atom_fea_list.append(atom_fea)
        charge_list.append(charges)
        nbr_fea_list.append(nbr_fea)
        nbr_idx_list.append(nbr_idx + atom_offset)
        crystal_atom_idx_list.append(torch.full((n_atoms,), crystal_id, dtype=torch.long))
        volume_list.append(torch.as_tensor(item['volume'], dtype=torch.float32))
        grid_list.append(torch.as_tensor(item['grid'], dtype=torch.float32))
        label_list.append(int(item['label']))
        group_list.append(int(item.get('group_id', -1)))
        if item.get('target') is not None:
            target_list.append(torch.as_tensor(item['target'], dtype=torch.float32))
        mof_ids.append(item['mof_id'])
        atom_offset += n_atoms
    return {
        'mof_id': mof_ids,
        'atom_fea': torch.cat(atom_fea_list, dim=0),
        'charges': torch.cat(charge_list, dim=0),
        'nbr_fea': torch.cat(nbr_fea_list, dim=0),
        'nbr_fea_idx': torch.cat(nbr_idx_list, dim=0),
        'crystal_atom_idx': torch.cat(crystal_atom_idx_list, dim=0),
        'volume': torch.stack(volume_list, dim=0),
        'grid': torch.stack(grid_list, dim=0),
        'label': torch.as_tensor(label_list, dtype=torch.long),
        'group_id': torch.as_tensor(group_list, dtype=torch.long),
        'target': torch.stack(target_list, dim=0) if target_list else None,
    }


def create_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders and per-(group, label) loss weights."""
    task = getattr(config, 'task', 'classification')
    labels = load_labels(config.label_file, task=task)
    target_cols = labels.pop('__target_cols__', None)
    if task == 'regression':
        n = len(target_cols)
        if getattr(config, 'n_outputs', None) in (None, 0):
            config.n_outputs = n
        elif config.n_outputs != n:
            raise ValueError(
                f"config.n_outputs={config.n_outputs} but the labels file has "
                f"{n} target columns: {target_cols}")
        config.target_cols = target_cols
        print(f"[regression] {n} targets: {target_cols}")
    split = load_split(config.split_file)
    group_file = getattr(config, 'group_file', None)
    groups = load_groups(group_file) if group_file else {}

    def _filter(ids):
        kept = []
        for mof_id in ids:
            g = os.path.join(config.graph_dir, f'{mof_id}.npz')
            e = os.path.join(config.grid_dir, f'{mof_id}.npy')
            if mof_id in labels and os.path.exists(g) and os.path.exists(e):
                kept.append(mof_id)
        return kept

    train_ids = _filter(split.train_ids)
    val_ids = _filter(split.val_ids)
    test_ids = _filter(split.test_ids)
    if group_file:
        missing_g = [m for m in train_ids + val_ids + test_ids if m not in groups]
        if missing_g:
            raise KeyError(
                f"{len(missing_g)} structures are not in group_file "
                f"({group_file}); e.g. {missing_g[:3]}")
    else:
        groups = {m: i for i, m in enumerate(train_ids + val_ids + test_ids)}
    for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        if not ids:
            raise ValueError(f"no usable structures in the {name} split: ids must be in the label "
                             f"file and have a graph in {config.graph_dir} and a grid in {config.grid_dir}")

    train_ds = GraphElectrostaticDataset(train_ids, labels, config.graph_dir, config.grid_dir, config.grid_size, groups)
    val_ds = GraphElectrostaticDataset(val_ids, labels, config.graph_dir, config.grid_dir, config.grid_size, groups)
    test_ds = GraphElectrostaticDataset(test_ids, labels, config.graph_dir, config.grid_dir, config.grid_size, groups)
    kwargs = dict(num_workers=config.num_workers, pin_memory=config.pin_memory,
                  collate_fn=collate_graph_electrostatic)
    if config.num_workers > 0:
        kwargs['persistent_workers'] = bool(getattr(config, 'persistent_workers', True))
        kwargs['prefetch_factor'] = int(getattr(config, 'prefetch_factor', 2))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, **kwargs)

    counts: Dict[Tuple[int, int], int] = {}
    for m in train_ids:
        k = (groups[m], int(labels[m]['label']))
        counts[k] = counts.get(k, 0) + 1
    train_loader.group_weight = {k: 1.0 / c for k, c in counts.items()}
    if task == 'classification':
        config.class_counts = [sum(1 for m in train_ids if int(labels[m]['label']) == c)
                               for c in range(config.n_classes)]
    return train_loader, val_loader, test_loader
