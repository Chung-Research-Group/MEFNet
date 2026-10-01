"""Predict with a trained MEFNet (built-in weights or your own run folder); evaluate when labels are given."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mefnet.configs.config import Config
from mefnet.data.dataset import (GraphElectrostaticDataset, collate_graph_electrostatic,
                                 load_labels, load_split)
from mefnet.engine import collect_outputs, compute_metrics, compute_regression_metrics
from mefnet.models.model import GraphElectrostaticModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_dir', required=True,
                   help='folder with best.pt and config.json: weights/mean, weights/gate, weights/gcca or a train.py run folder')
    p.add_argument('--graph_dir', required=True, help='graph .npz folder')
    p.add_argument('--grid_dir', required=True, help='grid .npy folder')
    p.add_argument('--split_file', default=None, help='split JSON; use with --subset')
    p.add_argument('--subset', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--label_file', default=None,
                   help='label CSV; with it metrics are computed (and, without --split_file, its ids are predicted)')
    p.add_argument('--group_file', default=None, help='CSV with id and group_id columns (material-level metrics)')
    p.add_argument('--output', default='predictions.csv', help='output CSV')
    p.add_argument('--batch_size', type=int, default=32)
    args = p.parse_args()

    with open(os.path.join(args.model_dir, 'config.json')) as f:
        config = Config.from_dict(json.load(f))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphElectrostaticModel(config).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best.pt'), map_location=device))
    model.eval()

    regression = config.task == 'regression'
    labels = load_labels(args.label_file, config.task) if args.label_file else None
    if labels is not None:
        labels.pop('__target_cols__', None)
    if args.split_file:
        ids = load_split(args.split_file).__dict__[f'{args.subset}_ids']
    elif labels is not None:
        ids = list(labels)
    else:
        ids = sorted(f[:-4] for f in os.listdir(args.graph_dir) if f.endswith('.npz'))
    ids = [i for i in ids if os.path.exists(os.path.join(args.graph_dir, f'{i}.npz'))
           and os.path.exists(os.path.join(args.grid_dir, f'{i}.npy'))
           and (labels is None or i in labels)]
    if not ids:
        sys.exit('no structure has both a graph and a grid (and a label, if --label_file is given)')

    groups = None
    if args.group_file:
        from mefnet.data.dataset import load_groups
        groups = load_groups(args.group_file)
    if groups is None:
        groups = {m: i for i, m in enumerate(ids)}
    use_labels = labels if labels is not None else {
        i: {'label': 0, 'target': np.zeros(config.n_outputs or 1, dtype=np.float32)} for i in ids}
    ds = GraphElectrostaticDataset(ids, use_labels, args.graph_dir, args.grid_dir,
                                   config.grid_size, groups)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_graph_electrostatic)

    o = collect_outputs(model, loader, device)
    out = pd.DataFrame({'id': o['ids']})
    if regression:
        pred = torch.softmax(o['outputs'], -1) if config.mpd_softmax else o['outputs']
        pred = pred.numpy().reshape(len(ids), -1)
        names = config.target_cols or [f'target_{i}' for i in range(pred.shape[1])]
        for j, n in enumerate(names):
            out[f'pred_{n}'] = pred[:, j]
    else:
        probs = torch.softmax(o['outputs'], dim=1).numpy()
        out['prob_class_1'] = probs[:, 1]
        out['pred_class'] = probs.argmax(axis=1)
    out.to_csv(args.output, index=False)
    print(f'Saved {len(out)} predictions to {args.output}')

    if labels is None:
        return
    if regression:
        metrics = compute_regression_metrics(model, loader, device, config.mpd_softmax)
    else:
        metrics = {k: v for k, v in compute_metrics(model, loader, device).items()
                   if k not in ('labels', 'probs', 'preds', 'ids', 'groups')}
        if 'auc' not in metrics:
            print('The evaluated set contains a single class: metrics are not defined.')
    for k, v in metrics.items():
        print(f'  {k:24s} {v:.4f}' if isinstance(v, float) else f'  {k:24s} {v}')
    with open(os.path.splitext(args.output)[0] + '_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
