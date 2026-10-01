"""Optuna hyperparameter search for MEFNet (material-level validation AUC, or MAE for regression)."""
import argparse
import gc
import json
import os
import sys

import optuna
import pandas as pd
import torch
import torch.nn as nn

torch.multiprocessing.set_sharing_strategy('file_system')

from mefnet.configs.config import Config
from mefnet.data.dataset import create_dataloaders
from mefnet.engine import compute_losses, set_seed, validation_score
from mefnet.models.model import GraphElectrostaticModel


def train_and_evaluate(config, device):
    """Train one trial and return the best validation score (AUC, or negative MAE for regression)."""
    set_seed(config.random_seed)
    train_loader, val_loader, _ = create_dataloaders(config)
    group_weight = getattr(train_loader, 'group_weight', None)
    model = GraphElectrostaticModel(config).to(device)
    class_weight = None
    if config.task == 'classification':
        class_weight = torch.tensor(config.class_weights, dtype=torch.float32, device=device)
    criterion_cls = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs, eta_min=1e-6)

    best_score = -float('inf')
    patience_counter = 0
    for epoch in range(config.max_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = compute_losses(model, batch, criterion_cls, config, device,
                                     group_weight=group_weight, class_weight=class_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        scheduler.step()
        score = validation_score(model, val_loader, device, config)
        if score > best_score:
            best_score = score
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.patience:
            break

    for _ld in (train_loader, val_loader):
        it = getattr(_ld, '_iterator', None)
        if it is not None:
            try:
                it._shutdown_workers()
            except Exception:
                pass
        _ld._iterator = None
    del train_loader, val_loader, model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_score


SEARCH_SPACE = {
    'lr':                (('float', 3e-5, 1e-3), None),
    'weight_decay':      (('float', 1e-4, 1.0),  None),

    'batch_size':        ([32, 64, 128, 256],       None),
    'dropout':           ([0.10, 0.15, 0.20, 0.30], None),
    'embed_dim':         ([128, 256],               None),
    'fusion_hidden_dim': ([128, 256],               None),

    'atom_fea_len':      ([64, 128, 256],   'graph'),
    'n_conv':            ([3, 4, 5],        'graph'),
    'graph_hidden_dim':  ([128, 256, 512],  'graph'),

    'grid_hidden_dim':   ([128, 256, 512],  'grid'),

    'n_attn_heads':      ([2, 4, 8],        'gcsa'),
}

FIXED_PARAMS = {
    'charge_embed_dim': 16,
}


def active_conditions(config):
    use_graph = bool(getattr(config, 'use_graph', True))
    use_grid = bool(config.use_grid)
    return {
        'graph':  use_graph,
        'charge': use_graph and bool(config.use_charge_embedding),
        'grid':   use_grid,
        'gcsa':   use_grid and use_graph and config.fusion_type == 'gcsa',
    }


def suggest_params(trial, config):
    """Sample the active hyperparameters for one trial."""
    cond = active_conditions(config)
    chosen = {}

    for name, val in FIXED_PARAMS.items():
        setattr(config, name, val)

    for name, (spec, need) in SEARCH_SPACE.items():
        if need is not None and not cond[need]:
            continue

        if isinstance(spec, tuple) and spec and spec[0] == 'float':
            _, low, high = spec
            val = trial.suggest_float(name, low, high, log=True)
        else:
            choices = list(spec)
            if name == 'n_attn_heads':
                choices = [c for c in choices if config.grid_hidden_dim % c == 0]
                if not choices:
                    choices = [1]
            val = trial.suggest_categorical(name, choices)

        setattr(config, name, val)
        chosen[name] = val
    return chosen


def build_config(trial, args):
    config = Config()
    config.use_grid = not args.no_grid
    config.use_graph = not args.no_graph
    config.use_volume = not args.no_volume
    config.use_charge_embedding = not args.no_charge
    config.fusion_type = args.fusion_type
    config.task = args.task
    config.mpd_softmax = not args.no_softmax
    config.patience = args.patience
    config.max_epochs = args.max_epochs
    config.random_seed = args.seed
    config.num_workers = args.num_workers
    for name in ('label_file', 'split_file', 'group_file', 'graph_dir', 'grid_dir'):
        if getattr(args, name):
            setattr(config, name, getattr(args, name))
    return config, suggest_params(trial, config)


def objective(trial):
    args = objective.args
    config, _ = build_config(trial, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    failed = -1e9 if args.task == 'regression' else 0.0
    try:
        score = train_and_evaluate(config, device)
    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        return failed
    return failed if score != score else score


def print_callback(study, trial):
    print(f"\nTrial {trial.number:03d} done")
    print(f"  value:   {trial.value:.4f}")
    print(f"  params:  {trial.params}")
    print(f"  current best: {study.best_value:.4f} (Trial #{study.best_trial.number})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='output folder')
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--study_name', default='mefnet_tuning')
    parser.add_argument('--task', choices=['classification', 'regression'], default='classification')
    parser.add_argument('--no_softmax', action='store_true', help='regression: no softmax on the outputs')
    parser.add_argument('--fusion_type', choices=['none', 'gate', 'gcsa'], default='none')
    parser.add_argument('--no_grid', action='store_true')
    parser.add_argument('--no_graph', action='store_true')
    parser.add_argument('--no_volume', action='store_true')
    parser.add_argument('--no_charge', action='store_true')
    parser.add_argument('--label_file', default=None, help='label CSV (default: data/labels.csv)')
    parser.add_argument('--split_file', default=None, help='split JSON (default: data/split.json)')
    parser.add_argument('--group_file', default=None, help='CSV with id and group_id columns (optional)')
    parser.add_argument('--graph_dir', default=None, help='graph .npz folder (default: data/graphs)')
    parser.add_argument('--grid_dir', default=None, help='grid .npy folder (default: data/grids)')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    if args.no_grid and args.no_graph:
        print('[STOP] --no_grid and --no_graph cannot be used together')
        return 2
    objective.args = args
    os.makedirs(args.output, exist_ok=True)

    probe = Config()
    probe.use_grid = not args.no_grid
    probe.use_graph = not args.no_graph
    probe.use_charge_embedding = not args.no_charge
    probe.fusion_type = args.fusion_type
    cond = active_conditions(probe)
    active = [n for n, (_, need) in SEARCH_SPACE.items() if need is None or cond[need]]
    skipped = [n for n, (_, need) in SEARCH_SPACE.items() if need is not None and not cond[need]]
    print('=' * 60)
    print(f"task={args.task} fusion={probe.fusion_type} grid={probe.use_grid} graph={probe.use_graph}")
    print(f"searching {len(active)} parameters: {', '.join(active)}")
    print(f"fixed: {FIXED_PARAMS}")
    print('=' * 60)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        study_name=args.study_name,
    )
    study.optimize(objective, n_trials=args.n_trials, callbacks=[print_callback])

    regression = args.task == 'regression'
    sign = -1.0 if regression else 1.0
    metric = 'val_mae' if regression else 'val_auc'
    print(f"\nBest {metric}: {sign * study.best_value:.4f}  (Trial #{study.best_trial.number})")
    for k, v in study.best_params.items():
        print(f"  {k:20s}: {v}")

    result = {
        f'best_{metric}': sign * study.best_value,
        'best_trial':   study.best_trial.number,
        'best_params':  study.best_params,
        'n_trials':     args.n_trials,
        'task':         args.task,
        'ablation': {
            'use_grid':   not args.no_grid,
            'use_graph':  not args.no_graph,
            'use_volume': not args.no_volume,
            'use_charge_embedding': not args.no_charge,
            'fusion_type': args.fusion_type,
        },
        'searched_params': active,
        'fixed_params':    dict(FIXED_PARAMS),
        'skipped_params':  skipped,
        'max_epochs': args.max_epochs,
        'patience':   args.patience,
        'seed':       args.seed,
    }
    with open(os.path.join(args.output, 'best_params.json'), 'w') as f:
        json.dump(result, f, indent=2)

    rows = [{'trial': t.number, metric: None if t.value is None else sign * t.value, **t.params}
            for t in study.trials]
    pd.DataFrame(rows).sort_values(metric, ascending=regression).to_csv(
        os.path.join(args.output, 'all_trials.csv'), index=False)

    try:
        importances = optuna.importance.get_param_importances(study)
        with open(os.path.join(args.output, 'param_importances.json'), 'w') as f:
            json.dump(dict(importances), f, indent=2)
    except Exception as e:
        print(f"Importance computation failed: {e}")
    print(f"Saved to {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
