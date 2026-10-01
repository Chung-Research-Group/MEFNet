"""Train and evaluate MEFNet (classification or regression) and save results."""
import argparse
import json
import os
import time
from dataclasses import asdict

import torch
import torch.nn as nn

torch.multiprocessing.set_sharing_strategy('file_system')

from mefnet.configs.config import Config
from mefnet.data.dataset import create_dataloaders
from mefnet.engine import (compute_losses, compute_metrics, compute_regression_metrics,
                           count_parameters, save_results, set_seed, validation_score)
from mefnet.models.model import GraphElectrostaticModel


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_name', required=True, help='output folder name under --result_dir')
    p.add_argument('--result_dir', default=None, help='parent folder of run folders (default: results)')
    p.add_argument('--label_file', default=None, help='label CSV (default: data/labels.csv)')
    p.add_argument('--split_file', default=None, help='split JSON (default: data/split.json)')
    p.add_argument('--group_file', default=None,
                   help='CSV with id and group_id columns; omitted = every structure is its own group')
    p.add_argument('--graph_dir', default=None, help='graph .npz folder (default: data/graphs)')
    p.add_argument('--grid_dir', default=None, help='grid .npy folder (default: data/grids)')
    p.add_argument('--task', choices=['classification', 'regression'], default=None,
                   help='regression uses every non-id column of the label file as a target')
    p.add_argument('--no_softmax', action='store_true',
                   help='regression: no softmax on the outputs (targets are not a probability distribution)')
    p.add_argument('--fusion_type', choices=['none', 'gate', 'gcsa'], default=None,
                   help='grid pooling: none = mean, gate = graph-gated, gcsa = graph-conditioned cross-attention (GCCA)')
    p.add_argument('--no_grid', action='store_true', help='drop the grid branch')
    p.add_argument('--no_graph', action='store_true', help='drop the graph branch')
    p.add_argument('--no_volume', action='store_true', help='drop the unit cell volume input')
    p.add_argument('--no_charge', action='store_true', help='drop the charge embedding')
    p.add_argument('--max_epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--patience', type=int, default=None)
    p.add_argument('--seed', type=int, default=None, help='training seed (default: 42)')
    p.add_argument('--params_json', default=None,
                   help='JSON with "best_params" (e.g. best_params.json written by tune.py)')
    p.add_argument('--num_workers', type=int, default=None)
    return p


def main():
    args = build_parser().parse_args()
    config = Config()
    if args.no_grid:
        config.use_grid = False
    if args.no_graph:
        config.use_graph = False
    if args.no_volume:
        config.use_volume = False
    if args.no_charge:
        config.use_charge_embedding = False
    if args.fusion_type:
        config.fusion_type = args.fusion_type
    if args.task:
        config.task = args.task
    if args.no_softmax:
        config.mpd_softmax = False
    for name in ('seed', 'max_epochs', 'batch_size', 'patience', 'num_workers'):
        if getattr(args, name) is not None:
            setattr(config, 'random_seed' if name == 'seed' else name, getattr(args, name))

    if args.params_json:
        with open(args.params_json) as f:
            p = json.load(f)
        for k, v in {**p.get('fixed_params', {}), **p['best_params']}.items():
            setattr(config, k, v)
        print(f'[params_json] applied {p["best_params"]}')
    for name in ('label_file', 'split_file', 'group_file', 'graph_dir', 'grid_dir', 'result_dir'):
        if getattr(args, name):
            setattr(config, name, getattr(args, name))

    result_dir = os.path.join(config.result_dir, args.run_name)
    os.makedirs(result_dir, exist_ok=True)

    set_seed(config.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Run: {args.run_name}   seed: {config.random_seed}")

    train_loader, val_loader, test_loader = create_dataloaders(config)
    with open(os.path.join(result_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Structures: train {len(train_loader.dataset)}  val {len(val_loader.dataset)}  "
          f"test {len(test_loader.dataset)}")

    model = GraphElectrostaticModel(config).to(device)
    n_params, n_params_branch = count_parameters(model)
    print(f"Trainable parameters: {n_params:,}")
    for k, v in n_params_branch.items():
        print(f"   {k:16s} {v:>10,}  ({100*v/n_params:5.1f}%)")

    class_weight = None
    if config.task == 'classification':
        class_weight = torch.tensor(config.class_weights, dtype=torch.float32, device=device)
    criterion_cls = nn.CrossEntropyLoss(reduction='none')
    group_weight = getattr(train_loader, 'group_weight', None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs, eta_min=1e-6)

    best_score = -float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_score': [], 'lr': []}
    best_path = os.path.join(result_dir, 'best.pt')
    score_name = 'val_neg_mae' if config.task == 'regression' else 'val_auc'

    for epoch in range(config.max_epochs):
        t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = compute_losses(model, batch, criterion_cls, config, device,
                                     group_weight=group_weight, class_weight=class_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        score = validation_score(model, val_loader, device, config)
        history['train_loss'].append(avg_loss)
        history['val_score'].append(score)
        history['lr'].append(scheduler.get_last_lr()[0])

        if score > best_score or not os.path.exists(best_path):
            best_score = score if score == score else best_score
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1:03d} | loss={avg_loss:.4f} | {score_name}={score:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")
        if patience_counter >= config.patience:
            print(f'Early stopping after epoch {epoch+1}')
            break

    with open(os.path.join(result_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    model.load_state_dict(torch.load(best_path, map_location=device))
    if config.task == 'regression':
        m = compute_regression_metrics(model, test_loader, device, config.mpd_softmax)
        print(f'\nBest val MAE (material-level): {-best_score:.4f}')
        for k in ('mae_material', 'rmse_material', 'js_div_material', 'mae_structure', 'rmse_structure'):
            if k in m:
                print(f'Test {k:18s} {m[k]:.4f}')
        result = dict(run_name=args.run_name, task='regression', seed=int(config.random_seed),
                      n_params=n_params, n_params_by_branch=n_params_branch,
                      target_cols=config.target_cols,
                      **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()})
        with open(os.path.join(result_dir, 'results.json'), 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {result_dir}/results.json")
        return

    m = compute_metrics(model, test_loader, device)
    m['_seed'] = int(config.random_seed)
    nan = float('nan')
    if 'auc' not in m:
        print('\nThe test set contains a single class: metrics are not defined; '
              'only predictions are saved.')
    else:
        print(f'\nBest val AUC (material-level): {best_score:.4f}')
        print(f'{"":14s} {"material":>10s} {"structure":>10s}')
        for label, key in (('Test AUC', 'auc'), ('Test Accuracy', 'accuracy'),
                           ('Test F1', 'f1_weighted')):
            print(f'  {label:13s}{m.get(key + "_material", nan):10.4f} {m[key]:10.4f}')
    save_results(m, result_dir, args.run_name, n_params, n_params_branch)


if __name__ == '__main__':
    main()
