"""Shared training and evaluation routines: losses, metrics and result files."""
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, ConfusionMatrixDisplay)

from .models.model import extract_patch_channel_means


def set_seed(seed):
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def count_parameters(model):
    """Return the trainable parameter count and the count per model branch."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups = {
        'graph_encoder':  ('graph_encoder',),
        'grid_encoder':   ('grid_encoder',),
        'fusion_gate':    ('graph_gate', 'grid_attn', 'query_proj'),
        'volume_proj':    ('volume_proj',),
        'fusion_mlp':     ('fusion', 'final_norm'),
        'heads':          ('classifier', 'patch_recon_head', 'mpd_head'),
    }
    branch = {}
    for name, prefixes in groups.items():
        n = sum(p.numel() for pn, p in model.named_parameters()
                if p.requires_grad and pn.split('.')[0] in prefixes)
        if n:
            branch[name] = n
    return total, branch


def collect_outputs(model, loader, device):
    """Run the model over a loader; return ids, groups, labels, targets and raw outputs."""
    model.eval()
    ids, groups, labels, targets, outs = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                atom_fea=batch['atom_fea'].to(device),
                charges=batch['charges'].to(device),
                nbr_fea=batch['nbr_fea'].to(device),
                nbr_fea_idx=batch['nbr_fea_idx'].to(device),
                crystal_atom_idx=batch['crystal_atom_idx'].to(device),
                grid=batch['grid'].to(device),
                volume=batch['volume'].to(device),
            )
            outs.append(out.cpu())
            ids.extend(batch['mof_id'])
            groups.extend(batch['group_id'].cpu().numpy())
            labels.extend(batch['label'].cpu().numpy())
            if batch['target'] is not None:
                targets.append(batch['target'].numpy())
    return {'ids': ids, 'groups': np.asarray(groups), 'labels': np.asarray(labels),
            'targets': np.concatenate(targets, axis=0) if targets else None,
            'outputs': torch.cat(outs, dim=0)}


def compute_regression_metrics(model, loader, device, softmax=True):
    """Structure- and material-level MAE/RMSE for regression."""
    o = collect_outputs(model, loader, device)
    pred = torch.softmax(o['outputs'], dim=-1).numpy() if softmax else o['outputs'].numpy()
    tgt = o['targets']
    if pred.ndim == 1:
        pred = pred[:, None]
    groups = o['groups']
    if softmax:
        tgt = np.clip(tgt, 0, None)
        tgt = tgt / np.clip(tgt.sum(axis=1, keepdims=True), 1e-12, None)

    def _metrics(p, t):
        mae = float(np.abs(p - t).mean())
        rmse = float(np.sqrt(((p - t) ** 2).mean()))
        out = {'mae': mae, 'rmse': rmse}
        if softmax:
            m = 0.5 * (p + t)
            def _kl(a, b):
                a = np.clip(a, 1e-12, None); b = np.clip(b, 1e-12, None)
                return (a * np.log(a / b)).sum(axis=1)
            out['js_div'] = float((0.5 * _kl(p, m) + 0.5 * _kl(t, m)).mean())
        return out

    res = {f'{k}_structure': v for k, v in _metrics(pred, tgt).items()}
    keys = np.array([f'{g}' for g in groups])
    uniq = np.unique(keys)
    gp = np.stack([pred[keys == k].mean(axis=0) for k in uniq])
    gt = np.stack([tgt[keys == k].mean(axis=0) for k in uniq])
    res.update({f'{k}_material': v for k, v in _metrics(gp, gt).items()})
    res['n_materials'] = int(len(uniq))
    res['n_structures'] = int(len(pred))
    return res


def per_class_metrics(y_true, y_pred, prefix=""):
    """Per-class precision, recall and F1 at threshold 0.5."""
    out = {}
    for cls in (0, 1):
        name = f"class_{cls}"
        out[f"{prefix}precision_{name}"] = float(
            precision_score(y_true, y_pred, pos_label=cls, zero_division=0))
        out[f"{prefix}recall_{name}"] = float(
            recall_score(y_true, y_pred, pos_label=cls, zero_division=0))
        out[f"{prefix}f1_{name}"] = float(
            f1_score(y_true, y_pred, pos_label=cls, zero_division=0))
        out[f"{prefix}support_{name}"] = int((np.asarray(y_true) == cls).sum())
    out[f"{prefix}f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    return out


def compute_metrics(model, loader, device):
    """Classification outputs plus structure- and material-level metrics.

    Material level = mean probability per (group_id, label). Metrics are omitted when
    the evaluated set contains a single class."""
    o = collect_outputs(model, loader, device)
    probs = torch.softmax(o['outputs'], dim=1)[:, 1].numpy()
    preds = torch.argmax(o['outputs'], dim=1).numpy()
    labels, groups = o['labels'], o['groups']
    out = {'labels': labels, 'probs': probs, 'preds': preds, 'ids': o['ids'], 'groups': groups}
    if len(np.unique(labels)) < 2:
        return out

    out.update({
        'auc':         float(roc_auc_score(labels, probs)),
        'accuracy':    float(accuracy_score(labels, preds)),
        'f1_weighted': float(f1_score(labels, preds, average='weighted')),
        **per_class_metrics(labels, preds),
    })
    keys = np.array([f"{g}_{l}" for g, l in zip(groups, labels)])
    uniq = np.unique(keys)
    g_prob = np.array([probs[keys == k].mean() for k in uniq])
    g_label = np.array([labels[keys == k][0] for k in uniq])
    g_pred = (g_prob >= 0.5).astype(int)
    if len(np.unique(g_label)) >= 2:
        out.update({
            'auc_material':         float(roc_auc_score(g_label, g_prob)),
            'accuracy_material':    float(accuracy_score(g_label, g_pred)),
            'f1_weighted_material': float(f1_score(g_label, g_pred, average='weighted')),
            'n_materials':          int(len(uniq)),
            **per_class_metrics(g_label, g_pred, prefix='mat_'),
        })
    return out


def validation_score(model, loader, device, config):
    """Scalar validation score where higher is better: material AUC or negative material MAE."""
    if config.task == 'regression':
        m = compute_regression_metrics(model, loader, device, config.mpd_softmax)
        return -m['mae_material']
    m = compute_metrics(model, loader, device)
    return m.get('auc_material', m.get('auc', float('nan')))


def _nan(metrics, key):
    return round(metrics.get(key, float('nan')), 4)


def save_results(metrics, result_dir, run_name, n_params=None, n_params_branch=None):
    """Write results.json, test_predictions.csv and the confusion matrix."""
    def per_class(prefix):
        return {f'class_{c}': {k: (_nan(metrics, f'{prefix}{k}_class_{c}') if k != 'support'
                                   else metrics.get(f'{prefix}support_class_{c}'))
                               for k in ('precision', 'recall', 'f1', 'support')}
                for c in (0, 1)}

    result = {
        'run_name':             run_name,
        'seed':                 metrics.get('_seed'),
        'auc_material':         _nan(metrics, 'auc_material'),
        'accuracy_material':    _nan(metrics, 'accuracy_material'),
        'f1_weighted_material': _nan(metrics, 'f1_weighted_material'),
        'f1_macro_material':    _nan(metrics, 'mat_f1_macro'),
        'n_materials':          metrics.get('n_materials'),
        'per_class_material':   per_class('mat_'),
        'per_class_structure':  per_class(''),
        'auc_structure':        _nan(metrics, 'auc'),
        'accuracy_structure':   _nan(metrics, 'accuracy'),
        'f1_weighted_structure': _nan(metrics, 'f1_weighted'),
        'n_structures':         int(len(metrics['labels'])),
        'n_params':             n_params,
        'n_params_by_branch':   n_params_branch,
    }
    with open(os.path.join(result_dir, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {result_dir}/results.json")

    pd.DataFrame({
        'mof_id':       metrics['ids'],
        'group_id':     metrics['groups'],
        'y_true':       metrics['labels'],
        'y_pred':       metrics['preds'],
        'y_pred_proba': metrics['probs'],
    }).to_csv(os.path.join(result_dir, 'test_predictions.csv'), index=False)
    print(f"Saved: {result_dir}/test_predictions.csv")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cm = confusion_matrix(metrics['labels'], metrics['preds'], labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['class_0', 'class_1'])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Confusion Matrix - {run_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"Saved: {result_dir}/confusion_matrix.png")


def _group_weights(batch, group_weight, like):
    gids = batch['group_id'].cpu().numpy()
    labs = batch['label'].cpu().numpy()
    return torch.as_tensor(
        [group_weight.get((int(g), int(l)), 1.0) for g, l in zip(gids, labs)],
        dtype=like.dtype, device=like.device)


def compute_losses(model, batch, criterion_cls, config, device, group_weight=None,
                   class_weight=None):
    """Weighted classification/regression loss plus the patch auxiliary loss."""
    grid = batch['grid'].to(device)
    outputs = model(
        atom_fea=batch['atom_fea'].to(device),
        charges=batch['charges'].to(device),
        nbr_fea=batch['nbr_fea'].to(device),
        nbr_fea_idx=batch['nbr_fea_idx'].to(device),
        crystal_atom_idx=batch['crystal_atom_idx'].to(device),
        grid=grid,
        volume=batch['volume'].to(device),
        return_aux=True,
    )
    logits = outputs['logits']

    if config.task == 'regression':
        target = batch['target'].to(device)
        if config.mpd_softmax:
            logp = F.log_softmax(logits, dim=-1)
            tgt = target.clamp_min(0)
            tgt = tgt / tgt.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            per_sample = F.kl_div(logp, tgt, reduction='none').sum(dim=-1)
        else:
            per_sample = F.smooth_l1_loss(logits, target, reduction='none').mean(dim=-1)
        w = torch.ones_like(per_sample)
        if group_weight is not None:
            w = w * _group_weights(batch, group_weight, per_sample)
        cls_loss = (per_sample * w).sum() / w.sum().clamp_min(1e-12)
        return config.cls_weight * cls_loss, {'cls_loss': float(cls_loss.detach().item()),
                                              'patch_aux_loss': 0.0}

    labels = batch['label'].to(device)
    per_sample = criterion_cls(logits, labels)
    w = torch.ones_like(per_sample)
    if class_weight is not None:
        w = w * class_weight[labels]
    if group_weight is not None:
        w = w * _group_weights(batch, group_weight, per_sample)
    cls_loss = (per_sample * w).sum() / w.sum().clamp_min(1e-12)
    if config.use_grid:
        patch_target = extract_patch_channel_means(grid, config.patch_size)
        patch_aux_loss = F.smooth_l1_loss(outputs['patch_recon_pred'], patch_target)
    else:
        patch_aux_loss = torch.zeros((), device=device)
    total_loss = config.cls_weight * cls_loss + config.patch_aux_weight * patch_aux_loss
    return total_loss, {'cls_loss': float(cls_loss.detach().item()),
                        'patch_aux_loss': float(patch_aux_loss.detach().item())}
