from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

BASE_PATH = Path(str(Path(__file__).resolve().parents[1] / 'legacy' / 'train_impl.py'))
spec = importlib.util.spec_from_file_location('coarse_legacy_train_impl', BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
HERE = Path(__file__).resolve().parent


def _dataset(*, dataset_cfg, num_inputs, num_outputs, seed):
    cache_path = Path(dataset_cfg['encoded_features_path'])
    info = json.loads((cache_path.parent / 'diagnostics.json').read_text())
    with np.load(cache_path) as data:
        xt, xv = data['x_train'], data['x_eval']
        yt, yv = data['y_train'], data['y_eval']
    if xt.shape[1] != num_inputs or xv.shape[1] != num_inputs:
        raise ValueError('Encoded feature width disagrees with optical inputs')
    if not np.isfinite(xt).all() or not np.isfinite(xv).all():
        raise ValueError('Encoded input contains non-finite values')
    classes = info['classes']
    summary = {
        'dataset_name': 'imagenet_scene_subset_single', 'root': dataset_cfg['root'],
        'train_split': str(Path(dataset_cfg['root']) / 'train'), 'eval_split': str(Path(dataset_cfg['root']) / 'val'),
        'split_source': 'explicit_train_val_dirs', 'feature_kind': f"rich_{info['encoding']}_lda2_pca14",
        'image_color_mode': 'rgb', 'image_size': [64, 64], 'pca_components': 16,
        'class_names_active': classes, 'class_names_padded': base._pad_class_names(classes, output_dim=num_outputs),
        'train_samples': len(yt), 'eval_samples': len(yv),
        'train_counts_used': {name: int((yt == idx).sum()) for idx, name in enumerate(classes)},
        'eval_counts_used': {name: int((yv == idx).sum()) for idx, name in enumerate(classes)},
        'input_train_min': float(xt.min()), 'input_train_max': float(xt.max()),
        'input_train_mean': float(xt.mean()), 'input_train_std': float(xt.std()),
        'encoder_fit_split': 'train_only', 'no_pretrained_network': True,
        'encoded_features_path': str(cache_path), 'encoder_path': str(cache_path.parent / 'encoder.joblib'),
        'path_hash': info['path_hash'], 'feature_pool_dim': info['feature_pool_dim'],
    }
    return xt, xv, base._one_hot_with_output_dim(yt, output_dim=num_outputs), base._one_hot_with_output_dim(yv, output_dim=num_outputs), 0.0, summary


def build_training_state(mapped_network_package, train_cfg, *, train_dir, seed):
    import copy
    train_cfg = copy.deepcopy(train_cfg)
    root = HERE.parent
    for key in ('root', 'encoded_features_path'):
        candidate = Path(train_cfg['dataset'][key])
        if not candidate.is_absolute():
            train_cfg['dataset'][key] = str((root / candidate).resolve())
    torch.set_num_threads(int(train_cfg.get('model', {}).get('cpu_threads', 4)))
    previous = base._make_imagenet_scene_subset_single_dataset
    base._make_imagenet_scene_subset_single_dataset = _dataset
    try:
        state = base.build_training_state(mapped_network_package, train_cfg, train_dir=train_dir, seed=seed)
    finally:
        base._make_imagenet_scene_subset_single_dataset = previous
    if train_cfg.get('optimizer', {}).get('schedule') == 'cosine':
        state['scheduler'] = torch.optim.lr_scheduler.CosineAnnealingLR(
            state['optimizer'], T_max=int(train_cfg['loop']['epochs']), eta_min=3e-5)
    state['gradient_clip_norm'] = float(train_cfg.get('optimizer', {}).get('gradient_clip_norm', 1.0))
    return state


def train_one_epoch(state, epoch):
    start = perf_counter()
    model, opt = state['model'], state['optimizer']
    model.train()
    total, correct, loss_total = 0, 0, 0.0
    for x, y in state['train_loader']:
        x, y = x.to(state['device']), y.to(state['device'])
        x = base._apply_train_input_noise(x, std=state['augment_noise_std'], input_low=state['input_low'], input_high=state['input_high'])
        opt.zero_grad(set_to_none=True)
        outputs = base._forward_single_wavelength(model, x, wavelength_um=state['task_wavelengths_um'][0], objective_cfg=state['objective_cfg'], device=state['device'])
        loss = state['loss_fn'](outputs['pred'], y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), state['gradient_clip_norm'])
        opt.step()
        base._apply_slot_length_constraints(model, state['constraints_cfg'])
        total += len(x)
        correct += base._accuracy_from_predictions(outputs['pred'], y)
        loss_total += float(loss.item()) * len(x)
    if state['scheduler'] is not None:
        state['scheduler'].step()
    wl = state['task_wavelength_keys'][0]
    return {'train_accuracy': correct/total, 'train_accuracy_combined': correct/total,
            f'train_accuracy_{wl}': correct/total, 'train_loss': loss_total/total,
            'train_loss_combined': loss_total/total, f'train_loss_{wl}': loss_total/total,
            'lr': opt.param_groups[0]['lr'], 'epoch_train_seconds': perf_counter()-start}


evaluate_model = base.evaluate_model
dump_reference_artifacts = base.dump_reference_artifacts


def summarize_training(state, history, *, best_epoch, best_metric):
    summary = base.summarize_training(state, history, best_epoch=best_epoch, best_metric=best_metric)
    best = next(record for record in history if record['epoch'] == best_epoch)
    summary.update({
        'epochs': history[-1]['epoch'], 'initial_evaluation_records': 1,
        'best_eval_accuracy_combined_epoch': best_epoch,
        'best_eval_accuracy_1.56': best['eval_accuracy_1.56'],
        'reference_prediction_by_wavelength': best['reference_prediction_by_wavelength'],
        'reference_prediction_combined': best['reference_prediction_combined'],
        'best_confusion_matrix_combined': best['confusion_matrix'],
    })
    return summary
