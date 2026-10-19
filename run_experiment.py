from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import warnings


ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'feature_upgrade_20261011'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runtime' / 'matplotlib'))
SAVED = ROOT / 'train_rich16_1560_20261011'


def verify():
    import numpy as np
    import torch
    import yaml
    from multiwavelenth_donn.training_pipeline import _resolve_training_module
    from multiwavelenth_donn.workspace_layout import resolve_training_job_layout

    config = yaml.safe_load((SAVED / 'config.yaml').read_text())
    layout = resolve_training_job_layout(SAVED / 'config.yaml')
    implementation, _ = _resolve_training_module(config['train'], layout=layout)
    mapped = torch.load(ROOT / 'map' / 'mapped_network_package.pt', map_location='cpu', weights_only=False)
    state = implementation.build_training_state(mapped, config['train'], train_dir=ROOT / 'runtime' / 'verify', seed=42)
    model = state['model']
    checkpoint = torch.load(SAVED / 'artifacts' / 'checkpoints' / 'best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    result = implementation.evaluate_model(state, 9)
    accuracy = float(result['eval_accuracy_combined'])
    matrix = np.asarray(result['confusion_matrix'])[:3, :3].tolist()
    expected = json.loads((SAVED / 'results_verified.json').read_text())
    if abs(accuracy - expected['best_val_accuracy']) > 1e-10 or matrix != expected['confusion_matrix']:
        raise AssertionError(f'Validation result changed: {accuracy}, {matrix}')
    train_images = len(list((ROOT / 'data' / 'imagenet_1k_sd_scene_3coarse_fixed' / 'train').rglob('*.jpg')))
    validation_images = len(list((ROOT / 'data' / 'imagenet_1k_sd_scene_3coarse_fixed' / 'val').rglob('*.jpg')))
    if (train_images, validation_images) != (7200, 1260):
        raise AssertionError('Dataset count changed')
    model.train()
    inputs, targets = next(iter(state['train_loader']))
    outputs = implementation.base._forward_single_wavelength(
        model, inputs[:4], wavelength_um=1.56, objective_cfg=state['objective_cfg'], device=state['device'])
    loss = state['loss_fn'](outputs['pred'], targets[:4])
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if not gradients or any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError('Missing or nonfinite training gradient')
    source_paths = {
        'optical_model': Path(sys.modules[model.__class__.__module__].__file__).resolve(),
        'training_implementation': Path(implementation.__file__).resolve(),
        'legacy_helpers': Path(implementation.base.__file__).resolve(),
    }
    if any(not path.is_relative_to(ROOT) for path in source_paths.values()):
        raise AssertionError('Source dependency escaped portable folder')
    report = {
        'wavelength_nm': 1560, 'validation_accuracy': accuracy,
        'validation_confusion_matrix': matrix, 'best_epoch': 9,
        'train_images': train_images, 'validation_images': validation_images,
        'forward_backward_finite': True, 'optimizer_step_performed': False,
        'source_paths': {name: str(path.relative_to(ROOT)) for name, path in source_paths.items()},
        'full_training_rerun': False,
    }
    target = ROOT / 'runtime' / 'verification.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def train():
    import yaml
    from multiwavelenth_donn.training_pipeline import run_local_training_job

    destination = ROOT / 'reruns' / datetime.now().strftime('train_1560_%Y%m%d_%H%M%S_%f')
    destination.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((SAVED / 'config.yaml').read_text())
    config['train']['implementation']['path'] = str(ROOT / 'feature_upgrade_20261011' / 'train_impl.py')
    config_path = destination / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    with (destination / 'training.log').open('w', encoding='utf-8') as log:
        original_stdout = sys.stdout

        class Tee:
            def write(self, text):
                original_stdout.write(text)
                log.write(text)
                log.flush()

            def flush(self):
                original_stdout.flush()
                log.flush()

        sys.stdout = Tee()
        try:
            result = run_local_training_job(config_path=config_path, project_root=ROOT)
            print('FINISHED', result['job_dir'], flush=True)
        finally:
            sys.stdout = original_stdout


def verify_files():
    expected = json.loads((ROOT / 'file_checksums.json').read_text())
    for name, record in expected.items():
        path = ROOT / name
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise AssertionError(f'Missing or wrong-size file: {name}')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != record['sha256']:
            raise AssertionError(f'Checksum mismatch: {name}')
    print(json.dumps({'verified_files': len(expected), 'all_match': True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description='Portable 1560 nm three-class experiment')
    parser.add_argument('action', choices=['verify', 'train', 'predict', 'prepare-features', 'verify-files'])
    parser.add_argument('images', nargs='*')
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.cuda.amp.autocast.*')
    if args.action == 'predict':
        if not args.images:
            parser.error('predict requires image paths')
        subprocess.run([sys.executable, str(ROOT / 'feature_upgrade_20261011' / 'predict.py'), *args.images], check=True)
    elif args.action == 'prepare-features':
        subprocess.run([sys.executable, str(ROOT / 'feature_upgrade_20261011' / 'prepare_features.py')], check=True)
    else:
        {'verify': verify, 'train': train, 'verify-files': verify_files}[args.action]()


if __name__ == '__main__':
    main()
