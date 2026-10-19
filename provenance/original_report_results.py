from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, '/Users/6t4/Desktop/d2nn/src')
HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent / 'train_rich16_1560_20261011'


def repair_saved_summary():
    from multiwavelenth_donn.experiment_artifacts import fingerprint_payload, save_package
    artifacts = BUNDLE / 'artifacts'
    verified = json.loads((BUNDLE / 'results_verified.json').read_text())
    history = json.loads((artifacts / 'train_metrics.json').read_text())['history']
    best = next(record for record in history if record['epoch'] == verified['best_epoch'])
    metrics_path = artifacts / 'eval_metrics.json'
    summary = json.loads(metrics_path.read_text())
    backup = artifacts / 'eval_metrics_before_summary_fix.json'
    if not backup.exists():
        backup.write_text(metrics_path.read_text())
    summary.update({
        'epochs': history[-1]['epoch'], 'initial_evaluation_records': 1,
        'best_eval_accuracy_combined_epoch': verified['best_epoch'],
        'best_eval_accuracy_1.56': verified['best_val_accuracy'],
        'reference_prediction_by_wavelength': best['reference_prediction_by_wavelength'],
        'reference_prediction_combined': best['reference_prediction_combined'],
        'best_confusion_matrix_combined': best['confusion_matrix'],
        'best_checkpoint_noiseless_train_accuracy': verified['best_checkpoint_noiseless_train_accuracy'],
        'saved_checkpoint_val_reverified': True,
    })
    metrics_path.write_text(json.dumps(summary, indent=2))
    for name in ['best_model.pt', 'final_model.pt']:
        file = artifacts / 'checkpoints' / name
        payload = torch.load(file, map_location='cpu', weights_only=False)
        payload['metrics_summary'] = summary
        save_package(file, payload)
    file = artifacts / 'trained_network_package.pt'
    payload = torch.load(file, map_location='cpu', weights_only=False)
    payload['metrics_summary'].update(summary)
    payload['trained_network_signature'] = fingerprint_payload({
        'mapped_network_signature': payload['mapped_network_package']['mapped_network_signature'],
        'metrics_summary': payload['metrics_summary'],
        'best_state_dict': payload['best_state_dict'], 'final_state_dict': payload['final_state_dict'],
    })
    save_package(file, payload)
    print('Corrected new-experiment summary metadata; model parameters preserved.', flush=True)


def main():
    artifacts = BUNDLE / 'artifacts'
    history = json.loads((artifacts / 'train_metrics.json').read_text())['history']
    best = max(history, key=lambda x: (x['eval_accuracy_combined'], x['epoch']))
    spec = importlib.util.spec_from_file_location('coarse_report_impl', HERE / 'train_impl.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = yaml.safe_load((BUNDLE / 'config.yaml').read_text())
    package = torch.load(HERE.parent / 'map/mapped_network_package.pt', map_location='cpu', weights_only=False)
    state = module.build_training_state(package, config['train'], train_dir=HERE / 'report_workspace', seed=42)
    checkpoint = torch.load(artifacts / 'checkpoints/best_model.pt', map_location='cpu', weights_only=False)
    state['model'].load_state_dict(checkpoint['state_dict'])
    val = module.evaluate_model(state, best['epoch'])
    assert abs(val['eval_accuracy_combined'] - best['eval_accuracy_combined']) < 1e-12
    state['eval_loader'] = DataLoader(state['train_loader'].dataset, batch_size=64, shuffle=False)
    train = module.evaluate_model(state, best['epoch'])
    matrix = np.array(val['confusion_matrix'])[:3, :3]
    recall = (np.diag(matrix) / matrix.sum(1)).tolist()
    result = {
        'wavelength_nm': 1560, 'best_epoch': best['epoch'],
        'baseline_best_val_accuracy': 0.7904761904761904,
        'best_val_accuracy': val['eval_accuracy_combined'],
        'best_checkpoint_noiseless_train_accuracy': train['eval_accuracy_combined'],
        'train_online_accuracy_at_best_epoch': best.get('train_accuracy_combined'),
        'final_epoch': history[-1]['epoch'], 'final_val_accuracy': history[-1]['eval_accuracy_combined'],
        'final_online_train_accuracy': history[-1]['train_accuracy_combined'],
        'improvement_percentage_points': 100 * (val['eval_accuracy_combined'] - 0.7904761904761904),
        'classes': state['dataset_summary']['class_names_active'], 'class_recall': recall,
        'confusion_matrix': matrix.tolist(), 'train_samples': 7200, 'val_samples': 1260,
        'independent_test_set': None, 'full_optical_sampling': state['monitor_samples_used'],
        'preprocessing': state['dataset_summary']['feature_kind'],
        'encoder_trained_on': 'training split only', 'pretrained_cnn': False,
    }
    (BUNDLE / 'results_verified.json').write_text(json.dumps(result, indent=2))
    repair_saved_summary()
    diagnostic = json.loads((HERE / 'diagnostics.json').read_text())
    rows = ['# 1560 nm three-class feature upgrade', '',
            'The original train/val split and all prior experiments are preserved.',
            'The dataset is synthetic Imagenet-1k-SD-1.4. There is no independent test split.', '',
            '## Method', '',
            'RGB images at 64x64 produce 568 handcrafted features: multiscale colors, contrasts, orientation histograms and low-frequency DCT.',
            'A training-only RBF Nystroem map (384 dimensions) feeds two LDA axes; 14 PCA axes retain unsupervised variation.',
            'The linear/kernel choice uses a 20% inner holdout from the training set, not the fixed validation set.',
            'Training-only percentile scaling maps the 16 components to input phases in [0.2,1.8] pi.',
            'No pretrained CNN or classifier probabilities are used as the optical input.',
            'The original calibrated optical map and all 11292 monitor samples are preserved.',
            'Adam uses cosine decay and gradient clipping. The final category still comes from three optical ports.', '',
            'The reported accuracies use the differentiable optical simulation. The newly trained geometry has not been revalidated with full FDTD.', '',
            '## Fixed-split results', '',
            '| Model | Validation accuracy |', '|---|---:|',
            '| Original 1560 nm D2NN | 79.05% |',
            f"| SVM on rich 568-D features (diagnostic) | {diagnostic['baselines']['rich_features_svm']['val_accuracy']:.2%} |",
            f"| SVM on compressed 16-D input (diagnostic) | {diagnostic['baselines']['encoded16_svm']['val_accuracy']:.2%} |",
            f"| New 1560 nm D2NN, best epoch {result['best_epoch']} | {result['best_val_accuracy']:.2%} |", '',
            f"Best checkpoint noiseless train accuracy: {result['best_checkpoint_noiseless_train_accuracy']:.2%}.",
            f"Final-epoch validation accuracy: {result['final_val_accuracy']:.2%}.",
            f"Improvement over the old best: {result['improvement_percentage_points']:.2f} percentage points.", '',
            '## Per-class recall at the best checkpoint', '', '| Class | Correct / total | Recall |', '|---|---:|---:|']
    for i,name in enumerate(result['classes']):
        rows.append(f"| {name} | {matrix[i,i]} / {matrix[i].sum()} | {recall[i]:.2%} |")
    rows.extend(['', '## Interpretation', '',
                 'This improvement includes a richer digital encoder and supervised compression; it is not evidence of a wavelength-count gain.',
                 'Any future dual-wavelength comparison must use matched input encoders and fixed splits.',
                 'The fixed validation set selects the best checkpoint, so an independent untouched test set is still needed for a final test claim.', '',
                 '## Reproduce', '',
                 'Run feature_upgrade_20261011/prepare_features.py, then train_rich16_1560_20261011/run.py.',
                 'The serialized encoder is encoder.joblib and the input cache is encoded16.npz.',
                 'report_results.py reloads the saved best checkpoint and verifies its full-validation result.'])
    (BUNDLE / 'README.md').write_text('\n'.join(rows)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old_history = json.loads((HERE.parent / 'train_scene_structure16_20ep/artifacts/train_metrics.json').read_text())['history']
    fig, (curve, heatmap) = plt.subplots(1, 2, figsize=(12, 4.7), gridspec_kw={'width_ratios': [1.5, 1]})
    curve.plot([r['epoch'] for r in old_history], [100*r['eval_accuracy_combined'] for r in old_history], color='#657787', label='Original structure16')
    curve.plot([r['epoch'] for r in history], [100*r['eval_accuracy_combined'] for r in history], color='#167D71', label='Rich supervised 16-D encoding')
    curve.axhline(90, color='#C85F3A', linestyle='--', linewidth=1, label='90% target')
    curve.set(xlabel='Epoch', ylabel='Fixed validation accuracy (%)', title='1560 nm, identical 1260 validation images', ylim=(25, 100))
    curve.grid(alpha=.15)
    curve.legend(loc='lower right', fontsize=9)
    heatmap.imshow(matrix, cmap='Blues', vmin=0, vmax=420)
    heatmap.set_xticks(range(3), ['Landform', 'Water/coast', 'Built'], rotation=20)
    heatmap.set_yticks(range(3), ['Landform', 'Water/coast', 'Built'])
    heatmap.set(xlabel='Predicted class', ylabel='True class', title=f"Best checkpoint, epoch {result['best_epoch']}")
    for i in range(3):
        for j in range(3):
            heatmap.text(j, i, str(matrix[i,j]), ha='center', va='center', color='white' if matrix[i,j] > 210 else '#172832', fontsize=13)
    fig.tight_layout()
    fig.savefig(BUNDLE / 'comparison.png', dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
