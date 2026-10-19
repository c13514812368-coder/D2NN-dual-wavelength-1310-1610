from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.fft import dctn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'data' / 'imagenet_1k_sd_scene_3coarse_fixed'
CLASSES = ['natural_landform', 'water_coast', 'built_environment']
BASE_IMPL = Path(str(Path(__file__).resolve().parents[1] / 'legacy' / 'train_impl.py'))
sys.path.insert(0, str(HERE.parent / 'src'))


def load_base():
    spec = importlib.util.spec_from_file_location('scene_coarse_base', BASE_IMPL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rich_features(flat):
    images = flat.reshape(-1, 64, 64, 3)
    red, green, blue = (images[..., i] for i in range(3))
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    gx = np.diff(gray, axis=2, prepend=gray[:, :, :1])
    gy = np.diff(gray, axis=1, prepend=gray[:, :1, :])
    mag = np.hypot(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    contrast = np.stack([gray, red-green, blue-green, images.max(-1)-images.min(-1), mag], -1)
    result = []
    for grid in [1, 2, 4]:
        size = 64 // grid
        cells = images.reshape(-1, grid, size, grid, size, 3)
        result.extend([cells.mean((2, 4)).reshape(len(flat), -1), cells.std((2, 4)).reshape(len(flat), -1)])
        blocks = contrast.reshape(-1, grid, size, grid, size, 5)
        result.extend([blocks.mean((2, 4)).reshape(len(flat), -1), blocks.std((2, 4)).reshape(len(flat), -1)])
        orientation = []
        for index in range(8):
            energy = mag * ((angle >= index * np.pi / 8) & (angle < (index+1) * np.pi / 8))
            orientation.append(energy.reshape(-1, grid, size, grid, size).mean((2, 4)))
        result.append(np.stack(orientation, -1).reshape(len(flat), -1))
    dct = dctn(gray, axes=(1, 2), norm='ortho')[:, :8, :8] / 64
    result.append(dct.reshape(len(flat), -1))
    return np.concatenate(result, axis=1).astype(np.float32)


def representation_fit(train, labels, kind, seed=42):
    scaler = StandardScaler().fit(train)
    norm = scaler.transform(train)
    pca = PCA(n_components=14, random_state=seed).fit(norm)
    if kind == 'linear':
        mapping = None
        space = norm
    else:
        mapping = Nystroem(kernel='rbf', gamma=0.002, n_components=384, random_state=seed).fit(norm)
        space = mapping.transform(norm)
    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', n_components=2).fit(space, labels)
    return {'scaler': scaler, 'pca': pca, 'mapping': mapping, 'lda': lda, 'kind': kind}


def representation_transform(encoder, features):
    norm = encoder['scaler'].transform(features)
    space = norm if encoder['mapping'] is None else encoder['mapping'].transform(norm)
    discriminant = encoder['lda'].transform(space)
    return np.concatenate([discriminant, encoder['pca'].transform(norm)], 1).astype(np.float32)


def report(model, train, yt, val, yv):
    pred = model.predict(val)
    return {'train_accuracy': float(accuracy_score(yt, model.predict(train))),
            'val_accuracy': float(accuracy_score(yv, pred)), 'confusion_matrix': confusion_matrix(yv, pred).tolist()}


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    cache = HERE / 'raw_features.npz'
    base = load_base()
    paths = {}
    for split in ['train', 'val']:
        records, _ = base._list_scene_image_records(ROOT / split, class_names=CLASSES)
        paths[split] = [str(path) for path, _ in records]
    if set(paths['train']) & set(paths['val']):
        raise ValueError('Training and validation paths overlap')
    if cache.exists():
        data = np.load(cache)
        xt, xv, yt, yv = [data[key] for key in ['xt', 'xv', 'yt', 'yv']]
        oldt, oldv = data['oldt'], data['oldv']
    else:
        arrays = []
        for split in ['train', 'val']:
            records, _ = base._list_scene_image_records(ROOT / split, class_names=CLASSES)
            images, labels = base._load_scene_image_matrix(records, image_size=(64, 64), color_mode='rgb')
            print(f'Loaded {split}: {len(labels)} images', flush=True)
            arrays.append((rich_features(images), labels))
        xt, yt = arrays[0]
        xv, yv = arrays[1]
        old = []
        for split in ['train', 'val']:
            records, _ = base._list_scene_image_records(ROOT / split, class_names=CLASSES)
            images, _ = base._load_scene_image_matrix(records, image_size=(32, 32), color_mode='rgb')
            old.append(base._make_scene_structure16_features(images, image_size=(32, 32), color_mode='rgb'))
        oldt, oldv = old
        np.savez_compressed(cache, xt=xt, xv=xv, yt=yt, yv=yv, oldt=oldt, oldv=oldv)
    fit_idx, hold_idx = train_test_split(np.arange(len(yt)), test_size=0.2, random_state=42, stratify=yt)
    inner = {}
    for kind in ['linear', 'kernel']:
        encoder = representation_fit(xt[fit_idx], yt[fit_idx], kind)
        a = representation_transform(encoder, xt[fit_idx])
        b = representation_transform(encoder, xt[hold_idx])
        probe = make_pipeline(StandardScaler(), SVC(C=8, gamma='scale')).fit(a, yt[fit_idx])
        inner[kind] = float(accuracy_score(yt[hold_idx], probe.predict(b)))
        print(f'Train-only holdout {kind}: {inner[kind]:.4%}', flush=True)
    choice = max(inner, key=inner.get)
    encoder = representation_fit(xt, yt, choice)
    ft, fv = representation_transform(encoder, xt), representation_transform(encoder, xv)
    low, high = np.percentile(ft, [0.5, 99.5], axis=0)
    span = np.maximum(high-low, 1e-6)
    # Keep inputs away from equivalent phase endpoints at 0 and 2 pi.
    phase_t = np.clip(0.2 + 1.6 * (ft-low)/span, 0.2, 1.8).astype(np.float32)
    phase_v = np.clip(0.2 + 1.6 * (fv-low)/span, 0.2, 1.8).astype(np.float32)
    encoder.update({'phase_low': low, 'phase_high': high, 'classes': CLASSES})
    joblib.dump(encoder, HERE / 'encoder.joblib')
    np.savez_compressed(HERE / 'encoded16.npz', x_train=phase_t, x_eval=phase_v, y_train=yt, y_eval=yv)
    info = {'root': str(ROOT), 'train_samples': len(yt), 'eval_samples': len(yv), 'classes': CLASSES,
            'feature_pool_dim': xt.shape[1], 'input_dim': 16, 'encoding': choice,
            'encoder_fit_split': 'train_only', 'selection_split': 'train_only_20_percent_holdout',
            'inner_accuracy': inner, 'no_pretrained_network': True,
            'path_hash': {k: hashlib.sha256('\n'.join(v).encode()).hexdigest() for k,v in paths.items()},
            'path_manifest': paths, 'baselines': {}}
    for name, a, b in [('original16_svm', oldt, oldv), ('rich_features_svm', xt, xv), ('encoded16_svm', phase_t, phase_v)]:
        model = make_pipeline(StandardScaler(), SVC(C=8, gamma='scale')).fit(a, yt)
        info['baselines'][name] = report(model, a, yt, b, yv)
        print(name, json.dumps(info['baselines'][name]), flush=True)
    (HERE / 'diagnostics.json').write_text(json.dumps(info, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
