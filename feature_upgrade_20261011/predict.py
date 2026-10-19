from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))
from prepare_features import load_base, rich_features, representation_transform
from multiwavelenth_donn.experiment_artifacts import instantiate_optical_donn_from_mapped_package


def main():
    parser = argparse.ArgumentParser(description='Predict scene class with the frozen encoder and 1560 nm optical model')
    parser.add_argument('images', nargs='+', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.cuda.amp.autocast.*')
    base = load_base()
    encoder = joblib.load(HERE / 'encoder.joblib')
    records = [(p.resolve(), 0) for p in args.images]
    images, _ = base._load_scene_image_matrix(records, image_size=(64, 64), color_mode='rgb')
    features = representation_transform(encoder, rich_features(images))
    phase = np.clip(0.2 + 1.6 * (features-encoder['phase_low']) / np.maximum(encoder['phase_high']-encoder['phase_low'], 1e-6), 0.2, 1.8).astype(np.float32)
    package = torch.load(HERE.parent / 'map/mapped_network_package.pt', map_location='cpu', weights_only=False)
    model = instantiate_optical_donn_from_mapped_package(package, work_dir=HERE.parent / 'runtime' / 'predict', device='cpu')
    checkpoint = torch.load(HERE.parent / 'train_rich16_1560_20261011/artifacts/checkpoints/best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.inference_mode():
        raw = model(torch.from_numpy(phase), wavelength_um=1.56).reshape(len(records), -1)[:, :3].clamp_min(0)
        probs = raw / raw.sum(-1, keepdim=True).clamp_min(1e-12)
    for index, (path, _) in enumerate(records):
        pred = int(probs[index].argmax())
        print(json.dumps({'image': str(path), 'prediction': encoder['classes'][pred],
                          'normalized_port_responses': dict(zip(encoder['classes'], probs[index].tolist()))}))


if __name__ == '__main__':
    main()
