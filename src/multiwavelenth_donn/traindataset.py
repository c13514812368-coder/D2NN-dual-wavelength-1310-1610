from __future__ import annotations

import numpy as np


def _split_dataset(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError(
            "inputs and targets must contain the same number of samples, "
            f"got {inputs.shape[0]} and {targets.shape[0]}"
        )
    if inputs.shape[0] < 2:
        raise ValueError("Need at least 2 samples to split train/eval")

    rng = np.random.default_rng(random_state)
    indices = np.arange(inputs.shape[0])
    rng.shuffle(indices)
    split = int(round(inputs.shape[0] * (1.0 - test_size)))
    split = min(max(split, 1), inputs.shape[0] - 1)
    train_idx = indices[:split]
    test_idx = indices[split:]
    return (
        inputs[train_idx],
        inputs[test_idx],
        targets[train_idx],
        targets[test_idx],
    )


def make_conv2x2_dualout_with_zero(
    *,
    n_samples: int = 1000,
    kernel1: np.ndarray | None = None,
    bias1: float = 0.0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a small supervised dataset used by exploratory notebooks.

    Inputs are four random values expanded to five ports by inserting a fixed zero
    at index 2. Targets are two outputs where the first is the normalized 2x2
    convolution response and the second is its complement.
    """

    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")

    rng = np.random.default_rng(random_state)
    if kernel1 is None:
        kernel = np.ones((2, 2), dtype=np.float32)
    else:
        kernel = np.asarray(kernel1, dtype=np.float32)
        if kernel.shape == (4,):
            kernel = kernel.reshape(2, 2)
        if kernel.shape != (2, 2):
            raise ValueError(
                f"kernel1 must have shape (2, 2) or (4,), got {kernel.shape}"
            )

    x4 = rng.uniform(0.0, 2.0, size=(n_samples, 4)).astype(np.float32)
    x5 = np.insert(x4, 2, 0.0, axis=1).astype(np.float32)

    response = (x4 @ kernel.reshape(-1) + float(bias1)).astype(np.float32)
    peak = float(np.max(np.abs(response)))
    if peak <= 0.0:
        raise ValueError("kernel1 and bias1 produced a zero response for every sample")
    y1 = response / peak
    y2 = 1.0 - y1
    targets = np.stack([y1, y2], axis=1).astype(np.float32)

    return _split_dataset(
        x5,
        targets,
        test_size=test_size,
        random_state=random_state,
    )


__all__ = ["make_conv2x2_dualout_with_zero"]
