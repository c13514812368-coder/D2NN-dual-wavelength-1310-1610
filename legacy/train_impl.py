from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision.datasets import CIFAR100, FashionMNIST

from multiwavelenth_donn.experiment_artifacts import build_mapped_network_package
from multiwavelenth_donn.experiment_artifacts import build_shape_spec
from multiwavelenth_donn.experiment_artifacts import instantiate_optical_donn_from_mapped_package
from multiwavelenth_donn.networkangler import NetworkShape
from multiwavelenth_donn.project_paths import get_project_root
from multiwavelenth_donn.wavelength_tools import canonical_wavelength_key
from multiwavelenth_donn.wavelength_tools import resolve_wavelength_choice


CLASS_NAMES = [str(idx) for idx in range(10)]
FASHION_MNIST_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

CIFAR100_DEFAULT_HIERARCHY = {
    "flowers": ["orchid", "rose", "sunflower"],
    "large_carnivores": ["bear", "lion", "tiger"],
    "vehicles_1": ["bicycle", "bus", "motorcycle"],
}

FASHION_ICON_VARIANT_SETS = {
    "icon_four_way": ["flower", "apple", "bear", "letters"],
    "icon_smiley_apple_bear_letters": ["smiley", "apple", "bear", "letters"],
    "icon_smiley_apple_bear_heart": ["smiley", "apple", "bear", "heart"],
    "icon_star_heart_apple_smiley": ["star", "heart", "apple", "smiley"],
    "icon_triangle_slash_apple_smiley": ["triangle", "diagonal_slash", "apple", "smiley"],
}
FASHION_ICON_KINDS = frozenset(
    icon_kind for icon_kinds in FASHION_ICON_VARIANT_SETS.values() for icon_kind in icon_kinds
)
IMAGE_DATASET_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def _resolve_device(device_value: Any) -> torch.device:
    try:
        device = torch.device(device_value)
    except Exception as exc:
        raise ValueError(f"Invalid train.model.device={device_value!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("train.model.device requests CUDA, but CUDA is not available")
    return device


def _build_optimizer(
    name: str,
    params: Sequence[torch.nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
    momentum: float = 0.0,
    nesterov: bool = False,
) -> torch.optim.Optimizer:
    trainable = [param for param in params if param.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters found in model.parameters()")
    name = str(name).lower()
    if name == "adam":
        return torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        if momentum < 0.0:
            raise ValueError("train.optimizer.momentum must be >= 0")
        use_nesterov = bool(nesterov and momentum > 0.0)
        return torch.optim.SGD(
            trainable,
            lr=lr,
            weight_decay=weight_decay,
            momentum=float(momentum),
            nesterov=use_nesterov,
        )
    raise ValueError(f"Unsupported train.optimizer.name={name!r}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    optimizer_cfg: Mapping[str, Any],
):
    milestones = optimizer_cfg.get("lr_milestones")
    if milestones is None:
        return None
    if not isinstance(milestones, Sequence) or isinstance(milestones, (str, bytes)):
        raise TypeError("train.optimizer.lr_milestones must be a sequence of positive integers")
    parsed_milestones = [int(value) for value in milestones]
    if not parsed_milestones or any(value <= 0 for value in parsed_milestones):
        raise ValueError("train.optimizer.lr_milestones must contain positive integers")
    gamma = float(optimizer_cfg.get("lr_gamma", 0.1))
    if gamma <= 0.0 or gamma >= 1.0:
        raise ValueError("train.optimizer.lr_gamma must be in (0, 1)")
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=parsed_milestones,
        gamma=gamma,
    )


def _build_loss(name: str):
    name = str(name).lower()
    if name == "cross_entropy_prob":
        def _loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return -(target * pred.clamp_min(1e-12).log()).sum(dim=-1).mean()
        return _loss
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported train.objective.loss={name!r}")


def _apply_output_activation(
    pred: torch.Tensor,
    objective_cfg: Mapping[str, Any],
) -> torch.Tensor:
    activation = str(objective_cfg.get("activation", "none")).strip().lower()
    if activation in {"", "none", "identity", "linear"}:
        return pred
    if activation == "relu":
        return torch.relu(pred)
    if activation == "gelu":
        return nn.functional.gelu(pred)
    if activation == "sigmoid":
        return torch.sigmoid(pred)
    if activation == "tanh":
        return torch.tanh(pred)
    raise ValueError(f"Unsupported train.objective.activation={activation!r}")


def _apply_output_transform(
    pred: torch.Tensor,
    objective_cfg: Mapping[str, Any],
) -> torch.Tensor:
    transform = str(objective_cfg.get("output_transform", "raw")).lower()
    out = _apply_output_activation(pred, objective_cfg)
    if transform == "port_normalized":
        out = out.clamp_min(float(objective_cfg.get("clamp_min", 0.0)))
        active_output_dim = int(objective_cfg.get("active_output_dim", out.shape[-1]))
        if active_output_dim <= 0 or active_output_dim > out.shape[-1]:
            raise ValueError(
                "train.objective.active_output_dim must be in [1, output_dim], "
                f"got {active_output_dim} for output_dim={out.shape[-1]}"
            )
        active = out[..., :active_output_dim]
        active = active / active.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if active_output_dim == out.shape[-1]:
            return active
        return torch.cat(
            [active, torch.zeros_like(out[..., active_output_dim:])],
            dim=-1,
        )
    if transform == "raw":
        return out
    raise ValueError(f"Unsupported train.objective.output_transform={transform!r}")


def _resolve_dual_hierarchy_config(
    *,
    task_cfg: Mapping[str, Any],
    objective_cfg: Mapping[str, Any],
    dataset_summary: Mapping[str, Any] | None,
    output_dim: int,
) -> Dict[str, Any] | None:
    mode = str(task_cfg.get("hierarchical_mode", "none")).strip().lower()
    if mode in {"", "none", "disabled", "false"}:
        return None
    if mode not in {
        "coarse_a_localfine_b",
        "coarse1560_fine1540",
        "coarse_a_sharedfine3_b",
        "coarse1560_sharedfine1540",
        "coarse_a_localfine_b_gated",
        "coarse1560_fine1540_gated",
    }:
        raise ValueError(
            "train.task.hierarchical_mode must be one of "
            "{'none', 'coarse_a_localfine_b', 'coarse1560_fine1540', "
            "'coarse_a_sharedfine3_b', 'coarse1560_sharedfine1540', "
            "'coarse_a_localfine_b_gated', 'coarse1560_fine1540_gated'}"
        )
    if not isinstance(dataset_summary, Mapping):
        raise ValueError("Hierarchical dual mode requires dataset_summary metadata")
    if str(dataset_summary.get("target_level", "")).strip().lower() != "fine":
        raise ValueError("Hierarchical dual mode requires train.dataset.target_level='fine'")
    output_transform = str(objective_cfg.get("output_transform", "raw")).strip().lower()
    if output_transform != "port_normalized":
        raise ValueError(
            "Hierarchical dual mode requires train.objective.output_transform='port_normalized'"
        )

    coarse_names = [str(name) for name in (dataset_summary.get("selected_coarse_names") or [])]
    fine_per_coarse = int(dataset_summary.get("fine_per_coarse", 0))
    active_class_count = int(dataset_summary.get("active_class_count", 0))
    if not coarse_names or fine_per_coarse <= 0 or active_class_count <= 0:
        raise ValueError("Hierarchical dual mode requires valid coarse/fine dataset metadata")

    expected_active_class_count = len(coarse_names) * fine_per_coarse
    if active_class_count != expected_active_class_count:
        raise ValueError(
            "Hierarchical dual mode requires contiguous coarse-major fine labels. "
            f"Expected active_class_count={expected_active_class_count}, got {active_class_count}"
        )
    if active_class_count > int(output_dim):
        raise ValueError(
            f"Hierarchical dual mode needs active classes <= output_dim, got "
            f"{active_class_count} > {int(output_dim)}"
        )

    fine_names_active = [str(name) for name in (dataset_summary.get("class_names_active") or [])]
    if len(fine_names_active) != active_class_count:
        raise ValueError(
            "Hierarchical dual mode requires dataset_summary.class_names_active to "
            f"list all active fine classes, got {len(fine_names_active)} entries"
        )

    group_ranges = [
        (group_idx * fine_per_coarse, (group_idx + 1) * fine_per_coarse)
        for group_idx in range(len(coarse_names))
    ]
    shared_local_ports_b = bool(
        task_cfg.get("branch_b_shared_local_ports", False)
        or mode in {"coarse_a_sharedfine3_b", "coarse1560_sharedfine1540"}
    )
    branch_b_uses_coarse_context = bool(
        task_cfg.get("branch_b_uses_coarse_context", False)
        or mode in {"coarse_a_localfine_b_gated", "coarse1560_fine1540_gated"}
    )
    branch_b_coarse_context_detach = bool(task_cfg.get("branch_b_coarse_context_detach", True))
    focus_coarse_raw = task_cfg.get("branch_b_focus_coarse")
    branch_b_focus_coarse_index = None
    branch_b_focus_coarse_name = None
    branch_b_focus_group_range = None
    if focus_coarse_raw is not None:
        if isinstance(focus_coarse_raw, str):
            focus_name = str(focus_coarse_raw).strip()
            if focus_name not in coarse_names:
                raise ValueError(
                    "train.task.branch_b_focus_coarse must name one of the selected coarse classes, "
                    f"got {focus_name!r} not in {coarse_names!r}"
                )
            branch_b_focus_coarse_index = coarse_names.index(focus_name)
        else:
            focus_index = int(focus_coarse_raw)
            if focus_index < 0 or focus_index >= len(coarse_names):
                raise ValueError(
                    "train.task.branch_b_focus_coarse index out of range: "
                    f"got {focus_index}, expected [0, {len(coarse_names) - 1}]"
                )
            branch_b_focus_coarse_index = focus_index
        branch_b_focus_coarse_name = coarse_names[int(branch_b_focus_coarse_index)]
        branch_b_focus_group_range = group_ranges[int(branch_b_focus_coarse_index)]
    branch_b_focus_local_ports_only = bool(task_cfg.get("branch_b_focus_local_ports_only", False))
    if branch_b_focus_local_ports_only:
        if branch_b_focus_coarse_index is None:
            raise ValueError(
                "train.task.branch_b_focus_local_ports_only requires "
                "train.task.branch_b_focus_coarse to be set"
            )
        if shared_local_ports_b:
            raise ValueError(
                "train.task.branch_b_focus_local_ports_only cannot be combined with "
                "shared local ports on branch B"
            )
        if branch_b_uses_coarse_context:
            raise ValueError(
                "train.task.branch_b_focus_local_ports_only cannot be combined with "
                "branch_b_uses_coarse_context / gated local-fine mode"
            )
    fine_names_padded = _pad_class_names(fine_names_active, output_dim=int(output_dim))
    return {
        "enabled": True,
        "mode": mode,
        "shared_local_ports_b": shared_local_ports_b,
        "branch_b_uses_coarse_context": branch_b_uses_coarse_context,
        "branch_b_coarse_context_detach": branch_b_coarse_context_detach,
        "branch_b_focus_coarse_index": branch_b_focus_coarse_index,
        "branch_b_focus_coarse_name": branch_b_focus_coarse_name,
        "branch_b_focus_group_range": branch_b_focus_group_range,
        "branch_b_focus_local_ports_only": branch_b_focus_local_ports_only,
        "coarse_count": int(len(coarse_names)),
        "fine_per_coarse": int(fine_per_coarse),
        "active_class_count": int(active_class_count),
        "group_ranges": group_ranges,
        "coarse_names": coarse_names,
        "fine_names_active": fine_names_active,
        "fine_names_padded": fine_names_padded,
        "wl_a_output_layout": coarse_names,
        "wl_b_output_layout": (
            [f"local_slot_{idx}" for idx in range(fine_per_coarse)]
            if shared_local_ports_b or branch_b_uses_coarse_context or branch_b_focus_local_ports_only
            else fine_names_padded
        ),
        "combined_output_layout": fine_names_padded,
    }


def _active_port_distribution(
    pred_full: torch.Tensor,
    dual_hierarchy: Mapping[str, Any],
) -> torch.Tensor:
    active_class_count = int(dual_hierarchy["active_class_count"])
    active = pred_full[:, :active_class_count].clamp_min(1e-12)
    active = active / active.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if pred_full.shape[-1] == active_class_count:
        return active
    out = torch.zeros_like(pred_full)
    out[:, :active_class_count] = active
    return out


def _sum_group_ranges(
    values: torch.Tensor,
    dual_hierarchy: Mapping[str, Any],
) -> torch.Tensor:
    grouped = [
        values[:, int(start):int(end)].sum(dim=-1, keepdim=True)
        for start, end in dual_hierarchy["group_ranges"]
    ]
    return torch.cat(grouped, dim=-1)


def _normalize_within_group_ranges(
    values: torch.Tensor,
    dual_hierarchy: Mapping[str, Any],
) -> torch.Tensor:
    out = torch.zeros_like(values)
    for start, end in dual_hierarchy["group_ranges"]:
        start_idx = int(start)
        end_idx = int(end)
        segment = values[:, start_idx:end_idx].clamp_min(1e-12)
        out[:, start_idx:end_idx] = segment / segment.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return out


def _normalize_shared_local_ports(
    values: torch.Tensor,
    dual_hierarchy: Mapping[str, Any],
) -> torch.Tensor:
    fine_per_coarse = int(dual_hierarchy["fine_per_coarse"])
    local = values[:, :fine_per_coarse].clamp_min(1e-12)
    return local / local.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _extract_branch_b_local_distribution(
    pred_full: torch.Tensor,
    dual_hierarchy: Mapping[str, Any],
) -> torch.Tensor:
    if bool(dual_hierarchy.get("branch_b_focus_local_ports_only", False)):
        return _normalize_shared_local_ports(pred_full, dual_hierarchy)
    if bool(dual_hierarchy.get("shared_local_ports_b", False)):
        return _normalize_shared_local_ports(pred_full, dual_hierarchy)
    pred_active = _active_port_distribution(pred_full, dual_hierarchy)
    return _normalize_within_group_ranges(pred_active, dual_hierarchy)


def _contextualize_branch_b_local_distribution(
    pred_grouped_full: torch.Tensor,
    coarse_prob: torch.Tensor,
    dual_hierarchy: Mapping[str, Any],
) -> torch.Tensor:
    if bool(dual_hierarchy.get("shared_local_ports_b", False)):
        return pred_grouped_full[:, : int(dual_hierarchy["fine_per_coarse"])]
    context = coarse_prob
    if bool(dual_hierarchy.get("branch_b_coarse_context_detach", True)):
        context = context.detach()
    fine_per_coarse = int(dual_hierarchy["fine_per_coarse"])
    coarse_count = int(dual_hierarchy["coarse_count"])
    grouped = pred_grouped_full.new_zeros((pred_grouped_full.shape[0], coarse_count, fine_per_coarse))
    for coarse_idx, (start, end) in enumerate(dual_hierarchy["group_ranges"]):
        grouped[:, coarse_idx, :] = pred_grouped_full[:, int(start):int(end)]
    contextual = torch.einsum("bc,bcf->bf", context, grouped)
    return contextual / contextual.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _compose_hierarchical_fine_distribution(
    coarse_prob: torch.Tensor,
    conditional_fine_prob: torch.Tensor,
    dual_hierarchy: Mapping[str, Any],
) -> torch.Tensor:
    output_dim = len(list(dual_hierarchy["fine_names_padded"]))
    out = coarse_prob.new_zeros((coarse_prob.shape[0], output_dim))
    shared_local_ports_b = bool(dual_hierarchy.get("shared_local_ports_b", False))
    focus_local_only = bool(dual_hierarchy.get("branch_b_focus_local_ports_only", False))
    focus_coarse_index = dual_hierarchy.get("branch_b_focus_coarse_index")
    uniform_local = None
    if focus_local_only:
        if focus_coarse_index is None:
            raise ValueError(
                "Focused local-port-only branch-B mode requires branch_b_focus_coarse_index"
            )
        fine_per_coarse = int(dual_hierarchy["fine_per_coarse"])
        uniform_local = conditional_fine_prob.new_full(
            (conditional_fine_prob.shape[0], fine_per_coarse),
            1.0 / max(fine_per_coarse, 1),
        )
    for coarse_idx, (start, end) in enumerate(dual_hierarchy["group_ranges"]):
        start_idx = int(start)
        end_idx = int(end)
        if focus_local_only:
            segment = (
                conditional_fine_prob
                if int(coarse_idx) == int(focus_coarse_index)
                else uniform_local
            )
        else:
            segment = (
                conditional_fine_prob
                if shared_local_ports_b
                else conditional_fine_prob[:, start_idx:end_idx]
            )
        out[:, start_idx:end_idx] = coarse_prob[:, coarse_idx:coarse_idx + 1] * segment
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _extract_dual_hierarchy_targets(
    target: torch.Tensor,
    dual_hierarchy: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    fine_idx = _class_indices_from_one_hot(target)
    active_class_count = int(dual_hierarchy["active_class_count"])
    if bool(torch.any(fine_idx >= active_class_count).item()):
        raise ValueError(
            "Hierarchical dual mode expects all targets to land inside the active fine classes, "
            f"got max target index {int(fine_idx.max().item())} with active_class_count={active_class_count}"
        )
    fine_per_coarse = int(dual_hierarchy["fine_per_coarse"])
    coarse_count = int(dual_hierarchy["coarse_count"])
    coarse_idx = torch.div(fine_idx, fine_per_coarse, rounding_mode="floor")
    local_idx = torch.remainder(fine_idx, fine_per_coarse)
    coarse_target = nn.functional.one_hot(coarse_idx, num_classes=coarse_count).to(dtype=target.dtype)
    local_target = nn.functional.one_hot(local_idx, num_classes=fine_per_coarse).to(dtype=target.dtype)
    return {
        "fine_idx": fine_idx,
        "coarse_idx": coarse_idx,
        "local_idx": local_idx,
        "coarse_target": coarse_target,
        "local_target": local_target,
    }


def _gather_local_group_predictions(
    pred_grouped_full: torch.Tensor,
    coarse_idx: torch.Tensor,
    dual_hierarchy: Mapping[str, Any],
) -> torch.Tensor:
    fine_per_coarse = int(dual_hierarchy["fine_per_coarse"])
    if bool(dual_hierarchy.get("shared_local_ports_b", False)):
        return pred_grouped_full[:, :fine_per_coarse]
    out = torch.zeros(
        (pred_grouped_full.shape[0], fine_per_coarse),
        dtype=pred_grouped_full.dtype,
        device=pred_grouped_full.device,
    )
    for group_idx, (start, end) in enumerate(dual_hierarchy["group_ranges"]):
        mask = coarse_idx == int(group_idx)
        if bool(mask.any().item()):
            out[mask] = pred_grouped_full[mask, int(start):int(end)]
    return out


def _resolve_branch_b_task_predictions(
    outputs: Mapping[str, torch.Tensor],
    hierarchy_targets: Mapping[str, torch.Tensor],
    dual_hierarchy: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    focus_coarse_index = dual_hierarchy.get("branch_b_focus_coarse_index")
    if bool(dual_hierarchy.get("branch_b_focus_local_ports_only", False)):
        if focus_coarse_index is None:
            raise ValueError(
                "Focused local-port-only branch-B mode requires branch_b_focus_coarse_index"
            )
        return outputs["pred_b"], hierarchy_targets["coarse_idx"] == int(focus_coarse_index)
    if bool(dual_hierarchy.get("branch_b_uses_coarse_context", False)):
        pred_b_local = outputs["pred_b"]
        if focus_coarse_index is None:
            return pred_b_local, None
        return pred_b_local, hierarchy_targets["coarse_idx"] == int(focus_coarse_index)

    if focus_coarse_index is None:
        pred_b_local = _gather_local_group_predictions(
            outputs["pred_b"],
            hierarchy_targets["coarse_idx"],
            dual_hierarchy,
        )
        return pred_b_local, None

    start, end = dual_hierarchy["branch_b_focus_group_range"]
    pred_b_local = outputs["pred_b"][:, int(start):int(end)]
    return pred_b_local, hierarchy_targets["coarse_idx"] == int(focus_coarse_index)


def _masked_loss_mean(
    loss_fn,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return loss_fn(pred, target)
    if mask.ndim != 1:
        raise ValueError(f"Expected 1D mask for masked loss, got shape {tuple(mask.shape)}")
    if pred.shape[0] != mask.shape[0] or target.shape[0] != mask.shape[0]:
        raise ValueError(
            "Masked loss expects batch-aligned tensors, got "
            f"pred={tuple(pred.shape)} target={tuple(target.shape)} mask={tuple(mask.shape)}"
        )
    if not bool(mask.any().item()):
        return pred.sum() * 0.0
    return loss_fn(pred[mask], target[mask])


def _normalized_difference_signal(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    objective_cfg: Mapping[str, Any],
) -> torch.Tensor:
    eps_floor = float(objective_cfg.get("difference_eps", 1e-6))
    if eps_floor <= 0.0:
        raise ValueError("train.objective.difference_eps must be > 0")

    alpha = float(objective_cfg.get("difference_adaptive_eps_alpha", 0.02))
    if alpha < 0.0:
        raise ValueError("train.objective.difference_adaptive_eps_alpha must be >= 0")

    signal_sum = raw_a + raw_b
    adaptive_eps = alpha * signal_sum.mean(dim=-1, keepdim=True)
    eps = adaptive_eps.clamp_min(eps_floor)
    diff = (raw_a - raw_b) / (signal_sum + eps)

    clip_abs = float(objective_cfg.get("difference_clip_abs", 3.0))
    if clip_abs > 0.0:
        diff = diff.clamp(min=-clip_abs, max=clip_abs)
    return diff


def _apply_difference_output_transform(
    diff_signal: torch.Tensor,
    objective_cfg: Mapping[str, Any],
) -> torch.Tensor:
    transform = str(objective_cfg.get("difference_output_transform", "softmax")).strip().lower()
    if transform == "softmax":
        return torch.softmax(diff_signal, dim=-1)
    if transform == "raw":
        return diff_signal
    if transform == "port_normalized":
        clamped = diff_signal.clamp_min(float(objective_cfg.get("clamp_min", 0.0)))
        return clamped / clamped.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    raise ValueError(
        f"Unsupported train.objective.difference_output_transform={transform!r}"
    )


def _mix_combined_predictions(
    pred_sum_branch: torch.Tensor,
    pred_diff_branch: torch.Tensor,
    objective_cfg: Mapping[str, Any],
    *,
    fusion_diff_weight: float | None = None,
) -> torch.Tensor:
    mode = str(objective_cfg.get("combined_fusion_mode", "sum_diff_mix")).strip().lower()
    if mode in {"sum", "sum_only"}:
        out = pred_sum_branch
    elif mode in {"diff", "difference", "difference_only"}:
        out = pred_diff_branch
    elif mode in {"mix", "sum_diff_mix"}:
        diff_weight = float(
            objective_cfg.get("fusion_diff_weight", 0.35)
            if fusion_diff_weight is None
            else fusion_diff_weight
        )
        if diff_weight < 0.0 or diff_weight > 1.0:
            raise ValueError("train.objective.fusion_diff_weight must be in [0, 1]")
        out = (1.0 - diff_weight) * pred_sum_branch + diff_weight * pred_diff_branch
    else:
        raise ValueError(f"Unsupported train.objective.combined_fusion_mode={mode!r}")

    # Keep combined output in probability space for cross_entropy_prob.
    out = out.clamp_min(1e-12)
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _mix_combined_raw_signals(
    raw_sum_branch: torch.Tensor,
    raw_diff_branch: torch.Tensor,
    objective_cfg: Mapping[str, Any],
    *,
    fusion_diff_weight: float | None = None,
) -> torch.Tensor:
    mode = str(objective_cfg.get("combined_fusion_mode", "sum_diff_mix")).strip().lower()
    if mode in {"sum", "sum_only"}:
        return raw_sum_branch
    if mode in {"diff", "difference", "difference_only"}:
        return raw_diff_branch
    if mode in {"mix", "sum_diff_mix"}:
        diff_weight = float(
            objective_cfg.get("fusion_diff_weight", 0.35)
            if fusion_diff_weight is None
            else fusion_diff_weight
        )
        if diff_weight < 0.0 or diff_weight > 1.0:
            raise ValueError("train.objective.fusion_diff_weight must be in [0, 1]")
        return (1.0 - diff_weight) * raw_sum_branch + diff_weight * raw_diff_branch
    raise ValueError(f"Unsupported train.objective.combined_fusion_mode={mode!r}")


def _resolve_epoch_fusion_diff_weight(
    objective_cfg: Mapping[str, Any],
    *,
    epoch: int,
    total_epochs: int,
) -> float:
    base_weight = float(objective_cfg.get("fusion_diff_weight", 0.35))
    if base_weight < 0.0 or base_weight > 1.0:
        raise ValueError("train.objective.fusion_diff_weight must be in [0, 1]")

    schedule = str(objective_cfg.get("fusion_diff_weight_schedule", "linear_warmup")).strip().lower()
    if schedule in {"", "none", "fixed"}:
        return base_weight

    start_weight = float(objective_cfg.get("fusion_diff_weight_start", 0.0))
    if start_weight < 0.0 or start_weight > 1.0:
        raise ValueError("train.objective.fusion_diff_weight_start must be in [0, 1]")

    warmup_epochs_cfg = objective_cfg.get("fusion_diff_weight_warmup_epochs")
    if warmup_epochs_cfg is None:
        warmup_epochs = max(1, int(total_epochs) // 3)
    else:
        warmup_epochs = int(warmup_epochs_cfg)
    warmup_epochs = max(1, warmup_epochs)

    if schedule in {"linear", "linear_warmup"}:
        step = min(max(int(epoch), 1), warmup_epochs)
        ratio = float(step) / float(warmup_epochs)
        return start_weight + (base_weight - start_weight) * ratio
    raise ValueError(f"Unsupported train.objective.fusion_diff_weight_schedule={schedule!r}")


def _estimate_dual_port_calibration(
    model: torch.nn.Module,
    *,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    wavelength_a_um: float,
    wavelength_b_um: float,
    device: torch.device,
    gain_min: float,
    gain_max: float,
    std_eps: float,
) -> Dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        raw_a = _coerce_branch_output(model(x_a, wavelength_um=wavelength_a_um), device=device)
        raw_b = _coerce_branch_output(model(x_b, wavelength_um=wavelength_b_um), device=device)

    mu_a = raw_a.mean(dim=0)
    mu_b = raw_b.mean(dim=0)
    std_a = raw_a.std(dim=0, unbiased=False)
    std_b = raw_b.std(dim=0, unbiased=False).clamp_min(std_eps)

    gain = (std_a / std_b).clamp(min=gain_min, max=gain_max)
    offset = mu_a - gain * mu_b
    return {
        "gain": gain.detach(),
        "offset": offset.detach(),
    }


def _apply_dual_port_calibration(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    *,
    calibration: Mapping[str, torch.Tensor] | None,
    objective_cfg: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not calibration:
        return raw_a, raw_b
    gain = calibration.get("gain")
    offset = calibration.get("offset")
    if gain is None or offset is None:
        return raw_a, raw_b

    gain_t = torch.as_tensor(gain, dtype=raw_b.dtype, device=raw_b.device).reshape(1, -1)
    offset_t = torch.as_tensor(offset, dtype=raw_b.dtype, device=raw_b.device).reshape(1, -1)
    raw_b_calibrated = raw_b * gain_t + offset_t
    if objective_cfg.get("calibration_clamp_min") is not None:
        raw_b_calibrated = raw_b_calibrated.clamp_min(float(objective_cfg.get("calibration_clamp_min")))
    return raw_a, raw_b_calibrated


def _apply_train_input_noise(
    xb: torch.Tensor,
    *,
    std: float,
    input_low: float,
    input_high: float,
) -> torch.Tensor:
    if std <= 0.0:
        return xb
    noisy = xb + torch.randn_like(xb) * std
    return noisy.clamp_(min=input_low, max=input_high)


def _fetch_mnist_arrays(data_home: Path) -> tuple[np.ndarray, np.ndarray]:
    data_home.mkdir(parents=True, exist_ok=True)
    fetch_kwargs = dict(
        name="mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
        data_home=str(data_home),
    )
    try:
        inputs, labels = fetch_openml(
            parser="auto",
            **fetch_kwargs,
        )
    except TypeError as exc:
        if "parser" not in str(exc):
            raise
        inputs, labels = fetch_openml(**fetch_kwargs)
    x = np.asarray(inputs, dtype=np.float32) / 255.0
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != 784:
        raise ValueError(f"MNIST inputs must be [N, 784], got {x.shape}")
    if y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("MNIST labels shape mismatch")
    return x, y


def _load_fashion_mnist_raw_arrays(data_home: Path) -> Dict[str, Any]:
    data_home.mkdir(parents=True, exist_ok=True)
    try:
        try:
            train_ds = FashionMNIST(root=str(data_home), train=True, download=False)
            eval_ds = FashionMNIST(root=str(data_home), train=False, download=False)
        except Exception:
            train_ds = FashionMNIST(root=str(data_home), train=True, download=True)
            eval_ds = FashionMNIST(root=str(data_home), train=False, download=True)
        x_train = np.asarray(train_ds.data, dtype=np.float32).reshape(-1, 784) / 255.0
        y_train = np.asarray(train_ds.targets, dtype=np.int64).reshape(-1)
        x_eval = np.asarray(eval_ds.data, dtype=np.float32).reshape(-1, 784) / 255.0
        y_eval = np.asarray(eval_ds.targets, dtype=np.int64).reshape(-1)
    except Exception:
        fetch_kwargs = dict(
            name="Fashion-MNIST",
            version=1,
            return_X_y=True,
            as_frame=False,
            data_home=str(data_home / "openml"),
        )
        try:
            inputs, labels = fetch_openml(parser="auto", **fetch_kwargs)
        except TypeError as exc:
            if "parser" not in str(exc):
                raise
            inputs, labels = fetch_openml(**fetch_kwargs)
        x_all = np.asarray(inputs, dtype=np.float32) / 255.0
        y_all = np.asarray(labels, dtype=np.int64).reshape(-1)
        if x_all.ndim != 2 or x_all.shape[1] != 784:
            raise ValueError(f"Fashion-MNIST inputs must be [N, 784], got {x_all.shape}")
        if y_all.ndim != 1 or y_all.shape[0] != x_all.shape[0]:
            raise ValueError("Fashion-MNIST labels shape mismatch")
        if x_all.shape[0] < 70000:
            raise ValueError(
                f"Fashion-MNIST OpenML fallback expected 70000 samples, got {x_all.shape[0]}"
            )
        x_train = x_all[:60000]
        y_train = y_all[:60000]
        x_eval = x_all[60000:70000]
        y_eval = y_all[60000:70000]
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_eval": x_eval,
        "y_eval": y_eval,
        "class_names": list(FASHION_MNIST_CLASS_NAMES),
    }


def _resolve_local_image_root(dataset_cfg: Mapping[str, Any]) -> Path:
    root_value = (
        dataset_cfg.get("root")
        or dataset_cfg.get("data_root")
        or dataset_cfg.get("dataset_root")
    )
    if root_value is None:
        raise ValueError(
            "imagenet_scene_subset_single requires train.dataset.root "
            "pointing at the local image dataset"
        )
    root = Path(str(root_value)).expanduser()
    if not root.is_absolute():
        root = get_project_root() / root
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Image dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Image dataset root must be a directory: {root}")
    return root


def _resolve_image_size(value: Any) -> tuple[int, int]:
    if value is None:
        return (32, 32)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError("train.dataset.image_size must be an int or [height, width]")
        height = int(value[0])
        width = int(value[1])
    else:
        height = width = int(value)
    if height <= 0 or width <= 0:
        raise ValueError("train.dataset.image_size values must be positive")
    return height, width


def _list_scene_image_records(
    split_dir: Path,
    *,
    class_names: Sequence[str],
    exclude_subclasses: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[list[tuple[Path, int]], Dict[str, int]]:
    records: list[tuple[Path, int]] = []
    counts: Dict[str, int] = {}
    exclusions = dict(exclude_subclasses or {})
    for class_idx, class_name in enumerate(class_names):
        class_dir = split_dir / str(class_name)
        if not class_dir.exists():
            counts[str(class_name)] = 0
            continue
        if not class_dir.is_dir():
            raise NotADirectoryError(f"Expected class directory: {class_dir}")
        excluded_names = {
            str(value).strip().lower().replace(" ", "_")
            for value in (exclusions.get(str(class_name)) or [])
        }
        class_paths = []
        for path in sorted(class_dir.rglob("*")):
            if not (
                path.is_file()
                and path.suffix.lower() in IMAGE_DATASET_EXTENSIONS
                and not path.name.startswith(".")
            ):
                continue
            relative_parts = path.relative_to(class_dir).parts
            subclass_name = (
                str(relative_parts[0]).strip().lower().replace(" ", "_")
                if relative_parts
                else ""
            )
            if subclass_name in excluded_names:
                continue
            class_paths.append(path)
        counts[str(class_name)] = int(len(class_paths))
        records.extend((path, int(class_idx)) for path in class_paths)
    return records, counts


def _load_scene_image_matrix(
    records: Sequence[tuple[Path, int]],
    *,
    image_size: tuple[int, int],
    color_mode: str = "grayscale",
) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("Cannot load an empty image split")
    height, width = image_size
    normalized_mode = str(color_mode).strip().lower()
    if normalized_mode in {"gray", "grey", "grayscale", "l"}:
        pil_mode = "L"
        channel_count = 1
        normalized_mode = "grayscale"
    elif normalized_mode in {"rgb", "color", "colour"}:
        pil_mode = "RGB"
        channel_count = 3
        normalized_mode = "rgb"
    else:
        raise ValueError(
            "train.dataset.image_color_mode must be 'grayscale' or 'rgb', "
            f"got {color_mode!r}"
        )
    images = np.empty((len(records), height * width * channel_count), dtype=np.float32)
    labels = np.empty((len(records),), dtype=np.int64)
    for idx, (path, label) in enumerate(records):
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert(pil_mode)
                image = image.resize((width, height), resample=Image.BILINEAR)
                images[idx] = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
        except Exception as exc:
            raise RuntimeError(f"Failed to load image {path}: {exc}") from exc
        labels[idx] = int(label)
    return images, labels


def _make_scene_spatial_color16_features(
    images_flat: np.ndarray,
    *,
    image_size: tuple[int, int],
    color_mode: str,
) -> np.ndarray:
    if str(color_mode).strip().lower() not in {"rgb", "color", "colour"}:
        raise ValueError("spatial_color16 features require train.dataset.image_color_mode='rgb'")
    height, width = image_size
    images = np.asarray(images_flat, dtype=np.float32).reshape(-1, height, width, 3)
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("spatial_color16 requires even image height and width")
    half_h = height // 2
    half_w = width // 2
    pooled = images.reshape(-1, 2, half_h, 2, half_w, 3).mean(axis=(2, 4))
    spatial_rgb = pooled.reshape(-1, 12)
    channel_std = images.std(axis=(1, 2))
    gray = np.tensordot(images, np.asarray([0.299, 0.587, 0.114], dtype=np.float32), axes=([-1], [0]))
    gradient_y = np.diff(gray, axis=1)
    gradient_x = np.diff(gray, axis=2)
    texture = np.sqrt(
        np.square(gradient_y[:, :, :-1]) + np.square(gradient_x[:, :-1, :])
    ).mean(axis=(1, 2), keepdims=False).reshape(-1, 1)
    return np.concatenate([spatial_rgb, channel_std, texture], axis=1).astype(np.float32)


def _make_scene_structure16_features(
    images_flat: np.ndarray,
    *,
    image_size: tuple[int, int],
    color_mode: str,
) -> np.ndarray:
    if str(color_mode).strip().lower() not in {"rgb", "color", "colour"}:
        raise ValueError("scene_structure16 features require train.dataset.image_color_mode='rgb'")
    height, width = image_size
    if height < 6 or width < 2:
        raise ValueError("scene_structure16 requires image dimensions of at least 6 x 2")

    images = np.asarray(images_flat, dtype=np.float32).reshape(-1, height, width, 3)
    band_edges = np.linspace(0, height, num=4, dtype=np.int64)
    bands = [
        images[:, band_edges[index] : band_edges[index + 1], :, :]
        for index in range(3)
    ]
    band_rgb = [band.mean(axis=(1, 2)) for band in bands]

    red, green, blue = (images[..., channel] for channel in range(3))
    brightness = (0.299 * red) + (0.587 * green) + (0.114 * blue)
    top_brightness = brightness[:, : band_edges[1], :].mean(axis=(1, 2))
    bottom_brightness = brightness[:, band_edges[2] :, :].mean(axis=(1, 2))
    top_blue_dominance = (blue[:, : band_edges[1], :] - red[:, : band_edges[1], :]).mean(
        axis=(1, 2)
    )
    bottom_green_dominance = (
        green[:, band_edges[2] :, :] - 0.5 * (red[:, band_edges[2] :, :] + blue[:, band_edges[2] :, :])
    ).mean(axis=(1, 2))

    gradient_y = np.diff(brightness, axis=1)
    gradient_x = np.diff(brightness, axis=2)
    horizontal_edge_energy = np.abs(gradient_y).mean(axis=(1, 2))
    vertical_edge_energy = np.abs(gradient_x).mean(axis=(1, 2))
    top_texture = np.abs(gradient_x[:, : max(1, band_edges[1] - 1), :]).mean(axis=(1, 2))
    bottom_texture = np.abs(gradient_x[:, band_edges[2] :, :]).mean(axis=(1, 2))

    return np.concatenate(
        [
            band_rgb[0],
            band_rgb[1],
            band_rgb[2],
            (top_brightness - bottom_brightness).reshape(-1, 1),
            top_blue_dominance.reshape(-1, 1),
            bottom_green_dominance.reshape(-1, 1),
            horizontal_edge_energy.reshape(-1, 1),
            vertical_edge_energy.reshape(-1, 1),
            top_texture.reshape(-1, 1),
            bottom_texture.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)


def _split_scene_records(
    records: Sequence[tuple[Path, int]],
    *,
    train_limit: int | None,
    eval_limit: int | None,
    eval_fraction: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    labels = np.asarray([label for _, label in records], dtype=np.int64)
    if labels.size == 0:
        raise ValueError("Image dataset contains no images")
    if len(set(labels.tolist())) < 2:
        raise ValueError("Image dataset needs at least two populated classes")
    if eval_limit is not None:
        test_size: float | int = int(eval_limit)
    else:
        test_size = float(eval_fraction)
    train_size: int | None = None if train_limit is None else int(train_limit)
    train_idx, eval_idx = train_test_split(
        np.arange(labels.shape[0], dtype=np.int64),
        train_size=train_size,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )
    return (
        [records[int(idx)] for idx in np.asarray(train_idx, dtype=np.int64).tolist()],
        [records[int(idx)] for idx in np.asarray(eval_idx, dtype=np.int64).tolist()],
    )


def _balanced_scene_records(
    records: Sequence[tuple[Path, int]],
    *,
    class_count: int,
    limit: int | None,
    seed: int,
) -> list[tuple[Path, int]]:
    labels = np.asarray([label for _, label in records], dtype=np.int64)
    if labels.size == 0:
        raise ValueError("Cannot balance an empty image split")
    if limit is None:
        per_class = min(
            int(np.sum(labels == class_idx)) for class_idx in range(class_count)
        )
    else:
        if int(limit) <= 0 or int(limit) % int(class_count) != 0:
            raise ValueError(
                "Balanced image split limits must be positive multiples of the "
                f"class count ({class_count}), got {limit}"
            )
        per_class = int(limit) // int(class_count)
    if per_class <= 0:
        raise ValueError("Balanced image split requires at least one image per class")

    rng = np.random.default_rng(seed)
    selected: list[tuple[Path, int]] = []
    for class_idx in range(class_count):
        class_indices = np.flatnonzero(labels == class_idx)
        if class_indices.size < per_class:
            raise ValueError(
                "Not enough images for balanced sampling: "
                f"class_index={class_idx}, available={class_indices.size}, requested={per_class}"
            )
        chosen = rng.permutation(class_indices)[:per_class]
        selected.extend(records[int(idx)] for idx in chosen.tolist())
    rng.shuffle(selected)
    return selected


def _balanced_scene_train_eval_split(
    records: Sequence[tuple[Path, int]],
    *,
    class_count: int,
    train_limit: int | None,
    eval_limit: int | None,
    eval_fraction: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    labels = np.asarray([label for _, label in records], dtype=np.int64)
    if labels.size == 0:
        raise ValueError("Image dataset contains no images")
    if train_limit is None or eval_limit is None:
        raise ValueError(
            "Balanced splitting from one image directory requires both "
            "train_limit and eval_limit"
        )
    if int(train_limit) <= 0 or int(eval_limit) <= 0:
        raise ValueError("Balanced image split limits must be positive")
    if int(train_limit) % class_count != 0 or int(eval_limit) % class_count != 0:
        raise ValueError(
            "Balanced image split limits must be divisible by the class count: "
            f"train_limit={train_limit}, eval_limit={eval_limit}, class_count={class_count}"
        )
    per_train = int(train_limit) // class_count
    per_eval = int(eval_limit) // class_count
    rng = np.random.default_rng(seed)
    train_records: list[tuple[Path, int]] = []
    eval_records: list[tuple[Path, int]] = []
    for class_idx in range(class_count):
        class_indices = np.flatnonzero(labels == class_idx)
        required = per_train + per_eval
        if class_indices.size < required:
            raise ValueError(
                "Not enough images for balanced train/eval sampling: "
                f"class_index={class_idx}, available={class_indices.size}, requested={required}"
            )
        chosen = rng.permutation(class_indices)[:required]
        train_records.extend(records[int(idx)] for idx in chosen[:per_train].tolist())
        eval_records.extend(records[int(idx)] for idx in chosen[per_train:].tolist())
    rng.shuffle(train_records)
    rng.shuffle(eval_records)
    return train_records, eval_records


def _make_imagenet_scene_subset_single_dataset(
    *,
    dataset_cfg: Mapping[str, Any],
    num_inputs: int,
    num_outputs: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    Dict[str, Any],
]:
    if num_inputs != 16:
        raise ValueError(
            f"imagenet_scene_subset_single requires model input_dim=16, got {num_inputs}"
        )
    if num_outputs != 10:
        raise ValueError(
            f"imagenet_scene_subset_single requires model output_dim=10, got {num_outputs}"
        )

    root = _resolve_local_image_root(dataset_cfg)
    train_dir = root / str(dataset_cfg.get("train_split", "train"))
    eval_dir = root / str(dataset_cfg.get("eval_split", "val"))
    exclude_subclasses = dict(dataset_cfg.get("exclude_subclasses") or {})
    class_names = dataset_cfg.get("classes") or dataset_cfg.get("class_names")
    if class_names is None:
        if not train_dir.exists():
            raise FileNotFoundError(f"Image train split does not exist: {train_dir}")
        class_names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    class_names = [str(name) for name in class_names]
    if not class_names:
        raise ValueError("imagenet_scene_subset_single requires at least one class")
    if len(class_names) > num_outputs:
        raise ValueError(
            f"Selected class count {len(class_names)} exceeds model output_dim={num_outputs}"
        )

    train_records_all, train_counts_all = _list_scene_image_records(
        train_dir,
        class_names=class_names,
        exclude_subclasses=exclude_subclasses,
    )
    eval_records_all, eval_counts_all = (
        _list_scene_image_records(
            eval_dir,
            class_names=class_names,
            exclude_subclasses=exclude_subclasses,
        )
        if eval_dir.exists()
        else ([], {name: 0 for name in class_names})
    )
    train_limit = dataset_cfg.get("train_limit")
    eval_limit = dataset_cfg.get("eval_limit")
    train_limit_int = None if train_limit is None else int(train_limit)
    eval_limit_int = None if eval_limit is None else int(eval_limit)
    if train_limit_int is not None and train_limit_int <= 0:
        train_limit_int = None
    if eval_limit_int is not None and eval_limit_int <= 0:
        eval_limit_int = None
    eval_fraction = float(dataset_cfg.get("eval_fraction", 0.2))
    if eval_fraction <= 0.0 or eval_fraction >= 1.0:
        raise ValueError("train.dataset.eval_fraction must be in (0, 1)")
    balance_classes = bool(dataset_cfg.get("balance_classes", False))

    if eval_records_all:
        if balance_classes:
            train_records = _balanced_scene_records(
                train_records_all,
                class_count=len(class_names),
                limit=train_limit_int,
                seed=seed,
            )
            eval_records = _balanced_scene_records(
                eval_records_all,
                class_count=len(class_names),
                limit=eval_limit_int,
                seed=seed + 1,
            )
        else:
            train_labels_all = np.asarray([label for _, label in train_records_all], dtype=np.int64)
            eval_labels_all = np.asarray([label for _, label in eval_records_all], dtype=np.int64)
            train_idx = _subsample_split_indices(
                train_labels_all,
                limit=train_limit_int,
                seed=seed,
            )
            eval_idx = _subsample_split_indices(
                eval_labels_all,
                limit=eval_limit_int,
                seed=seed + 1,
            )
            train_records = [train_records_all[int(idx)] for idx in train_idx.tolist()]
            eval_records = [eval_records_all[int(idx)] for idx in eval_idx.tolist()]
        split_source = "explicit_train_val_dirs"
    else:
        if balance_classes:
            train_records, eval_records = _balanced_scene_train_eval_split(
                train_records_all,
                class_count=len(class_names),
                train_limit=train_limit_int,
                eval_limit=eval_limit_int,
                eval_fraction=eval_fraction,
                seed=seed,
            )
        else:
            train_records, eval_records = _split_scene_records(
                train_records_all,
                train_limit=train_limit_int,
                eval_limit=eval_limit_int,
                eval_fraction=eval_fraction,
                seed=seed,
            )
        split_source = "stratified_split_from_train_dir"

    image_size = _resolve_image_size(dataset_cfg.get("image_size", 32))
    image_color_mode = str(dataset_cfg.get("image_color_mode", "grayscale"))
    x_train_raw, y_train_idx = _load_scene_image_matrix(
        train_records,
        image_size=image_size,
        color_mode=image_color_mode,
    )
    x_eval_raw, y_eval_idx = _load_scene_image_matrix(
        eval_records,
        image_size=image_size,
        color_mode=image_color_mode,
    )

    feature_kind = str(dataset_cfg.get("feature_kind", "pca16")).strip().lower()
    pca_components = int(dataset_cfg.get("pca_components", num_inputs))
    if feature_kind in {"pca", "pca16"}:
        if pca_components != num_inputs:
            raise ValueError(
                "imagenet_scene_subset_single PCA mode requires "
                f"train.dataset.pca_components == model input_dim={num_inputs}, got {pca_components}"
            )
        pca = PCA(
            n_components=pca_components,
            svd_solver=str(dataset_cfg.get("pca_solver", "randomized")),
            whiten=bool(dataset_cfg.get("pca_whiten", False)),
            random_state=seed,
        )
        x_train_features = pca.fit_transform(x_train_raw).astype(np.float32)
        x_eval_features = pca.transform(x_eval_raw).astype(np.float32)
        explained = float(np.sum(np.asarray(pca.explained_variance_ratio_, dtype=np.float64)))
    elif feature_kind in {"spatial_color16", "color16", "handcrafted16"}:
        x_train_features = _make_scene_spatial_color16_features(
            x_train_raw,
            image_size=image_size,
            color_mode=image_color_mode,
        )
        x_eval_features = _make_scene_spatial_color16_features(
            x_eval_raw,
            image_size=image_size,
            color_mode=image_color_mode,
        )
        if x_train_features.shape[1] != num_inputs:
            raise ValueError(
                "spatial_color16 feature count must match model input_dim: "
                f"features={x_train_features.shape[1]}, input_dim={num_inputs}"
            )
        explained = 0.0
    elif feature_kind in {"scene_structure16", "structure16", "scene16"}:
        x_train_features = _make_scene_structure16_features(
            x_train_raw,
            image_size=image_size,
            color_mode=image_color_mode,
        )
        x_eval_features = _make_scene_structure16_features(
            x_eval_raw,
            image_size=image_size,
            color_mode=image_color_mode,
        )
        if x_train_features.shape[1] != num_inputs:
            raise ValueError(
                "scene_structure16 feature count must match model input_dim: "
                f"features={x_train_features.shape[1]}, input_dim={num_inputs}"
            )
        explained = 0.0
    else:
        raise ValueError(
            "train.dataset.feature_kind must be 'pca16', 'spatial_color16', "
            "or 'scene_structure16', "
            f"got {feature_kind!r}"
        )
    input_low = float(dataset_cfg.get("input_low", 0.0))
    input_high = float(dataset_cfg.get("input_high", 2.0))
    if input_high <= input_low:
        raise ValueError("train.dataset.input_high must be greater than input_low")
    x_train, x_eval = _scale_feature_branch(
        x_train_features,
        x_eval_features,
        input_low=input_low,
        input_high=input_high,
    )
    y_train = _one_hot_with_output_dim(y_train_idx, output_dim=num_outputs)
    y_eval = _one_hot_with_output_dim(y_eval_idx, output_dim=num_outputs)
    train_counts_used = {
        class_name: int(np.sum(y_train_idx == class_idx))
        for class_idx, class_name in enumerate(class_names)
    }
    eval_counts_used = {
        class_name: int(np.sum(y_eval_idx == class_idx))
        for class_idx, class_name in enumerate(class_names)
    }
    dataset_summary = {
        "dataset_name": "imagenet_scene_subset_single",
        "root": str(root),
        "train_split": str(train_dir),
        "eval_split": str(eval_dir),
        "split_source": split_source,
        "exclude_subclasses": exclude_subclasses,
        "balance_classes": balance_classes,
        "class_names_active": list(class_names),
        "class_names_padded": _pad_class_names(class_names, output_dim=num_outputs),
        "selected_class_indices": list(range(len(class_names))),
        "train_counts_all": train_counts_all,
        "eval_counts_all": eval_counts_all,
        "train_counts_used": train_counts_used,
        "eval_counts_used": eval_counts_used,
        "train_samples": int(x_train.shape[0]),
        "eval_samples": int(x_eval.shape[0]),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "image_color_mode": image_color_mode,
        "pca_components": int(pca_components),
        "feature_kind": feature_kind,
        "component_split": "single_branch",
        "input_component_indices": list(range(int(pca_components))),
        "scaled_input_low": float(input_low),
        "scaled_input_high": float(input_high),
        "input_train_min": float(np.min(x_train)),
        "input_train_max": float(np.max(x_train)),
        "input_train_mean": float(np.mean(x_train)),
        "input_train_std": float(np.std(x_train)),
        "explained_variance_ratio_sum": explained,
    }
    return x_train, x_eval, y_train, y_eval, explained, dataset_summary


def _subsample_unstratified_indices(
    total_count: int,
    *,
    limit: int | None,
    seed: int,
) -> np.ndarray:
    if limit is None or limit <= 0 or limit >= int(total_count):
        return np.arange(int(total_count), dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(int(total_count))[: int(limit)]
    return np.asarray(chosen, dtype=np.int64)


def _resolve_fashion_variant_names(
    source_class_name: str,
    *,
    shape_kind: str,
    variant_mode: str = "four_way",
) -> list[str]:
    mode = str(variant_mode).strip().lower()
    if mode in {"chest_pattern_then_stripes", "chest_pattern_stripes", "pattern_then_stripes"}:
        return [
            f"{source_class_name}_chest_pattern",
            f"{source_class_name}_no_pattern_horizontal_stripes",
            f"{source_class_name}_no_pattern_vertical_stripes",
        ]
    if mode in FASHION_ICON_VARIANT_SETS:
        return [f"{source_class_name}_{icon_kind}" for icon_kind in FASHION_ICON_VARIANT_SETS[mode]]
    shape_label = "center_circle" if str(shape_kind).strip().lower() == "circle" else "center_square"
    return [
        f"{source_class_name}_horizontal_stripes",
        f"{source_class_name}_vertical_stripes",
        f"{source_class_name}_{shape_label}",
        f"{source_class_name}_plain",
    ]


def _draw_mask_disk(mask: np.ndarray, center_y: float, center_x: float, radius: float) -> None:
    h, w = mask.shape
    yy, xx = np.ogrid[:h, :w]
    mask |= (yy - float(center_y)) ** 2 + (xx - float(center_x)) ** 2 <= float(radius) ** 2


def _draw_mask_line(
    mask: np.ndarray,
    y0: float,
    x0: float,
    y1: float,
    x1: float,
    *,
    thickness: int = 1,
) -> None:
    steps = max(int(np.ceil(abs(float(y1) - float(y0)))), int(np.ceil(abs(float(x1) - float(x0)))), 1)
    radius = max(float(thickness) * 0.55, 0.6)
    ys = np.linspace(float(y0), float(y1), steps + 1)
    xs = np.linspace(float(x0), float(x1), steps + 1)
    for y, x in zip(ys, xs):
        _draw_mask_disk(mask, y, x, radius)


def _draw_mask_circle_outline(mask: np.ndarray, center_y: float, center_x: float, radius: float) -> None:
    point_count = max(18, int(round(float(radius) * 6.0)))
    angles = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    for angle in angles:
        _draw_mask_disk(
            mask,
            float(center_y) + float(radius) * float(np.sin(angle)),
            float(center_x) + float(radius) * float(np.cos(angle)),
            0.75,
        )


def _make_fashion_icon_mask(
    *,
    kind: str,
    image_shape: tuple[int, int] = (28, 28),
    offset_y: int = 0,
    offset_x: int = 0,
    style_index: int = 0,
) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=bool)
    cy = float(h // 2 - 1 + int(offset_y))
    cx = float(w // 2 + int(offset_x))
    key = str(kind).strip().lower()

    if key in {"triangle", "solid_triangle"}:
        top_y = cy - 6.2
        bottom_y = cy + 5.8
        max_half_width = 6.0
        y_start = max(0, int(np.floor(top_y)))
        y_end = min(h - 1, int(np.ceil(bottom_y)))
        for y in range(y_start, y_end + 1):
            ratio = np.clip((float(y) - top_y) / max(bottom_y - top_y, 1e-6), 0.0, 1.0)
            half_width = max_half_width * ratio
            _draw_mask_line(mask, y, cx - half_width, y, cx + half_width, thickness=1)
        _draw_mask_line(mask, top_y, cx, bottom_y, cx - max_half_width, thickness=1)
        _draw_mask_line(mask, top_y, cx, bottom_y, cx + max_half_width, thickness=1)
        _draw_mask_line(mask, bottom_y, cx - max_half_width, bottom_y, cx + max_half_width, thickness=1)
        return mask

    if key in {"diagonal_slash", "slash"}:
        _draw_mask_line(mask, cy + 6.2, cx - 6.2, cy - 6.2, cx + 6.2, thickness=3)
        _draw_mask_disk(mask, cy + 6.2, cx - 6.2, 1.3)
        _draw_mask_disk(mask, cy - 6.2, cx + 6.2, 1.3)
        return mask

    if key == "star":
        star_cy = cy - 1.4
        points: list[tuple[float, float]] = []
        for idx in range(10):
            angle = -np.pi / 2.0 + idx * np.pi / 5.0
            radius = 6.2 if idx % 2 == 0 else 2.7
            points.append((star_cy + radius * float(np.sin(angle)), cx + radius * float(np.cos(angle))))
        for (y0, x0), (y1, x1) in zip(points, points[1:] + points[:1]):
            _draw_mask_line(mask, y0, x0, y1, x1, thickness=1)
        return mask

    if key == "heart":
        heart_cy = cy + 1.4
        _draw_mask_disk(mask, heart_cy - 2.6, cx - 2.5, 3.2)
        _draw_mask_disk(mask, heart_cy - 2.6, cx + 2.5, 3.2)
        band_widths = [
            (-2.4, 5.6),
            (-1.2, 6.0),
            (0.0, 5.6),
            (1.2, 5.0),
            (2.4, 4.3),
            (3.6, 3.3),
            (4.8, 2.1),
            (5.8, 1.0),
        ]
        for dy, half_width in band_widths:
            _draw_mask_line(mask, heart_cy + dy, cx - half_width, heart_cy + dy, cx + half_width, thickness=2)
        _draw_mask_disk(mask, heart_cy + 5.9, cx, 1.2)
        return mask

    if key == "flower":
        _draw_mask_disk(mask, cy, cx, 1.2)
        petal_offsets = [(-2.8, 0.0), (-1.4, -2.4), (1.4, -2.4), (1.4, 2.4), (-1.4, 2.4)]
        for dy, dx in petal_offsets:
            _draw_mask_disk(mask, cy + dy, cx + dx, 1.45)
        _draw_mask_line(mask, cy + 2.5, cx, cy + 6.0, cx, thickness=1)
        _draw_mask_line(mask, cy + 5.0, cx, cy + 4.0, cx - 2.0, thickness=1)
        _draw_mask_line(mask, cy + 5.5, cx, cy + 4.6, cx + 1.8, thickness=1)
        return mask

    if key == "apple":
        _draw_mask_disk(mask, cy + 0.9, cx - 2.2, 2.8)
        _draw_mask_disk(mask, cy + 0.9, cx + 2.2, 2.8)
        _draw_mask_disk(mask, cy + 3.0, cx - 0.8, 2.7)
        _draw_mask_disk(mask, cy + 3.0, cx + 0.8, 2.7)
        yy, xx = np.ogrid[:h, :w]
        top_notch = (yy < float(cy - 2.0)) & (np.abs(xx - float(cx)) <= 1.2)
        mask[top_notch] = False
        _draw_mask_line(mask, cy - 2.8, cx, cy - 5.6, cx + 0.8, thickness=2)
        _draw_mask_line(mask, cy - 4.9, cx + 0.6, cy - 6.4, cx + 3.2, thickness=1)
        _draw_mask_line(mask, cy - 5.9, cx + 1.7, cy - 6.5, cx + 3.6, thickness=1)
        _draw_mask_disk(mask, cy - 6.1, cx + 3.2, 1.0)
        return mask

    if key == "smiley":
        _draw_mask_circle_outline(mask, cy, cx, 5.7)
        _draw_mask_disk(mask, cy - 1.7, cx - 2.1, 0.95)
        _draw_mask_disk(mask, cy - 1.7, cx + 2.1, 0.95)
        mouth_points = [
            (cy + 1.4, cx - 3.0),
            (cy + 2.7, cx - 1.5),
            (cy + 3.2, cx),
            (cy + 2.7, cx + 1.5),
            (cy + 1.4, cx + 3.0),
        ]
        for (y0, x0), (y1, x1) in zip(mouth_points[:-1], mouth_points[1:]):
            _draw_mask_line(mask, y0, x0, y1, x1, thickness=1)
        return mask

    if key == "bear":
        _draw_mask_disk(mask, cy - 4.0, cx - 4.2, 2.0)
        _draw_mask_disk(mask, cy - 4.0, cx + 4.2, 2.0)
        _draw_mask_disk(mask, cy, cx, 5.0)
        _draw_mask_disk(mask, cy + 1.8, cx, 1.8)
        _draw_mask_disk(mask, cy - 0.6, cx - 1.8, 0.75)
        _draw_mask_disk(mask, cy - 0.6, cx + 1.8, 0.75)
        _draw_mask_disk(mask, cy + 1.3, cx, 0.65)
        _draw_mask_line(mask, cy + 2.2, cx, cy + 3.3, cx - 1.1, thickness=1)
        _draw_mask_line(mask, cy + 2.2, cx, cy + 3.3, cx + 1.1, thickness=1)
        return mask

    if key == "letters":
        template_key = int(style_index) % 6
        if template_key == 0:
            _draw_mask_line(mask, cy + 5.2, cx - 4.6, cy - 5.2, cx, thickness=1)
            _draw_mask_line(mask, cy + 5.2, cx + 4.6, cy - 5.2, cx, thickness=1)
            _draw_mask_line(mask, cy + 0.8, cx - 2.3, cy + 0.8, cx + 2.3, thickness=1)
        elif template_key == 1:
            _draw_mask_line(mask, cy + 5.2, cx - 4.8, cy - 5.0, cx - 1.2, thickness=1)
            _draw_mask_line(mask, cy + 5.2, cx + 4.8, cy - 5.0, cx + 1.2, thickness=1)
            _draw_mask_line(mask, cy - 0.2, cx - 3.0, cy - 0.2, cx + 3.0, thickness=1)
            _draw_mask_line(mask, cy + 5.2, cx - 5.4, cy + 5.2, cx - 2.8, thickness=1)
            _draw_mask_line(mask, cy + 5.2, cx + 2.8, cy + 5.2, cx + 5.4, thickness=1)
        elif template_key == 2:
            _draw_mask_line(mask, cy + 5.2, cx - 4.8, cy - 5.0, cx - 4.8, thickness=1)
            _draw_mask_line(mask, cy + 5.2, cx + 4.8, cy - 5.0, cx + 4.8, thickness=1)
            _draw_mask_line(mask, cy + 5.2, cx - 4.8, cy - 0.4, cx, thickness=1)
            _draw_mask_line(mask, cy - 0.4, cx, cy + 5.2, cx + 4.8, thickness=1)
        elif template_key == 3:
            _draw_mask_line(mask, cy + 5.0, cx - 4.4, cy - 5.0, cx + 4.4, thickness=1)
            _draw_mask_line(mask, cy - 5.0, cx - 4.4, cy + 5.0, cx + 4.4, thickness=1)
        elif template_key == 4:
            _draw_mask_line(mask, cy + 5.2, cx - 4.5, cy - 5.0, cx - 4.5, thickness=1)
            _draw_mask_line(mask, cy + 5.2, cx + 4.2, cy - 5.0, cx + 4.2, thickness=1)
            _draw_mask_line(mask, cy, cx - 4.5, cy, cx + 4.2, thickness=1)
        else:
            _draw_mask_line(mask, cy + 5.2, cx - 4.6, cy - 5.0, cx - 4.6, thickness=1)
            _draw_mask_line(mask, cy + 5.2, cx - 4.6, cy - 5.0, cx + 2.0, thickness=1)
            _draw_mask_line(mask, cy, cx - 4.4, cy, cx + 2.2, thickness=1)
        return mask

    raise ValueError(f"Unsupported fashion icon kind={kind!r}")


def _make_fashion_pattern_mask(
    *,
    kind: str,
    shape_kind: str,
    image_shape: tuple[int, int] = (28, 28),
    stripe_width: int = 1,
    stripe_gap: int = 4,
    stripe_count: int = 3,
    stripe_length_fraction: float = 0.5,
    shape_size: int = 6,
) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=bool)
    if kind == "horizontal_stripes":
        stripe_total = int(stripe_count) * int(stripe_width) + max(int(stripe_count) - 1, 0) * int(stripe_gap)
        start_row = max(0, (h - stripe_total) // 2)
        stripe_len = max(1, min(w, int(round(w * float(stripe_length_fraction)))))
        x0 = max(0, (w - stripe_len) // 2)
        x1 = min(w, x0 + stripe_len)
        for stripe_idx in range(int(stripe_count)):
            row = start_row + stripe_idx * (int(stripe_width) + int(stripe_gap))
            mask[row:min(row + int(stripe_width), h), x0:x1] = True
    elif kind == "vertical_stripes":
        stripe_total = int(stripe_count) * int(stripe_width) + max(int(stripe_count) - 1, 0) * int(stripe_gap)
        start_col = max(0, (w - stripe_total) // 2)
        stripe_len = max(1, min(h, int(round(h * float(stripe_length_fraction)))))
        y0 = max(0, (h - stripe_len) // 2)
        y1 = min(h, y0 + stripe_len)
        for stripe_idx in range(int(stripe_count)):
            col = start_col + stripe_idx * (int(stripe_width) + int(stripe_gap))
            mask[y0:y1, col:min(col + int(stripe_width), w)] = True
    elif kind == "center_shape":
        center_y = h // 2
        center_x = w // 2
        half_size = max(int(shape_size) // 2, 1)
        if str(shape_kind).strip().lower() == "circle":
            yy, xx = np.ogrid[:h, :w]
            mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= float(half_size ** 2)
        else:
            y0 = max(center_y - half_size, 0)
            y1 = min(center_y + half_size, h)
            x0 = max(center_x - half_size, 0)
            x1 = min(center_x + half_size, w)
            mask[y0:y1, x0:x1] = True
    else:
        raise ValueError(f"Unsupported fashion pattern kind={kind!r}")
    return mask


def _apply_adaptive_high_contrast_pattern(
    image: np.ndarray,
    target_mask: np.ndarray,
    *,
    pattern_strength: float,
) -> None:
    if not bool(np.any(target_mask)):
        return
    strength = float(np.clip(pattern_strength, 0.0, 1.0))
    target_values = np.asarray(image[target_mask], dtype=np.float32)
    if target_values.size == 0:
        return
    reference = float(np.median(target_values))
    if reference >= 0.45:
        pattern_value = max(0.0, reference * (1.0 - strength))
    else:
        pattern_value = min(1.0, reference + (1.0 - reference) * strength)
    image[target_mask] = np.float32(pattern_value)


def _apply_fashion_variant_pattern(
    images_flat: np.ndarray,
    *,
    variant_kind: str,
    shape_kind: str,
    pattern_strength: float,
    foreground_threshold: float,
    stripe_width: int,
    stripe_gap: int,
    stripe_count: int,
    stripe_length_fraction: float,
    shape_size: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    images = np.asarray(images_flat, dtype=np.float32).reshape(-1, 28, 28).copy()
    variant_key = str(variant_kind).strip().lower()
    if variant_key == "plain":
        return images.reshape(images.shape[0], -1).astype(np.float32)

    if variant_key in FASHION_ICON_KINDS:
        local_rng = rng if rng is not None else np.random.default_rng(0)
        for idx in range(images.shape[0]):
            offset_y = int(local_rng.integers(-1, 2))
            offset_x = int(local_rng.integers(-1, 2))
            style_index = int(local_rng.integers(0, 6)) if variant_key == "letters" else 0
            mask = _make_fashion_icon_mask(
                kind=variant_key,
                image_shape=(28, 28),
                offset_y=offset_y,
                offset_x=offset_x,
                style_index=style_index,
            )
            foreground = images[idx] > float(foreground_threshold)
            target_mask = foreground & mask
            _apply_adaptive_high_contrast_pattern(
                images[idx],
                target_mask,
                pattern_strength=pattern_strength,
            )
        return images.reshape(images.shape[0], -1).astype(np.float32)

    if variant_key == "horizontal_stripes":
        mask = _make_fashion_pattern_mask(
            kind="horizontal_stripes",
            shape_kind=shape_kind,
            stripe_width=stripe_width,
            stripe_gap=stripe_gap,
            stripe_count=stripe_count,
            stripe_length_fraction=stripe_length_fraction,
            shape_size=shape_size,
        )
    elif variant_key == "vertical_stripes":
        mask = _make_fashion_pattern_mask(
            kind="vertical_stripes",
            shape_kind=shape_kind,
            stripe_width=stripe_width,
            stripe_gap=stripe_gap,
            stripe_count=stripe_count,
            stripe_length_fraction=stripe_length_fraction,
            shape_size=shape_size,
        )
    elif variant_key in {"center_shape", "shape"}:
        mask = _make_fashion_pattern_mask(
            kind="center_shape",
            shape_kind=shape_kind,
            stripe_width=stripe_width,
            stripe_gap=stripe_gap,
            stripe_count=stripe_count,
            stripe_length_fraction=stripe_length_fraction,
            shape_size=shape_size,
        )
    else:
        raise ValueError(f"Unsupported fashion variant kind={variant_kind!r}")

    foreground = images > float(foreground_threshold)
    target_masks = foreground & mask[np.newaxis, :, :]
    for idx in range(images.shape[0]):
        _apply_adaptive_high_contrast_pattern(
            images[idx],
            target_masks[idx],
            pattern_strength=pattern_strength,
        )
    return images.reshape(images.shape[0], -1).astype(np.float32)


def _compute_fashion_tshirt_chest_pattern_scores(
    images_flat: np.ndarray,
    *,
    foreground_threshold: float,
    chest_y0_fraction: float,
    chest_y1_fraction: float,
    chest_width_fraction: float,
) -> np.ndarray:
    images = np.asarray(images_flat, dtype=np.float32).reshape(-1, 28, 28)
    scores = np.zeros(images.shape[0], dtype=np.float32)
    for idx, image in enumerate(images):
        fg = image > float(foreground_threshold)
        ys, xs = np.where(fg)
        if xs.size == 0:
            continue
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())
        box_h = max(y_max - y_min + 1, 1)
        box_w = max(x_max - x_min + 1, 1)
        y0 = y_min + int(np.floor(box_h * float(chest_y0_fraction)))
        y1 = y_min + int(np.ceil(box_h * float(chest_y1_fraction)))
        center_x = (x_min + x_max) // 2
        half_w = max(1, int(np.ceil(box_w * float(chest_width_fraction) * 0.5)))
        x0 = max(0, center_x - half_w)
        x1 = min(image.shape[1], center_x + half_w + 1)
        y0 = max(0, min(image.shape[0] - 1, y0))
        y1 = max(y0 + 1, min(image.shape[0], y1))
        region = image[y0:y1, x0:x1]
        region_fg = fg[y0:y1, x0:x1]
        values = region[region_fg]
        if values.size < 4:
            continue
        contrast = float(np.std(values))
        row_edges = np.abs(np.diff(region, axis=0))
        col_edges = np.abs(np.diff(region, axis=1))
        edge_score = float(np.mean(row_edges)) + float(np.mean(col_edges))
        density = float(np.mean(region_fg))
        scores[idx] = contrast + 0.55 * edge_score + 0.05 * density
    return scores.astype(np.float32)


def _split_fashion_variant_dataset(
    images_flat: np.ndarray,
    *,
    source_class_name: str,
    seed: int,
    variant_mode: str = "four_way",
    shape_kind: str,
    pattern_strength: float,
    foreground_threshold: float,
    stripe_width: int = 1,
    stripe_gap: int = 4,
    stripe_count: int = 3,
    stripe_length_fraction: float = 0.5,
    shape_size: int = 6,
    chest_y0_fraction: float = 0.25,
    chest_y1_fraction: float = 0.68,
    chest_width_fraction: float = 0.55,
    confidence_keep_fraction: float = 1.0,
    confidence_min_margin: float = 0.0,
    confidence_split_name: str = "split",
    confidence_resample_to_before: bool = False,
    confidence_resample_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    images = np.asarray(images_flat, dtype=np.float32)
    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError(f"Expected Fashion-MNIST flat images [N, 784], got {images.shape}")
    variant_mode_key = str(variant_mode).strip().lower()
    variant_names = _resolve_fashion_variant_names(
        source_class_name,
        shape_kind=shape_kind,
        variant_mode=variant_mode_key,
    )
    if variant_mode_key in {"chest_pattern_then_stripes", "chest_pattern_stripes", "pattern_then_stripes"}:
        scores = _compute_fashion_tshirt_chest_pattern_scores(
            images,
            foreground_threshold=foreground_threshold,
            chest_y0_fraction=chest_y0_fraction,
            chest_y1_fraction=chest_y1_fraction,
            chest_width_fraction=chest_width_fraction,
        )
        order = np.argsort(scores, kind="stable")
        no_pattern_count = int(images.shape[0] // 2)
        no_pattern_indices = order[:no_pattern_count]
        chest_pattern_indices = order[no_pattern_count:]
        rng = np.random.default_rng(seed)
        no_pattern_perm = rng.permutation(no_pattern_indices)
        horizontal_indices, vertical_indices = np.array_split(no_pattern_perm, 2)
        out_images = [
            images[chest_pattern_indices].astype(np.float32),
            _apply_fashion_variant_pattern(
                images[horizontal_indices],
                variant_kind="horizontal_stripes",
                shape_kind=shape_kind,
                pattern_strength=pattern_strength,
                foreground_threshold=foreground_threshold,
                stripe_width=stripe_width,
                stripe_gap=stripe_gap,
                stripe_count=stripe_count,
                stripe_length_fraction=stripe_length_fraction,
                shape_size=shape_size,
            ),
            _apply_fashion_variant_pattern(
                images[vertical_indices],
                variant_kind="vertical_stripes",
                shape_kind=shape_kind,
                pattern_strength=pattern_strength,
                foreground_threshold=foreground_threshold,
                stripe_width=stripe_width,
                stripe_gap=stripe_gap,
                stripe_count=stripe_count,
                stripe_length_fraction=stripe_length_fraction,
                shape_size=shape_size,
            ),
        ]
        out_scores = [
            scores[chest_pattern_indices].astype(np.float32),
            _compute_fashion_variant_fit_scores(
                images[horizontal_indices],
                variant_kind="horizontal_stripes",
                shape_kind=shape_kind,
                foreground_threshold=foreground_threshold,
                stripe_width=stripe_width,
                stripe_gap=stripe_gap,
                stripe_count=stripe_count,
                stripe_length_fraction=stripe_length_fraction,
                shape_size=shape_size,
            ),
            _compute_fashion_variant_fit_scores(
                images[vertical_indices],
                variant_kind="vertical_stripes",
                shape_kind=shape_kind,
                foreground_threshold=foreground_threshold,
                stripe_width=stripe_width,
                stripe_gap=stripe_gap,
                stripe_count=stripe_count,
                stripe_length_fraction=stripe_length_fraction,
                shape_size=shape_size,
            ),
        ]
        out_labels = [
            np.full(chest_pattern_indices.shape[0], 0, dtype=np.int64),
            np.full(horizontal_indices.shape[0], 1, dtype=np.int64),
            np.full(vertical_indices.shape[0], 2, dtype=np.int64),
        ]
        x_out = np.concatenate(out_images, axis=0).astype(np.float32)
        y_out = np.concatenate(out_labels, axis=0).astype(np.int64)
        score_out = np.concatenate(out_scores, axis=0).astype(np.float32)
        filter_summary: Dict[str, Any] | None = None
        if confidence_keep_fraction < 1.0 or confidence_min_margin > 0.0:
            x_out, y_out, filter_summary = _filter_fashion_multiclass_score_split_by_confidence(
                x_out,
                y_out,
                score_out,
                class_names=variant_names,
                keep_fraction=confidence_keep_fraction,
                min_margin=confidence_min_margin,
                split_name=confidence_split_name,
            )
        if filter_summary is not None and bool(confidence_resample_to_before):
            x_out, y_out, _ = _resample_fashion_split_to_class_counts(
                x_out,
                y_out,
                class_names=variant_names,
                target_counts=filter_summary["class_counts_before"],
                seed=confidence_resample_seed,
                split_name=confidence_split_name,
            )
        return x_out, y_out, variant_names

    if variant_mode_key in FASHION_ICON_VARIANT_SETS:
        variant_kinds = FASHION_ICON_VARIANT_SETS[variant_mode_key]
    else:
        variant_kinds = ["horizontal_stripes", "vertical_stripes", "center_shape", "plain"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(images.shape[0])
    chunks = np.array_split(perm, len(variant_kinds))
    out_images: list[np.ndarray] = []
    out_labels: list[np.ndarray] = []
    out_scores: list[np.ndarray] = []
    for variant_idx, (variant_kind, chunk) in enumerate(zip(variant_kinds, chunks)):
        if chunk.size == 0:
            continue
        fit_scores = _compute_fashion_variant_fit_scores(
            images[chunk],
            variant_kind=variant_kind,
            shape_kind=shape_kind,
            foreground_threshold=foreground_threshold,
            stripe_width=stripe_width,
            stripe_gap=stripe_gap,
            stripe_count=stripe_count,
            stripe_length_fraction=stripe_length_fraction,
            shape_size=shape_size,
        )
        transformed = _apply_fashion_variant_pattern(
            images[chunk],
            variant_kind=variant_kind,
            shape_kind=shape_kind,
            pattern_strength=pattern_strength,
            foreground_threshold=foreground_threshold,
            stripe_width=stripe_width,
            stripe_gap=stripe_gap,
            stripe_count=stripe_count,
            stripe_length_fraction=stripe_length_fraction,
            shape_size=shape_size,
            rng=np.random.default_rng(seed + 1009 * (variant_idx + 1)),
        )
        out_images.append(transformed)
        out_labels.append(np.full(chunk.shape[0], int(variant_idx), dtype=np.int64))
        out_scores.append(fit_scores.astype(np.float32))
    if not out_images:
        raise ValueError("Fashion variant split produced an empty dataset")
    x_out = np.concatenate(out_images, axis=0).astype(np.float32)
    y_out = np.concatenate(out_labels, axis=0).astype(np.int64)
    score_out = np.concatenate(out_scores, axis=0).astype(np.float32)
    filter_summary = None
    if confidence_keep_fraction < 1.0 or confidence_min_margin > 0.0:
        x_out, y_out, filter_summary = _filter_fashion_multiclass_score_split_by_confidence(
            x_out,
            y_out,
            score_out,
            class_names=variant_names,
            keep_fraction=confidence_keep_fraction,
            min_margin=confidence_min_margin,
            split_name=confidence_split_name,
        )
    if filter_summary is not None and bool(confidence_resample_to_before):
        x_out, y_out, _ = _resample_fashion_split_to_class_counts(
            x_out,
            y_out,
            class_names=variant_names,
            target_counts=filter_summary["class_counts_before"],
            seed=confidence_resample_seed,
            split_name=confidence_split_name,
        )
    return x_out, y_out, variant_names


def _compute_fashion_sandal_sole_height_scores(
    images_flat: np.ndarray,
    *,
    foreground_threshold: float,
    sole_width_fraction: float,
    bottom_band_fraction: float,
) -> np.ndarray:
    images = np.asarray(images_flat, dtype=np.float32).reshape(-1, 28, 28)
    scores = np.zeros(images.shape[0], dtype=np.float32)

    for idx, image in enumerate(images):
        fg = image > float(foreground_threshold)
        ys, xs = np.where(fg)
        if xs.size == 0:
            continue
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())
        silhouette = fg[y_min:y_max + 1, x_min:x_max + 1]
        if silhouette.size <= 0:
            continue
        h, _ = silhouette.shape
        row_widths = silhouette.sum(axis=1, dtype=np.float32)
        max_width = float(np.max(row_widths))
        if max_width <= 0.0:
            continue
        bottom_start = max(0, h - int(np.ceil(h * float(bottom_band_fraction))))
        bottom_widths = row_widths[bottom_start:]
        width_threshold = max(1.0, float(sole_width_fraction) * max_width)
        wide_rows = bottom_widths >= width_threshold
        consecutive_from_bottom = 0
        for is_wide in wide_rows[::-1]:
            if bool(is_wide):
                consecutive_from_bottom += 1
            elif consecutive_from_bottom > 0:
                break
        bottom_mass = float(np.sum(bottom_widths)) / max(max_width, 1e-6)
        bottom_wide_count = float(np.sum(wide_rows))
        bottom_edge_width = float(np.mean(bottom_widths[-min(3, bottom_widths.size):])) / max(max_width, 1e-6)
        scores[idx] = (
            float(consecutive_from_bottom)
            + 0.35 * bottom_wide_count
            + 0.12 * bottom_mass
            + 0.65 * bottom_edge_width
        )
    return scores.astype(np.float32)


def _filter_fashion_score_split_by_confidence(
    images: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    class_names: Sequence[str],
    keep_fraction: float,
    min_margin: float,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if not (0.0 < float(keep_fraction) <= 1.0):
        raise ValueError(f"{split_name}_confidence_keep_fraction must be in (0, 1]")
    if float(min_margin) < 0.0:
        raise ValueError(f"{split_name}_confidence_min_margin must be >= 0")

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    margins = np.abs(scores - float(threshold))
    selected_parts: list[np.ndarray] = []
    before_counts: Dict[str, int] = {}
    after_counts: Dict[str, int] = {}
    for class_idx, class_name in enumerate(class_names):
        class_indices = np.where(labels == int(class_idx))[0]
        before_counts[str(class_name)] = int(class_indices.size)
        if class_indices.size == 0:
            after_counts[str(class_name)] = 0
            continue
        ranked = class_indices[np.argsort(margins[class_indices], kind="stable")[::-1]]
        if float(min_margin) > 0.0:
            ranked = ranked[margins[ranked] >= float(min_margin)]
        target_count = int(np.ceil(class_indices.size * float(keep_fraction)))
        target_count = max(1, target_count)
        selected = ranked[: min(target_count, ranked.size)]
        if selected.size == 0:
            raise ValueError(
                f"Confidence filtering removed every sample for class {class_name!r}; "
                f"reduce {split_name}_confidence_min_margin or increase {split_name}_confidence_keep_fraction"
            )
        selected_parts.append(np.sort(selected).astype(np.int64))
        after_counts[str(class_name)] = int(selected.size)

    if not selected_parts:
        raise ValueError("Confidence filtering produced an empty Fashion-MNIST subclass split")
    selected_indices = np.concatenate(selected_parts, axis=0)
    selected_indices.sort()
    summary = {
        "enabled": bool(float(keep_fraction) < 1.0 or float(min_margin) > 0.0),
        "split_name": str(split_name),
        "keep_fraction": float(keep_fraction),
        "min_margin": float(min_margin),
        "threshold": float(threshold),
        "samples_before": int(labels.shape[0]),
        "samples_after": int(selected_indices.shape[0]),
        "class_counts_before": before_counts,
        "class_counts_after": after_counts,
        "margin_min_after": float(np.min(margins[selected_indices])),
        "margin_mean_after": float(np.mean(margins[selected_indices])),
    }
    return (
        np.asarray(images, dtype=np.float32)[selected_indices],
        labels[selected_indices].astype(np.int64),
        summary,
    )


def _filter_fashion_multiclass_score_split_by_confidence(
    images: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    class_names: Sequence[str],
    keep_fraction: float,
    min_margin: float,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if not (0.0 < float(keep_fraction) <= 1.0):
        raise ValueError(f"{split_name}_confidence_keep_fraction must be in (0, 1]")
    if float(min_margin) < 0.0:
        raise ValueError(f"{split_name}_confidence_min_margin must be >= 0")

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    selected_parts: list[np.ndarray] = []
    before_counts: Dict[str, int] = {}
    after_counts: Dict[str, int] = {}
    threshold_by_class: Dict[str, float] = {}
    for class_idx, class_name in enumerate(class_names):
        class_key = str(class_name)
        class_indices = np.where(labels == int(class_idx))[0]
        before_counts[class_key] = int(class_indices.size)
        if class_indices.size == 0:
            after_counts[class_key] = 0
            threshold_by_class[class_key] = 0.0
            continue
        ranked = class_indices[np.argsort(scores[class_indices], kind="stable")[::-1]]
        target_count = int(np.ceil(class_indices.size * float(keep_fraction)))
        target_count = max(1, min(target_count, ranked.size))
        threshold_value = float(scores[ranked[target_count - 1]])
        threshold_by_class[class_key] = threshold_value
        if float(min_margin) > 0.0:
            ranked = ranked[scores[ranked] >= threshold_value + float(min_margin)]
        selected = ranked[: min(target_count, ranked.size)]
        if selected.size == 0:
            raise ValueError(
                f"Confidence filtering removed every sample for class {class_name!r}; "
                f"reduce {split_name}_confidence_min_margin or increase {split_name}_confidence_keep_fraction"
            )
        selected_parts.append(np.sort(selected).astype(np.int64))
        after_counts[class_key] = int(selected.size)

    if not selected_parts:
        raise ValueError("Confidence filtering produced an empty Fashion-MNIST synthetic variant split")
    selected_indices = np.concatenate(selected_parts, axis=0)
    selected_indices.sort()
    summary = {
        "enabled": bool(float(keep_fraction) < 1.0 or float(min_margin) > 0.0),
        "split_name": str(split_name),
        "keep_fraction": float(keep_fraction),
        "min_margin": float(min_margin),
        "score_thresholds": threshold_by_class,
        "samples_before": int(labels.shape[0]),
        "samples_after": int(selected_indices.shape[0]),
        "class_counts_before": before_counts,
        "class_counts_after": after_counts,
        "score_min_after": float(np.min(scores[selected_indices])),
        "score_mean_after": float(np.mean(scores[selected_indices])),
        "score_max_after": float(np.max(scores[selected_indices])),
    }
    return (
        np.asarray(images, dtype=np.float32)[selected_indices],
        labels[selected_indices].astype(np.int64),
        summary,
    )


def _resample_fashion_split_to_class_counts(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    class_names: Sequence[str],
    target_counts: Mapping[str, int],
    seed: int,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    selected_parts: list[np.ndarray] = []
    before_counts: Dict[str, int] = {}
    after_counts: Dict[str, int] = {}
    resolved_targets: Dict[str, int] = {}
    for class_idx, class_name in enumerate(class_names):
        class_key = str(class_name)
        class_indices = np.where(labels == int(class_idx))[0]
        before_counts[class_key] = int(class_indices.size)
        target_count = int(target_counts.get(class_key, class_indices.size))
        if target_count < 0:
            raise ValueError(f"{split_name} resample target count for {class_key!r} must be non-negative")
        resolved_targets[class_key] = int(target_count)
        if target_count == 0:
            after_counts[class_key] = 0
            continue
        if class_indices.size == 0:
            raise ValueError(f"Cannot resample {split_name} class {class_key!r}; no confident samples remain")
        if target_count <= class_indices.size:
            selected = rng.choice(class_indices, size=target_count, replace=False)
        else:
            full_repeats = target_count // class_indices.size
            remainder = target_count % class_indices.size
            repeated = np.tile(class_indices, full_repeats)
            if remainder > 0:
                extra = rng.choice(class_indices, size=remainder, replace=False)
                selected = np.concatenate([repeated, extra], axis=0)
            else:
                selected = repeated
        selected_parts.append(np.asarray(selected, dtype=np.int64))
        after_counts[class_key] = int(target_count)

    if not selected_parts:
        raise ValueError(f"Resampling produced an empty {split_name} split")
    selected_indices = np.concatenate(selected_parts, axis=0)
    selected_indices = rng.permutation(selected_indices)
    summary = {
        "enabled": True,
        "split_name": str(split_name),
        "seed": int(seed),
        "samples_before_resample": int(labels.shape[0]),
        "samples_after_resample": int(selected_indices.shape[0]),
        "class_counts_before_resample": before_counts,
        "class_counts_target": resolved_targets,
        "class_counts_after_resample": after_counts,
    }
    return images[selected_indices], labels[selected_indices].astype(np.int64), summary


def _compute_fashion_variant_fit_scores(
    images_flat: np.ndarray,
    *,
    variant_kind: str,
    shape_kind: str,
    foreground_threshold: float,
    stripe_width: int,
    stripe_gap: int,
    stripe_count: int,
    stripe_length_fraction: float,
    shape_size: int,
) -> np.ndarray:
    images = np.asarray(images_flat, dtype=np.float32).reshape(-1, 28, 28)
    foreground = images > float(foreground_threshold)
    variant_key = str(variant_kind).strip().lower()
    best_scores = np.zeros(images.shape[0], dtype=np.float32)

    def _update_best(mask: np.ndarray) -> None:
        area = float(np.sum(mask))
        if area <= 0.0:
            return
        intersections = np.logical_and(foreground, mask[np.newaxis, :, :]).sum(axis=(1, 2), dtype=np.float32)
        np.maximum(best_scores, intersections / area, out=best_scores)

    if variant_key in {"horizontal_stripes", "vertical_stripes", "center_shape", "shape"}:
        mask_kind = "center_shape" if variant_key in {"center_shape", "shape"} else variant_key
        _update_best(
            _make_fashion_pattern_mask(
                kind=mask_kind,
                shape_kind=shape_kind,
                stripe_width=stripe_width,
                stripe_gap=stripe_gap,
                stripe_count=stripe_count,
                stripe_length_fraction=stripe_length_fraction,
                shape_size=shape_size,
            )
        )
        return best_scores.astype(np.float32)

    if variant_key in FASHION_ICON_KINDS:
        style_range = range(6) if variant_key == "letters" else range(1)
        for offset_y in (-1, 0, 1):
            for offset_x in (-1, 0, 1):
                for style_index in style_range:
                    _update_best(
                        _make_fashion_icon_mask(
                            kind=variant_key,
                            image_shape=(28, 28),
                            offset_y=offset_y,
                            offset_x=offset_x,
                            style_index=style_index,
                        )
                    )
        return best_scores.astype(np.float32)

    if variant_key == "plain":
        return np.ones(images.shape[0], dtype=np.float32)
    raise ValueError(f"Unsupported fashion variant kind={variant_kind!r}")


def _split_fashion_sandal_sole_height_dataset(
    images_flat: np.ndarray,
    *,
    source_class_name: str,
    foreground_threshold: float,
    sole_width_fraction: float,
    bottom_band_fraction: float,
    threshold: float | None = None,
    confidence_keep_fraction: float = 1.0,
    confidence_min_margin: float = 0.0,
    confidence_split_name: str = "split",
    confidence_resample_to_before: bool = False,
    confidence_resample_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str], Dict[str, Any]]:
    images = np.asarray(images_flat, dtype=np.float32)
    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError(f"Expected Fashion-MNIST flat images [N, 784], got {images.shape}")
    scores = _compute_fashion_sandal_sole_height_scores(
        images,
        foreground_threshold=foreground_threshold,
        sole_width_fraction=sole_width_fraction,
        bottom_band_fraction=bottom_band_fraction,
    )
    if threshold is None:
        order = np.argsort(scores, kind="stable")
        labels = np.zeros(scores.shape[0], dtype=np.int64)
        split_point = int(scores.shape[0] // 2)
        labels[order[split_point:]] = 1
        resolved_threshold = float(scores[order[split_point]]) if scores.shape[0] > 0 else 0.0
        threshold_mode = "rank_median"
    else:
        resolved_threshold = float(threshold)
        labels = np.where(scores >= resolved_threshold, 1, 0).astype(np.int64)
        threshold_mode = "fixed_threshold"
    class_names = [
        f"{source_class_name}_flat_sole",
        f"{source_class_name}_thick_sole",
    ]
    filter_summary: Dict[str, Any] | None = None
    if confidence_keep_fraction < 1.0 or confidence_min_margin > 0.0:
        images, labels, filter_summary = _filter_fashion_score_split_by_confidence(
            images,
            labels,
            scores,
            threshold=resolved_threshold,
            class_names=class_names,
            keep_fraction=confidence_keep_fraction,
            min_margin=confidence_min_margin,
            split_name=confidence_split_name,
        )
    resample_summary: Dict[str, Any] | None = None
    if filter_summary is not None and bool(confidence_resample_to_before):
        images, labels, resample_summary = _resample_fashion_split_to_class_counts(
            images,
            labels,
            class_names=class_names,
            target_counts=filter_summary["class_counts_before"],
            seed=confidence_resample_seed,
            split_name=confidence_split_name,
        )
    summary = {
        "source_class_name": str(source_class_name),
        "foreground_threshold": float(foreground_threshold),
        "sole_width_fraction": float(sole_width_fraction),
        "bottom_band_fraction": float(bottom_band_fraction),
        "score_threshold": float(resolved_threshold),
        "threshold_mode": str(threshold_mode),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_median": float(np.median(scores)),
        "class_counts": {
            class_names[0]: int(np.sum(labels == 0)),
            class_names[1]: int(np.sum(labels == 1)),
        },
    }
    if filter_summary is not None:
        summary["confidence_filter"] = filter_summary
    if resample_summary is not None:
        summary["confidence_resample"] = resample_summary
    return images.astype(np.float32), labels, class_names, summary


def _compute_fashion_bag_top_handle_scores(
    images_flat: np.ndarray,
    *,
    foreground_threshold: float,
    body_row_width_fraction: float,
    top_band_fraction: float,
    center_window_fraction: float,
) -> np.ndarray:
    images = np.asarray(images_flat, dtype=np.float32).reshape(-1, 28, 28)
    scores = np.zeros(images.shape[0], dtype=np.float32)
    for idx, image in enumerate(images):
        fg = image > float(foreground_threshold)
        ys, xs = np.where(fg)
        if xs.size == 0:
            continue
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())
        silhouette = fg[y_min:y_max + 1, x_min:x_max + 1]
        if silhouette.size <= 0:
            continue
        h, w = silhouette.shape
        row_widths = silhouette.sum(axis=1, dtype=np.float32)
        max_width = float(np.max(row_widths))
        if max_width <= 0.0:
            continue
        body_threshold = max(2.0, float(body_row_width_fraction) * max_width)
        body_rows = np.where(row_widths >= body_threshold)[0]
        body_start = int(body_rows[0]) if body_rows.size else int(np.argmax(row_widths > 0))
        top_end = max(1, min(h, int(np.ceil(h * float(top_band_fraction)))))
        appendage_end = min(body_start, top_end)
        if appendage_end <= 0:
            continue
        upper = silhouette[:appendage_end, :]
        upper_widths = upper.sum(axis=1, dtype=np.float32)
        upper_mass = float(np.sum(upper_widths))
        if upper_mass <= 0.0:
            continue
        narrow_bonus = float(
            np.sum(
                upper_widths
                * np.clip((body_threshold - upper_widths) / max(body_threshold, 1e-6), 0.0, 1.0)
            )
        )
        center_half = max(1, int(np.ceil(w * float(center_window_fraction) * 0.5)))
        center_x = w // 2
        center_x0 = max(0, center_x - center_half)
        center_x1 = min(w, center_x + center_half + 1)
        center_mass = float(np.sum(upper[:, center_x0:center_x1]))
        center_ratio = center_mass / max(upper_mass, 1e-6)
        vertical_margin = float(body_start) / max(float(h), 1.0)
        scores[idx] = (
            upper_mass
            + 1.35 * narrow_bonus
            + 0.45 * center_ratio * upper_mass
            + 2.0 * vertical_margin
        )
    return scores.astype(np.float32)


def _split_fashion_bag_top_handle_dataset(
    images_flat: np.ndarray,
    *,
    source_class_name: str,
    foreground_threshold: float,
    body_row_width_fraction: float,
    top_band_fraction: float,
    center_window_fraction: float,
    threshold: float | None = None,
    confidence_keep_fraction: float = 1.0,
    confidence_min_margin: float = 0.0,
    confidence_split_name: str = "split",
    confidence_resample_to_before: bool = False,
    confidence_resample_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str], Dict[str, Any]]:
    images = np.asarray(images_flat, dtype=np.float32)
    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError(f"Expected Fashion-MNIST flat images [N, 784], got {images.shape}")
    scores = _compute_fashion_bag_top_handle_scores(
        images,
        foreground_threshold=foreground_threshold,
        body_row_width_fraction=body_row_width_fraction,
        top_band_fraction=top_band_fraction,
        center_window_fraction=center_window_fraction,
    )
    if threshold is None:
        order = np.argsort(scores, kind="stable")
        labels = np.zeros(scores.shape[0], dtype=np.int64)
        split_point = int(scores.shape[0] // 2)
        labels[order[split_point:]] = 1
        resolved_threshold = float(scores[order[split_point]]) if scores.shape[0] > 0 else 0.0
        threshold_mode = "rank_median"
    else:
        resolved_threshold = float(threshold)
        labels = np.where(scores >= resolved_threshold, 1, 0).astype(np.int64)
        threshold_mode = "fixed_threshold"
    class_names = [
        f"{source_class_name}_without_obvious_top_handle",
        f"{source_class_name}_with_top_handle",
    ]
    filter_summary: Dict[str, Any] | None = None
    if confidence_keep_fraction < 1.0 or confidence_min_margin > 0.0:
        images, labels, filter_summary = _filter_fashion_score_split_by_confidence(
            images,
            labels,
            scores,
            threshold=resolved_threshold,
            class_names=class_names,
            keep_fraction=confidence_keep_fraction,
            min_margin=confidence_min_margin,
            split_name=confidence_split_name,
        )
    resample_summary: Dict[str, Any] | None = None
    if filter_summary is not None and bool(confidence_resample_to_before):
        images, labels, resample_summary = _resample_fashion_split_to_class_counts(
            images,
            labels,
            class_names=class_names,
            target_counts=filter_summary["class_counts_before"],
            seed=confidence_resample_seed,
            split_name=confidence_split_name,
        )
    summary = {
        "source_class_name": str(source_class_name),
        "foreground_threshold": float(foreground_threshold),
        "body_row_width_fraction": float(body_row_width_fraction),
        "top_band_fraction": float(top_band_fraction),
        "center_window_fraction": float(center_window_fraction),
        "score_threshold": float(resolved_threshold),
        "threshold_mode": str(threshold_mode),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_median": float(np.median(scores)),
        "class_counts": {
            class_names[0]: int(np.sum(labels == 0)),
            class_names[1]: int(np.sum(labels == 1)),
        },
    }
    if filter_summary is not None:
        summary["confidence_filter"] = filter_summary
    if resample_summary is not None:
        summary["confidence_resample"] = resample_summary
    return images.astype(np.float32), labels, class_names, summary


def _canonical_dataset_label_name(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _resolve_named_label_indices(
    requested: Sequence[Any],
    *,
    names: Sequence[str],
    label_kind: str,
) -> list[int]:
    if not isinstance(requested, Sequence) or isinstance(requested, (str, bytes)):
        raise TypeError(f"{label_kind} selection must be a sequence")
    name_to_idx = {
        _canonical_dataset_label_name(name): idx
        for idx, name in enumerate(names)
    }
    resolved: list[int] = []
    for value in requested:
        if isinstance(value, (int, np.integer)):
            idx = int(value)
        else:
            key = _canonical_dataset_label_name(value)
            if key not in name_to_idx:
                raise ValueError(
                    f"Unknown {label_kind} {value!r}; available values include {list(names)[:8]}"
                )
            idx = int(name_to_idx[key])
        if idx < 0 or idx >= len(names):
            raise ValueError(f"{label_kind} index {idx} is out of range [0, {len(names)})")
        resolved.append(idx)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{label_kind} selection contains duplicate entries: {requested!r}")
    return resolved


def _pad_class_names(active_names: Sequence[str], *, output_dim: int) -> list[str]:
    names = [str(name) for name in active_names]
    if len(names) > output_dim:
        raise ValueError(
            f"Active class count {len(names)} exceeds model output_dim={output_dim}"
        )
    if len(names) < output_dim:
        names.extend(f"unused_port_{idx}" for idx in range(len(names), output_dim))
    return names


def _one_hot_with_output_dim(
    class_indices: np.ndarray,
    *,
    output_dim: int,
) -> np.ndarray:
    indices = np.asarray(class_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("Cannot build one-hot labels for an empty dataset")
    if np.min(indices) < 0 or np.max(indices) >= output_dim:
        raise ValueError(
            f"Class indices must stay within [0, {output_dim}), got "
            f"min={int(np.min(indices))}, max={int(np.max(indices))}"
        )
    out = np.zeros((indices.shape[0], output_dim), dtype=np.float32)
    out[np.arange(indices.shape[0]), indices] = 1.0
    return out


def _subsample_split_indices(
    target_indices: np.ndarray,
    *,
    limit: int | None,
    seed: int,
) -> np.ndarray:
    targets = np.asarray(target_indices, dtype=np.int64).reshape(-1)
    if limit is None or limit <= 0 or limit >= targets.shape[0]:
        return np.arange(targets.shape[0], dtype=np.int64)
    chosen, _ = train_test_split(
        np.arange(targets.shape[0], dtype=np.int64),
        train_size=int(limit),
        stratify=targets,
        random_state=seed,
    )
    return np.asarray(chosen, dtype=np.int64)


def _load_cifar100_raw_arrays(data_home: Path) -> Dict[str, Any]:
    data_home.mkdir(parents=True, exist_ok=True)
    train_ds = CIFAR100(root=str(data_home), train=True, download=True)
    CIFAR100(root=str(data_home), train=False, download=True)
    base_dir = data_home / str(train_ds.base_folder)
    with (base_dir / "meta").open("rb") as handle:
        meta = pickle.load(handle, encoding="latin1")
    with (base_dir / "train").open("rb") as handle:
        train_payload = pickle.load(handle, encoding="latin1")
    with (base_dir / "test").open("rb") as handle:
        test_payload = pickle.load(handle, encoding="latin1")

    def _decode(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        images = np.asarray(payload["data"], dtype=np.float32).reshape(-1, 3, 32, 32)
        images = np.transpose(images, (0, 2, 3, 1)) / 255.0
        fine = np.asarray(payload["fine_labels"], dtype=np.int64)
        coarse = np.asarray(payload["coarse_labels"], dtype=np.int64)
        return images, fine, coarse

    x_train, fine_train, coarse_train = _decode(train_payload)
    x_eval, fine_eval, coarse_eval = _decode(test_payload)
    return {
        "x_train": x_train,
        "fine_train": fine_train,
        "coarse_train": coarse_train,
        "x_eval": x_eval,
        "fine_eval": fine_eval,
        "coarse_eval": coarse_eval,
        "fine_names": [str(name) for name in meta["fine_label_names"]],
        "coarse_names": [str(name) for name in meta["coarse_label_names"]],
    }


def _rgb_to_grayscale(images: np.ndarray) -> np.ndarray:
    rgb = np.asarray(images, dtype=np.float32)
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected CIFAR images [N, H, W, 3], got shape {rgb.shape}")
    weights = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    return np.tensordot(rgb, weights, axes=([-1], [0])).astype(np.float32)


def _make_cifar_avgpool_features(
    inputs: np.ndarray,
    *,
    grid_size: int,
) -> tuple[np.ndarray, list[int]]:
    gray = _rgb_to_grayscale(inputs)
    if gray.shape[1] % grid_size != 0 or gray.shape[2] % grid_size != 0:
        raise ValueError(
            f"CIFAR image shape {(gray.shape[1], gray.shape[2])} is not divisible by grid_size={grid_size}"
        )
    block_h = gray.shape[1] // grid_size
    block_w = gray.shape[2] // grid_size
    pooled = gray.reshape(-1, grid_size, block_h, grid_size, block_w).mean(axis=(2, 4))
    return pooled.reshape(gray.shape[0], grid_size * grid_size).astype(np.float32), [block_h, block_w]


def _make_cifar_rgb_block_mean(
    inputs: np.ndarray,
    *,
    grid_size: int,
) -> tuple[np.ndarray, list[int]]:
    rgb = np.asarray(inputs, dtype=np.float32)
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected CIFAR images [N, H, W, 3], got shape {rgb.shape}")
    if rgb.shape[1] % grid_size != 0 or rgb.shape[2] % grid_size != 0:
        raise ValueError(
            f"CIFAR image shape {(rgb.shape[1], rgb.shape[2])} is not divisible by grid_size={grid_size}"
        )
    block_h = rgb.shape[1] // grid_size
    block_w = rgb.shape[2] // grid_size
    pooled = rgb.reshape(-1, grid_size, block_h, grid_size, block_w, 3).mean(axis=(2, 4))
    return pooled.astype(np.float32), [block_h, block_w]


def _make_cifar_rgb_residual_flat_features(
    inputs: np.ndarray,
    *,
    grid_size: int,
) -> tuple[np.ndarray, list[int]]:
    rgb = np.asarray(inputs, dtype=np.float32)
    pooled_rgb, block_shape = _make_cifar_rgb_block_mean(rgb, grid_size=grid_size)
    block_h, block_w = block_shape
    upsampled = np.repeat(np.repeat(pooled_rgb, repeats=block_h, axis=1), repeats=block_w, axis=2)
    residual = rgb - upsampled
    return residual.reshape(rgb.shape[0], -1).astype(np.float32), block_shape


def _make_cifar_rgb_lowfreq_flat_features(
    inputs: np.ndarray,
    *,
    grid_size: int,
) -> tuple[np.ndarray, list[int]]:
    pooled_rgb, block_shape = _make_cifar_rgb_block_mean(inputs, grid_size=grid_size)
    return pooled_rgb.reshape(pooled_rgb.shape[0], -1).astype(np.float32), block_shape


def _make_cifar_branch_b_features(
    x_train_raw: np.ndarray,
    x_eval_raw: np.ndarray,
    *,
    dataset_cfg: Mapping[str, Any],
    num_branch_inputs: int,
    pca_components: int,
    avgpool_grid_size: int,
    input_low: float,
    input_high: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    feature_kind_raw = str(
        dataset_cfg.get("branch_b_feature_kind", f"avgpool{avgpool_grid_size}x{avgpool_grid_size}_grayscale")
    ).strip()
    feature_kind = feature_kind_raw.lower()

    if feature_kind in {
        "avgpool",
        "avgpool16",
        f"avgpool{avgpool_grid_size}x{avgpool_grid_size}",
        f"avgpool{avgpool_grid_size}x{avgpool_grid_size}_grayscale",
    }:
        x_train_values, pool_block_shape = _make_cifar_avgpool_features(
            x_train_raw,
            grid_size=avgpool_grid_size,
        )
        x_eval_values, eval_pool_block_shape = _make_cifar_avgpool_features(
            x_eval_raw,
            grid_size=avgpool_grid_size,
        )
        if eval_pool_block_shape != pool_block_shape:
            raise ValueError(
                "Avg-pool block shape mismatch between CIFAR train and eval: "
                f"train={pool_block_shape}, eval={eval_pool_block_shape}"
            )
        x_train_b, x_eval_b = _scale_feature_branch(
            x_train_values,
            x_eval_values,
            input_low=input_low,
            input_high=input_high,
        )
        metadata = {
            "feature_kind": f"avgpool{avgpool_grid_size}x{avgpool_grid_size}_grayscale",
            "pool_grid_shape": [avgpool_grid_size, avgpool_grid_size],
            "pool_block_shape": pool_block_shape,
        }
        return x_train_b, x_eval_b, metadata

    if feature_kind == "pca16_rgb_flat":
        x_train_source = x_train_raw.reshape(x_train_raw.shape[0], -1).astype(np.float32)
        x_eval_source = x_eval_raw.reshape(x_eval_raw.shape[0], -1).astype(np.float32)
        pca = PCA(
            n_components=pca_components,
            svd_solver=str(dataset_cfg.get("pca_solver", "randomized")),
            whiten=bool(dataset_cfg.get("pca_whiten", False)),
            random_state=seed,
        )
        x_train_pca = pca.fit_transform(x_train_source).astype(np.float32)
        x_eval_pca = pca.transform(x_eval_source).astype(np.float32)
        x_train_b, x_eval_b = _scale_feature_branch(
            x_train_pca,
            x_eval_pca,
            input_low=input_low,
            input_high=input_high,
        )
        explained = float(np.sum(np.asarray(pca.explained_variance_ratio_, dtype=np.float64)))
        metadata = {
            "feature_kind": "pca16_rgb_flat",
            "explained_variance_ratio_sum": explained,
        }
        return x_train_b, x_eval_b, metadata

    if feature_kind in {"pca16_rgb_residual_flat", "pca16_rgb_highpass_flat"}:
        x_train_source, residual_block_shape = _make_cifar_rgb_residual_flat_features(
            x_train_raw,
            grid_size=avgpool_grid_size,
        )
        x_eval_source, eval_residual_block_shape = _make_cifar_rgb_residual_flat_features(
            x_eval_raw,
            grid_size=avgpool_grid_size,
        )
        if eval_residual_block_shape != residual_block_shape:
            raise ValueError(
                "Residual block shape mismatch between CIFAR train and eval: "
                f"train={residual_block_shape}, eval={eval_residual_block_shape}"
            )
        pca = PCA(
            n_components=pca_components,
            svd_solver=str(dataset_cfg.get("pca_solver", "randomized")),
            whiten=bool(dataset_cfg.get("pca_whiten", False)),
            random_state=seed,
        )
        x_train_pca = pca.fit_transform(x_train_source).astype(np.float32)
        x_eval_pca = pca.transform(x_eval_source).astype(np.float32)
        x_train_b, x_eval_b = _scale_feature_branch(
            x_train_pca,
            x_eval_pca,
            input_low=input_low,
            input_high=input_high,
        )
        explained = float(np.sum(np.asarray(pca.explained_variance_ratio_, dtype=np.float64)))
        metadata = {
            "feature_kind": "pca16_rgb_residual_flat",
            "residual_grid_shape": [avgpool_grid_size, avgpool_grid_size],
            "residual_block_shape": residual_block_shape,
            "explained_variance_ratio_sum": explained,
        }
        return x_train_b, x_eval_b, metadata

    if feature_kind in {"pca16_rgb_lowfreq_flat", "pca16_rgb_blockmean_flat", "pca16_rgb_colorlowfreq_flat"}:
        x_train_source, lowfreq_block_shape = _make_cifar_rgb_lowfreq_flat_features(
            x_train_raw,
            grid_size=avgpool_grid_size,
        )
        x_eval_source, eval_lowfreq_block_shape = _make_cifar_rgb_lowfreq_flat_features(
            x_eval_raw,
            grid_size=avgpool_grid_size,
        )
        if eval_lowfreq_block_shape != lowfreq_block_shape:
            raise ValueError(
                "Low-frequency RGB block shape mismatch between CIFAR train and eval: "
                f"train={lowfreq_block_shape}, eval={eval_lowfreq_block_shape}"
            )
        pca = PCA(
            n_components=pca_components,
            svd_solver=str(dataset_cfg.get("pca_solver", "randomized")),
            whiten=bool(dataset_cfg.get("pca_whiten", False)),
            random_state=seed,
        )
        x_train_pca = pca.fit_transform(x_train_source).astype(np.float32)
        x_eval_pca = pca.transform(x_eval_source).astype(np.float32)
        x_train_b, x_eval_b = _scale_feature_branch(
            x_train_pca,
            x_eval_pca,
            input_low=input_low,
            input_high=input_high,
        )
        explained = float(np.sum(np.asarray(pca.explained_variance_ratio_, dtype=np.float64)))
        metadata = {
            "feature_kind": "pca16_rgb_lowfreq_flat",
            "lowfreq_grid_shape": [avgpool_grid_size, avgpool_grid_size],
            "lowfreq_block_shape": lowfreq_block_shape,
            "explained_variance_ratio_sum": explained,
        }
        return x_train_b, x_eval_b, metadata

    raise ValueError(
        "Unsupported train.dataset.branch_b_feature_kind="
        f"{feature_kind_raw!r}. Expected one of "
        f"{{'avgpool{avgpool_grid_size}x{avgpool_grid_size}_grayscale', 'pca16_rgb_flat', "
        "'pca16_rgb_residual_flat', 'pca16_rgb_lowfreq_flat'}}"
    )


def _resolve_cifar100_hierarchy(
    *,
    dataset_cfg: Mapping[str, Any],
    coarse_names: Sequence[str],
    fine_names: Sequence[str],
    fine_candidates_by_coarse: Mapping[int, Sequence[int]],
) -> Dict[str, Any]:
    requested_coarse = dataset_cfg.get("coarse_classes")
    requested_fine_by_coarse = dataset_cfg.get("fine_classes_by_coarse")
    fine_per_coarse = int(dataset_cfg.get("fine_per_coarse", 3))
    if fine_per_coarse <= 0:
        raise ValueError("train.dataset.fine_per_coarse must be positive")

    if requested_coarse is None and requested_fine_by_coarse is None:
        requested_coarse = list(CIFAR100_DEFAULT_HIERARCHY.keys())
        requested_fine_by_coarse = dict(CIFAR100_DEFAULT_HIERARCHY)
    elif requested_coarse is None:
        requested_coarse = list(dict(requested_fine_by_coarse).keys())

    coarse_indices = _resolve_named_label_indices(
        requested_coarse,
        names=coarse_names,
        label_kind="CIFAR-100 coarse class",
    )
    if len(coarse_indices) < 1 or len(coarse_indices) > 5:
        raise ValueError(
            "cifar100_hierarchical_subset expects 1-5 selected coarse classes, "
            f"got {len(coarse_indices)}"
        )

    fine_indices_by_coarse: Dict[int, list[int]] = {}
    if requested_fine_by_coarse is not None:
        raw_mapping = dict(requested_fine_by_coarse)
        coarse_name_to_idx = {
            _canonical_dataset_label_name(name): idx
            for idx, name in enumerate(coarse_names)
        }
        for coarse_key, requested_fine in raw_mapping.items():
            coarse_key_norm = _canonical_dataset_label_name(coarse_key)
            if coarse_key_norm not in coarse_name_to_idx:
                raise ValueError(
                    f"Unknown CIFAR-100 coarse class {coarse_key!r} in fine_classes_by_coarse"
                )
            coarse_idx = int(coarse_name_to_idx[coarse_key_norm])
            if coarse_idx not in coarse_indices:
                continue
            fine_idx = _resolve_named_label_indices(
                requested_fine,
                names=fine_names,
                label_kind=f"CIFAR-100 fine class under {coarse_names[coarse_idx]}",
            )
            allowed = set(int(value) for value in fine_candidates_by_coarse[coarse_idx])
            invalid = [idx for idx in fine_idx if idx not in allowed]
            if invalid:
                raise ValueError(
                    f"fine_classes_by_coarse[{coarse_names[coarse_idx]!r}] includes labels outside that coarse group: {invalid}"
                )
            fine_indices_by_coarse[coarse_idx] = fine_idx

    for coarse_idx in coarse_indices:
        if coarse_idx in fine_indices_by_coarse:
            picked = fine_indices_by_coarse[coarse_idx]
        else:
            picked = list(fine_candidates_by_coarse[coarse_idx])[:fine_per_coarse]
        if len(picked) != fine_per_coarse:
            raise ValueError(
                f"Each selected coarse class must contribute exactly {fine_per_coarse} fine classes; "
                f"{coarse_names[coarse_idx]!r} has {len(picked)}"
            )
        fine_indices_by_coarse[coarse_idx] = [int(idx) for idx in picked]

    coarse_display_names = [str(coarse_names[idx]) for idx in coarse_indices]
    fine_display_names = {
        str(coarse_names[coarse_idx]): [str(fine_names[idx]) for idx in fine_indices_by_coarse[coarse_idx]]
        for coarse_idx in coarse_indices
    }
    fine_display_flat = [
        f"{coarse_names[coarse_idx]}/{fine_names[fine_idx]}"
        for coarse_idx in coarse_indices
        for fine_idx in fine_indices_by_coarse[coarse_idx]
    ]
    return {
        "coarse_indices": coarse_indices,
        "coarse_names": coarse_display_names,
        "fine_indices_by_coarse": fine_indices_by_coarse,
        "fine_names_by_coarse": fine_display_names,
        "fine_display_flat": fine_display_flat,
        "fine_per_coarse": fine_per_coarse,
    }


def _make_mnist_pca_split_dataset(
    *,
    dataset_cfg: Mapping[str, Any],
    num_branch_inputs: int,
    num_outputs: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    Dict[str, Any],
]:
    if num_branch_inputs != 16:
        raise ValueError(
            f"mnist_pca_split16 requires model input_dim=16 per wavelength, got {num_branch_inputs}"
        )
    if num_outputs != 10:
        raise ValueError(f"mnist_pca_split16 requires model output_dim=10, got {num_outputs}")

    train_limit = int(dataset_cfg.get("train_limit", 0))
    eval_limit = int(dataset_cfg.get("eval_limit", 0))
    if train_limit <= 0 or eval_limit <= 0:
        raise ValueError("mnist_pca_split16 requires positive train_limit and eval_limit")

    pca_components = int(dataset_cfg.get("pca_components", 32))
    if pca_components < num_branch_inputs:
        raise ValueError(
            "train.dataset.pca_components must be >= branch input_dim, "
            f"got pca_components={pca_components}, branch_input_dim={num_branch_inputs}"
        )

    input_low = float(dataset_cfg.get("input_low", 0.0))
    input_high = float(dataset_cfg.get("input_high", 2.0))
    if input_high <= input_low:
        raise ValueError("train.dataset.input_high must be greater than input_low")

    data_home = get_project_root() / str(dataset_cfg.get("data_home", "data/cache/openml"))
    x_all, y_all = _fetch_mnist_arrays(data_home)
    if train_limit + eval_limit > int(x_all.shape[0]):
        raise ValueError("train_limit + eval_limit exceeds MNIST dataset size")

    x_train_raw, x_eval_raw, y_train_idx, y_eval_idx = train_test_split(
        x_all,
        y_all,
        train_size=train_limit,
        test_size=eval_limit,
        stratify=y_all,
        random_state=seed,
    )

    pca = PCA(
        n_components=pca_components,
        svd_solver=str(dataset_cfg.get("pca_solver", "randomized")),
        whiten=bool(dataset_cfg.get("pca_whiten", False)),
        random_state=seed,
    )
    x_train_pca = pca.fit_transform(x_train_raw).astype(np.float32)
    x_eval_pca = pca.transform(x_eval_raw).astype(np.float32)

    train_min = x_train_pca.min(axis=0, keepdims=True)
    train_max = x_train_pca.max(axis=0, keepdims=True)
    scale = np.maximum(train_max - train_min, 1e-12)
    x_train = input_low + (x_train_pca - train_min) * (input_high - input_low) / scale
    x_eval = input_low + (x_eval_pca - train_min) * (input_high - input_low) / scale
    x_train = np.clip(x_train, input_low, input_high).astype(np.float32)
    x_eval = np.clip(x_eval, input_low, input_high).astype(np.float32)

    branch_a_indices_cfg = dataset_cfg.get("branch_component_indices_a")
    branch_b_indices_cfg = dataset_cfg.get("branch_component_indices_b")
    if (branch_a_indices_cfg is None) != (branch_b_indices_cfg is None):
        raise ValueError(
            "train.dataset.branch_component_indices_a and branch_component_indices_b "
            "must be provided together"
        )
    if branch_a_indices_cfg is not None:
        branch_a_indices = np.asarray(list(branch_a_indices_cfg), dtype=np.int64).reshape(-1)
        branch_b_indices = np.asarray(list(branch_b_indices_cfg), dtype=np.int64).reshape(-1)
        component_split = "explicit_indices"
    else:
        component_split = str(dataset_cfg.get("component_split", "contiguous")).strip().lower()
        if component_split == "contiguous":
            if pca_components != 2 * num_branch_inputs:
                raise ValueError(
                    "train.dataset.component_split='contiguous' requires "
                    f"pca_components == 2 * branch_input_dim, got {pca_components}"
                )
            branch_a_indices = np.arange(0, num_branch_inputs, dtype=np.int64)
            branch_b_indices = np.arange(num_branch_inputs, 2 * num_branch_inputs, dtype=np.int64)
        elif component_split in {"interleaved", "alternating"}:
            if pca_components != 2 * num_branch_inputs:
                raise ValueError(
                    "train.dataset.component_split='interleaved' requires "
                    f"pca_components == 2 * branch_input_dim, got {pca_components}"
                )
            branch_a_indices = np.arange(0, 2 * num_branch_inputs, 2, dtype=np.int64)
            branch_b_indices = np.arange(1, 2 * num_branch_inputs, 2, dtype=np.int64)
        else:
            raise ValueError(
                "train.dataset.component_split must be 'contiguous' or 'interleaved', "
                f"got {component_split!r}"
            )

    if branch_a_indices.size != num_branch_inputs or branch_b_indices.size != num_branch_inputs:
        raise ValueError(
            "Each wavelength branch must receive exactly branch_input_dim PCA components, "
            f"got branch_a={branch_a_indices.size}, branch_b={branch_b_indices.size}, "
            f"branch_input_dim={num_branch_inputs}"
        )
    if len(set(branch_a_indices.astype(int).tolist())) != branch_a_indices.size:
        raise ValueError("train.dataset.branch_component_indices_a contains duplicate indices")
    if len(set(branch_b_indices.astype(int).tolist())) != branch_b_indices.size:
        raise ValueError("train.dataset.branch_component_indices_b contains duplicate indices")
    if np.min(branch_a_indices) < 0 or np.min(branch_b_indices) < 0:
        raise ValueError("branch component indices must be non-negative")
    if np.max(branch_a_indices) >= pca_components or np.max(branch_b_indices) >= pca_components:
        raise ValueError(
            "branch component indices exceed available PCA dimension: "
            f"pca_components={pca_components}, "
            f"branch_a_max={int(np.max(branch_a_indices))}, branch_b_max={int(np.max(branch_b_indices))}"
        )

    x_train_a = x_train[:, branch_a_indices].astype(np.float32)
    x_train_b = x_train[:, branch_b_indices].astype(np.float32)
    x_eval_a = x_eval[:, branch_a_indices].astype(np.float32)
    x_eval_b = x_eval[:, branch_b_indices].astype(np.float32)

    y_train = np.eye(num_outputs, dtype=np.float32)[y_train_idx]
    y_eval = np.eye(num_outputs, dtype=np.float32)[y_eval_idx]
    explained_variance_ratio = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    explained = float(np.sum(explained_variance_ratio))
    shared_indices = sorted(set(branch_a_indices.astype(int).tolist()) & set(branch_b_indices.astype(int).tolist()))
    union_indices = sorted(set(branch_a_indices.astype(int).tolist()) | set(branch_b_indices.astype(int).tolist()))
    dataset_summary = {
        "pca_components": int(pca_components),
        "component_split": component_split,
        "branch_a_component_indices": branch_a_indices.astype(int).tolist(),
        "branch_b_component_indices": branch_b_indices.astype(int).tolist(),
        "shared_component_indices": shared_indices,
        "unique_component_count": len(union_indices),
        "shared_component_count": len(shared_indices),
        "branch_a_explained_variance_ratio_sum": float(np.sum(explained_variance_ratio[branch_a_indices])),
        "branch_b_explained_variance_ratio_sum": float(np.sum(explained_variance_ratio[branch_b_indices])),
        "shared_explained_variance_ratio_sum": float(np.sum(explained_variance_ratio[shared_indices])),
        "scaled_input_low": float(input_low),
        "scaled_input_high": float(input_high),
        "branch_a_train_min": float(np.min(x_train_a)),
        "branch_a_train_max": float(np.max(x_train_a)),
        "branch_b_train_min": float(np.min(x_train_b)),
        "branch_b_train_max": float(np.max(x_train_b)),
        "branch_a_train_mean": float(np.mean(x_train_a)),
        "branch_b_train_mean": float(np.mean(x_train_b)),
        "branch_a_train_std": float(np.std(x_train_a)),
        "branch_b_train_std": float(np.std(x_train_b)),
    }
    return x_train_a, x_train_b, x_eval_a, x_eval_b, y_train, y_eval, explained, dataset_summary


def _make_mnist_pca16_single_dataset(
    *,
    dataset_cfg: Mapping[str, Any],
    num_inputs: int,
    num_outputs: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    Dict[str, Any],
]:
    if num_inputs != 16:
        raise ValueError(f"mnist_pca16_single requires model input_dim=16, got {num_inputs}")
    if num_outputs != 10:
        raise ValueError(f"mnist_pca16_single requires model output_dim=10, got {num_outputs}")

    train_limit = int(dataset_cfg.get("train_limit", 0))
    eval_limit = int(dataset_cfg.get("eval_limit", 0))
    if train_limit <= 0 or eval_limit <= 0:
        raise ValueError("mnist_pca16_single requires positive train_limit and eval_limit")

    pca_components = int(dataset_cfg.get("pca_components", num_inputs))
    if pca_components != num_inputs:
        raise ValueError(
            f"mnist_pca16_single requires train.dataset.pca_components == model input_dim={num_inputs}, "
            f"got {pca_components}"
        )

    input_low = float(dataset_cfg.get("input_low", 0.0))
    input_high = float(dataset_cfg.get("input_high", 2.0))
    if input_high <= input_low:
        raise ValueError("train.dataset.input_high must be greater than input_low")

    data_home = get_project_root() / str(dataset_cfg.get("data_home", "data/cache/openml"))
    x_all, y_all = _fetch_mnist_arrays(data_home)
    if train_limit + eval_limit > int(x_all.shape[0]):
        raise ValueError("train_limit + eval_limit exceeds MNIST dataset size")

    x_train_raw, x_eval_raw, y_train_idx, y_eval_idx = train_test_split(
        x_all,
        y_all,
        train_size=train_limit,
        test_size=eval_limit,
        stratify=y_all,
        random_state=seed,
    )

    pca = PCA(
        n_components=pca_components,
        svd_solver=str(dataset_cfg.get("pca_solver", "randomized")),
        whiten=bool(dataset_cfg.get("pca_whiten", False)),
        random_state=seed,
    )
    x_train_pca = pca.fit_transform(x_train_raw).astype(np.float32)
    x_eval_pca = pca.transform(x_eval_raw).astype(np.float32)

    train_min = x_train_pca.min(axis=0, keepdims=True)
    train_max = x_train_pca.max(axis=0, keepdims=True)
    scale = np.maximum(train_max - train_min, 1e-12)
    x_train = input_low + (x_train_pca - train_min) * (input_high - input_low) / scale
    x_eval = input_low + (x_eval_pca - train_min) * (input_high - input_low) / scale
    x_train = np.clip(x_train, input_low, input_high).astype(np.float32)
    x_eval = np.clip(x_eval, input_low, input_high).astype(np.float32)

    y_train = np.eye(num_outputs, dtype=np.float32)[y_train_idx]
    y_eval = np.eye(num_outputs, dtype=np.float32)[y_eval_idx]
    explained_variance_ratio = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    explained = float(np.sum(explained_variance_ratio))
    dataset_summary = {
        "pca_components": int(pca_components),
        "component_split": "single_branch",
        "input_component_indices": list(range(int(pca_components))),
        "scaled_input_low": float(input_low),
        "scaled_input_high": float(input_high),
        "input_train_min": float(np.min(x_train)),
        "input_train_max": float(np.max(x_train)),
        "input_train_mean": float(np.mean(x_train)),
        "input_train_std": float(np.std(x_train)),
        "explained_variance_ratio_sum": explained,
    }
    return x_train, x_eval, y_train, y_eval, explained, dataset_summary


def _make_fashion_mnist_subset_single_dataset(
    *,
    dataset_cfg: Mapping[str, Any],
    num_inputs: int,
    num_outputs: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    Dict[str, Any],
]:
    if num_inputs != 16:
        raise ValueError(
            f"fashion_mnist_subset_single requires model input_dim=16, got {num_inputs}"
        )
    if num_outputs != 10:
        raise ValueError(
            f"fashion_mnist_subset_single requires model output_dim=10, got {num_outputs}"
        )

    requested_classes = dataset_cfg.get("classes") or dataset_cfg.get("class_names")
    if requested_classes is None:
        raise ValueError(
            "fashion_mnist_subset_single requires train.dataset.classes "
            "with the selected Fashion-MNIST labels"
        )
    selected_class_indices = _resolve_named_label_indices(
        requested_classes,
        names=FASHION_MNIST_CLASS_NAMES,
        label_kind="fashion_mnist class",
    )
    if len(selected_class_indices) <= 0:
        raise ValueError("fashion_mnist_subset_single requires at least one selected class")
    synthetic_variants_cfg = dataset_cfg.get("synthetic_variants")
    synthetic_variants_enabled = bool(
        isinstance(synthetic_variants_cfg, Mapping)
        and synthetic_variants_cfg.get("enabled", True)
    )
    semantic_split_cfg = dataset_cfg.get("semantic_split")
    semantic_split_enabled = bool(
        isinstance(semantic_split_cfg, Mapping)
        and semantic_split_cfg.get("enabled", True)
    )
    if synthetic_variants_enabled and semantic_split_enabled:
        raise ValueError(
            "fashion_mnist_subset_single cannot enable synthetic_variants and semantic_split at the same time"
        )
    if synthetic_variants_enabled and len(selected_class_indices) != 1:
        raise ValueError(
            "fashion_mnist_subset_single synthetic_variants mode requires exactly one base class"
        )
    if semantic_split_enabled and len(selected_class_indices) != 1:
        raise ValueError(
            "fashion_mnist_subset_single semantic_split mode requires exactly one base class"
        )
    synthetic_variant_mode = "four_way"
    if synthetic_variants_enabled:
        synthetic_variant_mode = str(synthetic_variants_cfg.get("variant_mode", "four_way")).strip().lower()
        if synthetic_variant_mode not in {
            "four_way",
            "legacy_four_way",
            "icon_four_way",
            "icon_smiley_apple_bear_letters",
            "icon_smiley_apple_bear_heart",
            "icon_star_heart_apple_smiley",
            "icon_triangle_slash_apple_smiley",
            "chest_pattern_then_stripes",
            "chest_pattern_stripes",
            "pattern_then_stripes",
        }:
            raise ValueError(
                "fashion_mnist_subset_single synthetic_variants.variant_mode must be 'four_way' or 'chest_pattern_then_stripes'"
            )
    active_class_count = (
        (3 if synthetic_variant_mode in {"chest_pattern_then_stripes", "chest_pattern_stripes", "pattern_then_stripes"} else 4)
        if synthetic_variants_enabled
        else 2 if semantic_split_enabled
        else len(selected_class_indices)
    )
    if active_class_count > num_outputs:
        raise ValueError(
            f"fashion_mnist_subset_single active class count {active_class_count} "
            f"exceeds model output_dim={num_outputs}"
        )

    pca_components = int(dataset_cfg.get("pca_components", num_inputs))
    if pca_components != num_inputs:
        raise ValueError(
            "fashion_mnist_subset_single requires "
            f"train.dataset.pca_components == model input_dim={num_inputs}, got {pca_components}"
        )

    input_low = float(dataset_cfg.get("input_low", 0.0))
    input_high = float(dataset_cfg.get("input_high", 2.0))
    if input_high <= input_low:
        raise ValueError("train.dataset.input_high must be greater than input_low")

    data_home = get_project_root() / str(dataset_cfg.get("data_home", "data/cache/fashion_mnist"))
    raw = _load_fashion_mnist_raw_arrays(data_home)
    selected_names = [str(FASHION_MNIST_CLASS_NAMES[idx]) for idx in selected_class_indices]
    class_remap = {
        int(source_idx): int(target_idx)
        for target_idx, source_idx in enumerate(selected_class_indices)
    }

    train_mask = np.isin(raw["y_train"], np.asarray(selected_class_indices, dtype=np.int64))
    eval_mask = np.isin(raw["y_eval"], np.asarray(selected_class_indices, dtype=np.int64))
    x_train_raw = raw["x_train"][train_mask]
    x_eval_raw = raw["x_eval"][eval_mask]
    y_train_src = raw["y_train"][train_mask]
    y_eval_src = raw["y_eval"][eval_mask]
    y_train_idx = np.asarray([class_remap[int(value)] for value in y_train_src], dtype=np.int64)
    y_eval_idx = np.asarray([class_remap[int(value)] for value in y_eval_src], dtype=np.int64)

    train_limit_raw = dataset_cfg.get("train_limit")
    eval_limit_raw = dataset_cfg.get("eval_limit")
    train_limit = None if train_limit_raw is None else int(train_limit_raw)
    eval_limit = None if eval_limit_raw is None else int(eval_limit_raw)
    if synthetic_variants_enabled or semantic_split_enabled:
        train_subset = _subsample_unstratified_indices(
            int(x_train_raw.shape[0]),
            limit=train_limit,
            seed=seed,
        )
        eval_subset = _subsample_unstratified_indices(
            int(x_eval_raw.shape[0]),
            limit=eval_limit,
            seed=seed + 1,
        )
    else:
        train_subset = _subsample_split_indices(y_train_idx, limit=train_limit, seed=seed)
        eval_subset = _subsample_split_indices(y_eval_idx, limit=eval_limit, seed=seed + 1)
    x_train_raw = x_train_raw[train_subset]
    x_eval_raw = x_eval_raw[eval_subset]
    y_train_idx = y_train_idx[train_subset]
    y_eval_idx = y_eval_idx[eval_subset]
    active_names = list(selected_names)

    if synthetic_variants_enabled:
        shape_kind = str(synthetic_variants_cfg.get("shape_kind", "square")).strip().lower()
        if shape_kind not in {"square", "circle"}:
            raise ValueError(
                "fashion_mnist_subset_single synthetic_variants.shape_kind must be 'square' or 'circle'"
            )
        pattern_strength = float(synthetic_variants_cfg.get("pattern_strength", 0.85))
        if pattern_strength <= 0.0 or pattern_strength > 1.0:
            raise ValueError(
                "fashion_mnist_subset_single synthetic_variants.pattern_strength must be in (0, 1]"
            )
        foreground_threshold = float(synthetic_variants_cfg.get("foreground_threshold", 0.05))
        stripe_width = int(synthetic_variants_cfg.get("stripe_width", 1))
        stripe_gap = int(synthetic_variants_cfg.get("stripe_gap", 4))
        stripe_count = int(synthetic_variants_cfg.get("stripe_count", 3))
        stripe_length_fraction = float(synthetic_variants_cfg.get("stripe_length_fraction", 0.5))
        shape_size = int(synthetic_variants_cfg.get("shape_size", 6))
        if stripe_width <= 0 or stripe_gap < 0 or stripe_count <= 0 or shape_size <= 0:
            raise ValueError(
                "fashion_mnist_subset_single synthetic_variants stripe_width/stripe_count/shape_size must be positive and stripe_gap must be >= 0"
            )
        if not (0.0 < stripe_length_fraction <= 1.0):
            raise ValueError(
                "fashion_mnist_subset_single synthetic_variants.stripe_length_fraction must be in (0, 1]"
            )
        chest_y0_fraction = float(synthetic_variants_cfg.get("chest_y0_fraction", 0.25))
        chest_y1_fraction = float(synthetic_variants_cfg.get("chest_y1_fraction", 0.68))
        chest_width_fraction = float(synthetic_variants_cfg.get("chest_width_fraction", 0.55))
        confidence_keep_fraction = float(synthetic_variants_cfg.get("confidence_keep_fraction", 1.0))
        confidence_min_margin = float(synthetic_variants_cfg.get("confidence_min_margin", 0.0))
        confidence_resample_to_before = bool(synthetic_variants_cfg.get("confidence_resample_to_before", False))
        train_confidence_keep_fraction = float(
            synthetic_variants_cfg.get("train_confidence_keep_fraction", confidence_keep_fraction)
        )
        train_confidence_min_margin = float(
            synthetic_variants_cfg.get("train_confidence_min_margin", confidence_min_margin)
        )
        train_confidence_resample_to_before = bool(
            synthetic_variants_cfg.get(
                "train_confidence_resample_to_before",
                confidence_resample_to_before,
            )
        )
        eval_confidence_keep_fraction = float(
            synthetic_variants_cfg.get("eval_confidence_keep_fraction", confidence_keep_fraction)
        )
        eval_confidence_min_margin = float(
            synthetic_variants_cfg.get("eval_confidence_min_margin", confidence_min_margin)
        )
        eval_confidence_resample_to_before = bool(
            synthetic_variants_cfg.get(
                "eval_confidence_resample_to_before",
                confidence_resample_to_before,
            )
        )
        if not (0.0 <= chest_y0_fraction < 1.0):
            raise ValueError(
                "fashion_mnist_subset_single synthetic_variants.chest_y0_fraction must be in [0, 1)"
            )
        if not (0.0 < chest_y1_fraction <= 1.0) or chest_y1_fraction <= chest_y0_fraction:
            raise ValueError(
                "fashion_mnist_subset_single synthetic_variants.chest_y1_fraction must be in (0, 1] and exceed chest_y0_fraction"
            )
        if not (0.0 < chest_width_fraction <= 1.0):
            raise ValueError(
                "fashion_mnist_subset_single synthetic_variants.chest_width_fraction must be in (0, 1]"
            )
        x_train_raw, y_train_idx, active_names = _split_fashion_variant_dataset(
            x_train_raw,
            source_class_name=selected_names[0],
            seed=seed,
            variant_mode=synthetic_variant_mode,
            shape_kind=shape_kind,
            pattern_strength=pattern_strength,
            foreground_threshold=foreground_threshold,
            stripe_width=stripe_width,
            stripe_gap=stripe_gap,
            stripe_count=stripe_count,
            stripe_length_fraction=stripe_length_fraction,
            shape_size=shape_size,
            chest_y0_fraction=chest_y0_fraction,
            chest_y1_fraction=chest_y1_fraction,
            chest_width_fraction=chest_width_fraction,
            confidence_keep_fraction=train_confidence_keep_fraction,
            confidence_min_margin=train_confidence_min_margin,
            confidence_split_name="train",
            confidence_resample_to_before=train_confidence_resample_to_before,
            confidence_resample_seed=seed,
        )
        x_eval_raw, y_eval_idx, _ = _split_fashion_variant_dataset(
            x_eval_raw,
            source_class_name=selected_names[0],
            seed=seed + 1,
            variant_mode=synthetic_variant_mode,
            shape_kind=shape_kind,
            pattern_strength=pattern_strength,
            foreground_threshold=foreground_threshold,
            stripe_width=stripe_width,
            stripe_gap=stripe_gap,
            stripe_count=stripe_count,
            stripe_length_fraction=stripe_length_fraction,
            shape_size=shape_size,
            chest_y0_fraction=chest_y0_fraction,
            chest_y1_fraction=chest_y1_fraction,
            chest_width_fraction=chest_width_fraction,
            confidence_keep_fraction=eval_confidence_keep_fraction,
            confidence_min_margin=eval_confidence_min_margin,
            confidence_split_name="eval",
            confidence_resample_to_before=eval_confidence_resample_to_before,
            confidence_resample_seed=seed + 1,
        )
    elif semantic_split_enabled:
        rule = str(semantic_split_cfg.get("rule", "sandal_sole_height")).strip().lower()
        if rule not in {"sandal_sole_height", "sole_height", "bag_top_handle", "top_handle"}:
            raise ValueError(
                "fashion_mnist_subset_single semantic_split.rule currently only supports "
                "'sandal_sole_height' or 'bag_top_handle'"
            )
        foreground_threshold = float(semantic_split_cfg.get("foreground_threshold", 0.05))
        if rule in {"sandal_sole_height", "sole_height"}:
            if _canonical_dataset_label_name(selected_names[0]) != _canonical_dataset_label_name("Sandal"):
                raise ValueError(
                    "fashion_mnist_subset_single semantic_split.sandal_sole_height currently only supports "
                    "the base class 'Sandal'"
                )
            sole_width_fraction = float(semantic_split_cfg.get("sole_width_fraction", 0.45))
            bottom_band_fraction = float(semantic_split_cfg.get("bottom_band_fraction", 0.35))
            if not (0.0 < sole_width_fraction <= 1.0):
                raise ValueError(
                    "fashion_mnist_subset_single semantic_split.sole_width_fraction must be in (0, 1]"
                )
            if not (0.0 < bottom_band_fraction <= 1.0):
                raise ValueError(
                    "fashion_mnist_subset_single semantic_split.bottom_band_fraction must be in (0, 1]"
                )
            x_train_raw, y_train_idx, active_names, semantic_summary = _split_fashion_sandal_sole_height_dataset(
                x_train_raw,
                source_class_name=selected_names[0],
                foreground_threshold=foreground_threshold,
                sole_width_fraction=sole_width_fraction,
                bottom_band_fraction=bottom_band_fraction,
                threshold=None,
            )
            x_eval_raw, y_eval_idx, _, eval_semantic_summary = _split_fashion_sandal_sole_height_dataset(
                x_eval_raw,
                source_class_name=selected_names[0],
                foreground_threshold=foreground_threshold,
                sole_width_fraction=sole_width_fraction,
                bottom_band_fraction=bottom_band_fraction,
                threshold=float(semantic_summary["score_threshold"]),
            )
        else:
            if _canonical_dataset_label_name(selected_names[0]) != _canonical_dataset_label_name("Bag"):
                raise ValueError(
                    "fashion_mnist_subset_single semantic_split.bag_top_handle currently only supports "
                    "the base class 'Bag'"
                )
            body_row_width_fraction = float(semantic_split_cfg.get("body_row_width_fraction", 0.62))
            top_band_fraction = float(semantic_split_cfg.get("top_band_fraction", 0.42))
            center_window_fraction = float(semantic_split_cfg.get("center_window_fraction", 0.45))
            if not (0.0 < body_row_width_fraction <= 1.0):
                raise ValueError(
                    "fashion_mnist_subset_single semantic_split.body_row_width_fraction must be in (0, 1]"
                )
            if not (0.0 < top_band_fraction <= 1.0):
                raise ValueError(
                    "fashion_mnist_subset_single semantic_split.top_band_fraction must be in (0, 1]"
                )
            if not (0.0 < center_window_fraction <= 1.0):
                raise ValueError(
                    "fashion_mnist_subset_single semantic_split.center_window_fraction must be in (0, 1]"
                )
            x_train_raw, y_train_idx, active_names, semantic_summary = _split_fashion_bag_top_handle_dataset(
                x_train_raw,
                source_class_name=selected_names[0],
                foreground_threshold=foreground_threshold,
                body_row_width_fraction=body_row_width_fraction,
                top_band_fraction=top_band_fraction,
                center_window_fraction=center_window_fraction,
                threshold=None,
            )
            x_eval_raw, y_eval_idx, _, eval_semantic_summary = _split_fashion_bag_top_handle_dataset(
                x_eval_raw,
                source_class_name=selected_names[0],
                foreground_threshold=foreground_threshold,
                body_row_width_fraction=body_row_width_fraction,
                top_band_fraction=top_band_fraction,
                center_window_fraction=center_window_fraction,
                threshold=float(semantic_summary["score_threshold"]),
            )

    pca = PCA(
        n_components=pca_components,
        svd_solver=str(dataset_cfg.get("pca_solver", "randomized")),
        whiten=bool(dataset_cfg.get("pca_whiten", False)),
        random_state=seed,
    )
    x_train_pca = pca.fit_transform(x_train_raw).astype(np.float32)
    x_eval_pca = pca.transform(x_eval_raw).astype(np.float32)
    x_train, x_eval = _scale_feature_branch(
        x_train_pca,
        x_eval_pca,
        input_low=input_low,
        input_high=input_high,
    )

    y_train = _one_hot_with_output_dim(y_train_idx, output_dim=num_outputs)
    y_eval = _one_hot_with_output_dim(y_eval_idx, output_dim=num_outputs)
    explained_variance_ratio = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    explained = float(np.sum(explained_variance_ratio))
    dataset_summary = {
        "dataset_name": "fashion_mnist_subset_single",
        "target_level": (
            "synthetic_variant"
            if synthetic_variants_enabled
            else "semantic_split"
            if semantic_split_enabled
            else "subset_class"
        ),
        "selected_class_indices": [int(idx) for idx in selected_class_indices],
        "selected_class_names": selected_names,
        "class_names_active": list(active_names),
        "class_names_padded": _pad_class_names(active_names, output_dim=num_outputs),
        "pca_components": int(pca_components),
        "component_split": "single_branch",
        "input_component_indices": list(range(int(pca_components))),
        "scaled_input_low": float(input_low),
        "scaled_input_high": float(input_high),
        "input_train_min": float(np.min(x_train)),
        "input_train_max": float(np.max(x_train)),
        "input_train_mean": float(np.mean(x_train)),
        "input_train_std": float(np.std(x_train)),
        "explained_variance_ratio_sum": explained,
        "train_samples": int(x_train.shape[0]),
        "eval_samples": int(x_eval.shape[0]),
        "train_limit_applied": None if train_limit is None else int(train_limit),
        "eval_limit_applied": None if eval_limit is None else int(eval_limit),
    }
    if synthetic_variants_enabled:
        dataset_summary["synthetic_variants"] = {
            "enabled": True,
            "source_class_name": selected_names[0],
            "variant_class_names": list(active_names),
            "variant_mode": synthetic_variant_mode,
            "shape_kind": shape_kind,
            "pattern_strength": float(pattern_strength),
            "foreground_threshold": float(foreground_threshold),
            "stripe_width": int(stripe_width),
            "stripe_gap": int(stripe_gap),
            "stripe_count": int(stripe_count),
            "stripe_length_fraction": float(stripe_length_fraction),
            "shape_size": int(shape_size),
            "chest_y0_fraction": float(chest_y0_fraction),
            "chest_y1_fraction": float(chest_y1_fraction),
            "chest_width_fraction": float(chest_width_fraction),
            "confidence_keep_fraction": float(confidence_keep_fraction),
            "confidence_min_margin": float(confidence_min_margin),
            "confidence_resample_to_before": bool(confidence_resample_to_before),
        }
    if semantic_split_enabled:
        dataset_summary["semantic_split"] = {
            "enabled": True,
            "rule": rule,
            "train": semantic_summary,
            "eval": eval_semantic_summary,
        }
    return x_train, x_eval, y_train, y_eval, explained, dataset_summary


def _apply_fashion_subclass_rule(
    *,
    images_train_raw: np.ndarray,
    images_eval_raw: np.ndarray,
    source_class_name: str,
    split_cfg: Mapping[str, Any],
    train_seed: int,
    eval_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], Dict[str, Any]]:
    rule = str(split_cfg.get("rule", "")).strip().lower()
    foreground_threshold = float(split_cfg.get("foreground_threshold", 0.05))
    if rule in {"synthetic_variants", "tshirt_variants", "variants"}:
        variant_mode = str(split_cfg.get("variant_mode", "four_way")).strip().lower()
        if variant_mode not in {
            "four_way",
            "legacy_four_way",
            "icon_four_way",
            "icon_smiley_apple_bear_letters",
            "icon_smiley_apple_bear_heart",
            "icon_star_heart_apple_smiley",
            "icon_triangle_slash_apple_smiley",
            "chest_pattern_then_stripes",
            "chest_pattern_stripes",
            "pattern_then_stripes",
        }:
            raise ValueError(
                "fashion_mnist_subclass_composite_single synthetic_variants.variant_mode must be 'four_way' or 'chest_pattern_then_stripes'"
            )
        shape_kind = str(split_cfg.get("shape_kind", "square")).strip().lower()
        if shape_kind not in {"square", "circle"}:
            raise ValueError(
                "fashion_mnist_subclass_composite_single synthetic_variants.shape_kind must be 'square' or 'circle'"
            )
        pattern_strength = float(split_cfg.get("pattern_strength", 0.85))
        if pattern_strength <= 0.0 or pattern_strength > 1.0:
            raise ValueError(
                "fashion_mnist_subclass_composite_single synthetic_variants.pattern_strength must be in (0, 1]"
            )
        stripe_width = int(split_cfg.get("stripe_width", 1))
        stripe_gap = int(split_cfg.get("stripe_gap", 4))
        stripe_count = int(split_cfg.get("stripe_count", 3))
        stripe_length_fraction = float(split_cfg.get("stripe_length_fraction", 0.5))
        shape_size = int(split_cfg.get("shape_size", 6))
        if stripe_width <= 0 or stripe_gap < 0 or stripe_count <= 0 or shape_size <= 0:
            raise ValueError(
                "fashion_mnist_subclass_composite_single synthetic_variants stripe_width/stripe_count/shape_size must be positive and stripe_gap must be >= 0"
            )
        if not (0.0 < stripe_length_fraction <= 1.0):
            raise ValueError(
                "fashion_mnist_subclass_composite_single synthetic_variants.stripe_length_fraction must be in (0, 1]"
            )
        chest_y0_fraction = float(split_cfg.get("chest_y0_fraction", 0.25))
        chest_y1_fraction = float(split_cfg.get("chest_y1_fraction", 0.68))
        chest_width_fraction = float(split_cfg.get("chest_width_fraction", 0.55))
        confidence_keep_fraction = float(split_cfg.get("confidence_keep_fraction", 1.0))
        confidence_min_margin = float(split_cfg.get("confidence_min_margin", 0.0))
        confidence_resample_to_before = bool(split_cfg.get("confidence_resample_to_before", False))
        train_confidence_keep_fraction = float(
            split_cfg.get("train_confidence_keep_fraction", confidence_keep_fraction)
        )
        train_confidence_min_margin = float(split_cfg.get("train_confidence_min_margin", confidence_min_margin))
        train_confidence_resample_to_before = bool(
            split_cfg.get("train_confidence_resample_to_before", confidence_resample_to_before)
        )
        eval_confidence_keep_fraction = float(
            split_cfg.get("eval_confidence_keep_fraction", confidence_keep_fraction)
        )
        eval_confidence_min_margin = float(split_cfg.get("eval_confidence_min_margin", confidence_min_margin))
        eval_confidence_resample_to_before = bool(
            split_cfg.get("eval_confidence_resample_to_before", confidence_resample_to_before)
        )
        if not (0.0 <= chest_y0_fraction < 1.0):
            raise ValueError(
                "fashion_mnist_subclass_composite_single synthetic_variants.chest_y0_fraction must be in [0, 1)"
            )
        if not (0.0 < chest_y1_fraction <= 1.0) or chest_y1_fraction <= chest_y0_fraction:
            raise ValueError(
                "fashion_mnist_subclass_composite_single synthetic_variants.chest_y1_fraction must be in (0, 1] and exceed chest_y0_fraction"
            )
        if not (0.0 < chest_width_fraction <= 1.0):
            raise ValueError(
                "fashion_mnist_subclass_composite_single synthetic_variants.chest_width_fraction must be in (0, 1]"
            )
        x_train_split, y_train_idx, active_names = _split_fashion_variant_dataset(
            images_train_raw,
            source_class_name=source_class_name,
            seed=train_seed,
            variant_mode=variant_mode,
            shape_kind=shape_kind,
            pattern_strength=pattern_strength,
            foreground_threshold=foreground_threshold,
            stripe_width=stripe_width,
            stripe_gap=stripe_gap,
            stripe_count=stripe_count,
            stripe_length_fraction=stripe_length_fraction,
            shape_size=shape_size,
            chest_y0_fraction=chest_y0_fraction,
            chest_y1_fraction=chest_y1_fraction,
            chest_width_fraction=chest_width_fraction,
            confidence_keep_fraction=train_confidence_keep_fraction,
            confidence_min_margin=train_confidence_min_margin,
            confidence_split_name="train",
            confidence_resample_to_before=train_confidence_resample_to_before,
            confidence_resample_seed=train_seed,
        )
        x_eval_split, y_eval_idx, _ = _split_fashion_variant_dataset(
            images_eval_raw,
            source_class_name=source_class_name,
            seed=eval_seed,
            variant_mode=variant_mode,
            shape_kind=shape_kind,
            pattern_strength=pattern_strength,
            foreground_threshold=foreground_threshold,
            stripe_width=stripe_width,
            stripe_gap=stripe_gap,
            stripe_count=stripe_count,
            stripe_length_fraction=stripe_length_fraction,
            shape_size=shape_size,
            chest_y0_fraction=chest_y0_fraction,
            chest_y1_fraction=chest_y1_fraction,
            chest_width_fraction=chest_width_fraction,
            confidence_keep_fraction=eval_confidence_keep_fraction,
            confidence_min_margin=eval_confidence_min_margin,
            confidence_split_name="eval",
            confidence_resample_to_before=eval_confidence_resample_to_before,
            confidence_resample_seed=eval_seed,
        )
        split_summary = {
            "rule": "synthetic_variants",
            "source_class_name": str(source_class_name),
            "class_names": list(active_names),
            "variant_mode": variant_mode,
            "shape_kind": shape_kind,
            "pattern_strength": float(pattern_strength),
            "foreground_threshold": float(foreground_threshold),
            "stripe_width": int(stripe_width),
            "stripe_gap": int(stripe_gap),
            "stripe_count": int(stripe_count),
            "stripe_length_fraction": float(stripe_length_fraction),
            "shape_size": int(shape_size),
            "chest_y0_fraction": float(chest_y0_fraction),
            "chest_y1_fraction": float(chest_y1_fraction),
            "chest_width_fraction": float(chest_width_fraction),
            "confidence_keep_fraction": float(confidence_keep_fraction),
            "confidence_min_margin": float(confidence_min_margin),
            "confidence_resample_to_before": bool(confidence_resample_to_before),
        }
        return x_train_split, y_train_idx, x_eval_split, y_eval_idx, active_names, split_summary

    if rule in {"sandal_sole_height", "sole_height"}:
        if _canonical_dataset_label_name(source_class_name) != _canonical_dataset_label_name("Sandal"):
            raise ValueError(
                "fashion_mnist_subclass_composite_single semantic_split.sandal_sole_height currently only supports "
                "the base class 'Sandal'"
            )
        sole_width_fraction = float(split_cfg.get("sole_width_fraction", 0.45))
        bottom_band_fraction = float(split_cfg.get("bottom_band_fraction", 0.35))
        confidence_keep_fraction = float(split_cfg.get("confidence_keep_fraction", 1.0))
        confidence_min_margin = float(split_cfg.get("confidence_min_margin", 0.0))
        confidence_resample_to_before = bool(split_cfg.get("confidence_resample_to_before", False))
        train_confidence_keep_fraction = float(
            split_cfg.get("train_confidence_keep_fraction", confidence_keep_fraction)
        )
        train_confidence_min_margin = float(split_cfg.get("train_confidence_min_margin", confidence_min_margin))
        train_confidence_resample_to_before = bool(
            split_cfg.get("train_confidence_resample_to_before", confidence_resample_to_before)
        )
        eval_confidence_keep_fraction = float(split_cfg.get("eval_confidence_keep_fraction", confidence_keep_fraction))
        eval_confidence_min_margin = float(split_cfg.get("eval_confidence_min_margin", confidence_min_margin))
        eval_confidence_resample_to_before = bool(
            split_cfg.get("eval_confidence_resample_to_before", confidence_resample_to_before)
        )
        if not (0.0 < sole_width_fraction <= 1.0):
            raise ValueError(
                "fashion_mnist_subclass_composite_single semantic_split.sole_width_fraction must be in (0, 1]"
            )
        if not (0.0 < bottom_band_fraction <= 1.0):
            raise ValueError(
                "fashion_mnist_subclass_composite_single semantic_split.bottom_band_fraction must be in (0, 1]"
            )
        x_train_split, y_train_idx, active_names, train_summary = _split_fashion_sandal_sole_height_dataset(
            images_train_raw,
            source_class_name=source_class_name,
            foreground_threshold=foreground_threshold,
            sole_width_fraction=sole_width_fraction,
            bottom_band_fraction=bottom_band_fraction,
            threshold=None,
            confidence_keep_fraction=train_confidence_keep_fraction,
            confidence_min_margin=train_confidence_min_margin,
            confidence_split_name="train",
            confidence_resample_to_before=train_confidence_resample_to_before,
            confidence_resample_seed=train_seed,
        )
        x_eval_split, y_eval_idx, _, eval_summary = _split_fashion_sandal_sole_height_dataset(
            images_eval_raw,
            source_class_name=source_class_name,
            foreground_threshold=foreground_threshold,
            sole_width_fraction=sole_width_fraction,
            bottom_band_fraction=bottom_band_fraction,
            threshold=float(train_summary["score_threshold"]),
            confidence_keep_fraction=eval_confidence_keep_fraction,
            confidence_min_margin=eval_confidence_min_margin,
            confidence_split_name="eval",
            confidence_resample_to_before=eval_confidence_resample_to_before,
            confidence_resample_seed=eval_seed,
        )
        split_summary = {
            "rule": "sandal_sole_height",
            "train": train_summary,
            "eval": eval_summary,
        }
        return x_train_split, y_train_idx, x_eval_split, y_eval_idx, active_names, split_summary

    if rule in {"bag_top_handle", "top_handle"}:
        if _canonical_dataset_label_name(source_class_name) != _canonical_dataset_label_name("Bag"):
            raise ValueError(
                "fashion_mnist_subclass_composite_single semantic_split.bag_top_handle currently only supports "
                "the base class 'Bag'"
            )
        body_row_width_fraction = float(split_cfg.get("body_row_width_fraction", 0.62))
        top_band_fraction = float(split_cfg.get("top_band_fraction", 0.42))
        center_window_fraction = float(split_cfg.get("center_window_fraction", 0.45))
        confidence_keep_fraction = float(split_cfg.get("confidence_keep_fraction", 1.0))
        confidence_min_margin = float(split_cfg.get("confidence_min_margin", 0.0))
        confidence_resample_to_before = bool(split_cfg.get("confidence_resample_to_before", False))
        train_confidence_keep_fraction = float(
            split_cfg.get("train_confidence_keep_fraction", confidence_keep_fraction)
        )
        train_confidence_min_margin = float(split_cfg.get("train_confidence_min_margin", confidence_min_margin))
        train_confidence_resample_to_before = bool(
            split_cfg.get("train_confidence_resample_to_before", confidence_resample_to_before)
        )
        eval_confidence_keep_fraction = float(split_cfg.get("eval_confidence_keep_fraction", confidence_keep_fraction))
        eval_confidence_min_margin = float(split_cfg.get("eval_confidence_min_margin", confidence_min_margin))
        eval_confidence_resample_to_before = bool(
            split_cfg.get("eval_confidence_resample_to_before", confidence_resample_to_before)
        )
        if not (0.0 < body_row_width_fraction <= 1.0):
            raise ValueError(
                "fashion_mnist_subclass_composite_single semantic_split.body_row_width_fraction must be in (0, 1]"
            )
        if not (0.0 < top_band_fraction <= 1.0):
            raise ValueError(
                "fashion_mnist_subclass_composite_single semantic_split.top_band_fraction must be in (0, 1]"
            )
        if not (0.0 < center_window_fraction <= 1.0):
            raise ValueError(
                "fashion_mnist_subclass_composite_single semantic_split.center_window_fraction must be in (0, 1]"
            )
        x_train_split, y_train_idx, active_names, train_summary = _split_fashion_bag_top_handle_dataset(
            images_train_raw,
            source_class_name=source_class_name,
            foreground_threshold=foreground_threshold,
            body_row_width_fraction=body_row_width_fraction,
            top_band_fraction=top_band_fraction,
            center_window_fraction=center_window_fraction,
            threshold=None,
            confidence_keep_fraction=train_confidence_keep_fraction,
            confidence_min_margin=train_confidence_min_margin,
            confidence_split_name="train",
            confidence_resample_to_before=train_confidence_resample_to_before,
            confidence_resample_seed=train_seed,
        )
        x_eval_split, y_eval_idx, _, eval_summary = _split_fashion_bag_top_handle_dataset(
            images_eval_raw,
            source_class_name=source_class_name,
            foreground_threshold=foreground_threshold,
            body_row_width_fraction=body_row_width_fraction,
            top_band_fraction=top_band_fraction,
            center_window_fraction=center_window_fraction,
            threshold=float(train_summary["score_threshold"]),
            confidence_keep_fraction=eval_confidence_keep_fraction,
            confidence_min_margin=eval_confidence_min_margin,
            confidence_split_name="eval",
            confidence_resample_to_before=eval_confidence_resample_to_before,
            confidence_resample_seed=eval_seed,
        )
        split_summary = {
            "rule": "bag_top_handle",
            "train": train_summary,
            "eval": eval_summary,
        }
        return x_train_split, y_train_idx, x_eval_split, y_eval_idx, active_names, split_summary

    raise ValueError(
        "fashion_mnist_subclass_composite_single subclass_specs.rule must be one of "
        "{'synthetic_variants', 'sandal_sole_height', 'bag_top_handle'}"
    )


def _make_fashion_mnist_subclass_composite_single_dataset(
    *,
    dataset_cfg: Mapping[str, Any],
    num_inputs: int,
    num_outputs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Dict[str, Any]]:
    if num_inputs != 16:
        raise ValueError(
            f"fashion_mnist_subclass_composite_single requires model input_dim=16, got {num_inputs}"
        )

    subclass_specs_raw = dataset_cfg.get("subclass_specs")
    if not isinstance(subclass_specs_raw, Sequence) or isinstance(subclass_specs_raw, (str, bytes)):
        raise ValueError(
            "fashion_mnist_subclass_composite_single requires train.dataset.subclass_specs as a sequence"
        )
    subclass_specs = [dict(spec or {}) for spec in subclass_specs_raw]
    if not subclass_specs:
        raise ValueError("fashion_mnist_subclass_composite_single requires at least one subclass spec")

    input_low = float(dataset_cfg.get("input_low", 0.0))
    input_high = float(dataset_cfg.get("input_high", 2.0))
    if input_high <= input_low:
        raise ValueError("train.dataset.input_high must be greater than input_low")

    pca_components = int(dataset_cfg.get("pca_components", num_inputs))
    if pca_components != num_inputs:
        raise ValueError(
            "fashion_mnist_subclass_composite_single requires "
            f"train.dataset.pca_components == model input_dim={num_inputs}, got {pca_components}"
        )

    data_home = get_project_root() / str(dataset_cfg.get("data_home", "data/cache/fashion_mnist"))
    raw = _load_fashion_mnist_raw_arrays(data_home)
    class_name_lookup = {
        _canonical_dataset_label_name(name): int(idx)
        for idx, name in enumerate(FASHION_MNIST_CLASS_NAMES)
    }

    x_train_parts: list[np.ndarray] = []
    x_eval_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []
    y_eval_parts: list[np.ndarray] = []
    active_names: list[str] = []
    component_summaries: list[Dict[str, Any]] = []
    selected_base_names: list[str] = []
    next_class_offset = 0

    for spec_idx, spec in enumerate(subclass_specs):
        class_name = spec.get("class_name", spec.get("source_class_name"))
        if class_name is None:
            raise ValueError(
                f"fashion_mnist_subclass_composite_single subclass_specs[{spec_idx}] requires class_name"
            )
        class_key = _canonical_dataset_label_name(class_name)
        if class_key not in class_name_lookup:
            raise ValueError(
                f"Unknown Fashion-MNIST class {class_name!r}; expected one of {list(FASHION_MNIST_CLASS_NAMES)}"
            )
        class_idx = int(class_name_lookup[class_key])
        source_class_name = str(FASHION_MNIST_CLASS_NAMES[class_idx])
        if source_class_name in selected_base_names:
            raise ValueError(
                f"fashion_mnist_subclass_composite_single duplicated base class {source_class_name!r}"
            )
        selected_base_names.append(source_class_name)

        train_mask = raw["y_train"] == class_idx
        eval_mask = raw["y_eval"] == class_idx
        x_train_raw = raw["x_train"][train_mask]
        x_eval_raw = raw["x_eval"][eval_mask]
        x_train_split, y_train_local, x_eval_split, y_eval_local, component_names, component_summary = _apply_fashion_subclass_rule(
            images_train_raw=x_train_raw,
            images_eval_raw=x_eval_raw,
            source_class_name=source_class_name,
            split_cfg=spec,
            train_seed=seed + spec_idx * 37,
            eval_seed=seed + spec_idx * 37 + 1,
        )

        component_class_count = len(component_names)
        y_train_parts.append(np.asarray(y_train_local, dtype=np.int64) + next_class_offset)
        y_eval_parts.append(np.asarray(y_eval_local, dtype=np.int64) + next_class_offset)
        x_train_parts.append(np.asarray(x_train_split, dtype=np.float32))
        x_eval_parts.append(np.asarray(x_eval_split, dtype=np.float32))
        active_names.extend([str(name) for name in component_names])
        component_summaries.append(
            {
                "source_class_name": source_class_name,
                "source_class_index": int(class_idx),
                "active_class_names": list(component_names),
                "active_class_offset": int(next_class_offset),
                "active_class_count": int(component_class_count),
                "split": component_summary,
            }
        )
        next_class_offset += int(component_class_count)

    active_class_count = int(next_class_offset)
    if active_class_count > num_outputs:
        raise ValueError(
            f"fashion_mnist_subclass_composite_single active class count {active_class_count} "
            f"exceeds model output_dim={num_outputs}"
        )

    x_train_raw = np.concatenate(x_train_parts, axis=0).astype(np.float32)
    x_eval_raw = np.concatenate(x_eval_parts, axis=0).astype(np.float32)
    y_train_idx = np.concatenate(y_train_parts, axis=0).astype(np.int64)
    y_eval_idx = np.concatenate(y_eval_parts, axis=0).astype(np.int64)

    train_limit_raw = dataset_cfg.get("train_limit")
    eval_limit_raw = dataset_cfg.get("eval_limit")
    train_limit = None if train_limit_raw is None else int(train_limit_raw)
    eval_limit = None if eval_limit_raw is None else int(eval_limit_raw)
    train_subset = _subsample_split_indices(y_train_idx, limit=train_limit, seed=seed)
    eval_subset = _subsample_split_indices(y_eval_idx, limit=eval_limit, seed=seed + 1)
    x_train_raw = x_train_raw[train_subset]
    x_eval_raw = x_eval_raw[eval_subset]
    y_train_idx = y_train_idx[train_subset]
    y_eval_idx = y_eval_idx[eval_subset]

    pca = PCA(
        n_components=pca_components,
        svd_solver=str(dataset_cfg.get("pca_solver", "randomized")),
        whiten=bool(dataset_cfg.get("pca_whiten", False)),
        random_state=seed,
    )
    x_train_pca = pca.fit_transform(x_train_raw).astype(np.float32)
    x_eval_pca = pca.transform(x_eval_raw).astype(np.float32)
    x_train, x_eval = _scale_feature_branch(
        x_train_pca,
        x_eval_pca,
        input_low=input_low,
        input_high=input_high,
    )

    y_train = _one_hot_with_output_dim(y_train_idx, output_dim=num_outputs)
    y_eval = _one_hot_with_output_dim(y_eval_idx, output_dim=num_outputs)
    explained_variance_ratio = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    explained = float(np.sum(explained_variance_ratio))
    dataset_summary = {
        "dataset_name": "fashion_mnist_subclass_composite_single",
        "target_level": "composite_subclass",
        "selected_base_class_names": list(selected_base_names),
        "class_names_active": list(active_names),
        "class_names_padded": _pad_class_names(active_names, output_dim=num_outputs),
        "active_class_count": int(active_class_count),
        "subclass_specs": component_summaries,
        "pca_components": int(pca_components),
        "component_split": "single_branch",
        "input_component_indices": list(range(int(pca_components))),
        "scaled_input_low": float(input_low),
        "scaled_input_high": float(input_high),
        "input_train_min": float(np.min(x_train)),
        "input_train_max": float(np.max(x_train)),
        "input_train_mean": float(np.mean(x_train)),
        "input_train_std": float(np.std(x_train)),
        "explained_variance_ratio_sum": explained,
        "train_samples": int(x_train.shape[0]),
        "eval_samples": int(x_eval.shape[0]),
        "train_limit_applied": None if train_limit is None else int(train_limit),
        "eval_limit_applied": None if eval_limit is None else int(eval_limit),
    }
    return x_train, x_eval, y_train, y_eval, explained, dataset_summary


def _make_fashion_mnist_subclass_composite_dual_dataset(
    *,
    dataset_cfg: Mapping[str, Any],
    num_branch_inputs: int,
    num_outputs: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    Dict[str, Any],
]:
    if num_branch_inputs != 16:
        raise ValueError(
            "fashion_mnist_subclass_composite_dual requires model input_dim=16 per wavelength, "
            f"got {num_branch_inputs}"
        )

    single_cfg = dict(dataset_cfg)
    x_train_a, x_eval_a, y_train, y_eval, explained, single_summary = _make_fashion_mnist_subclass_composite_single_dataset(
        dataset_cfg=single_cfg,
        num_inputs=num_branch_inputs,
        num_outputs=num_outputs,
        seed=seed,
    )

    raw = _load_fashion_mnist_raw_arrays(
        get_project_root() / str(dataset_cfg.get("data_home", "data/cache/fashion_mnist"))
    )
    subclass_specs = [dict(spec or {}) for spec in dataset_cfg.get("subclass_specs", [])]
    class_name_lookup = {
        _canonical_dataset_label_name(name): int(idx)
        for idx, name in enumerate(FASHION_MNIST_CLASS_NAMES)
    }

    x_train_raw_parts: list[np.ndarray] = []
    x_eval_raw_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []
    y_eval_parts: list[np.ndarray] = []
    next_class_offset = 0
    for spec_idx, spec in enumerate(subclass_specs):
        class_name = spec.get("class_name", spec.get("source_class_name"))
        if class_name is None:
            raise ValueError(
                f"fashion_mnist_subclass_composite_dual subclass_specs[{spec_idx}] requires class_name"
            )
        class_key = _canonical_dataset_label_name(class_name)
        if class_key not in class_name_lookup:
            raise ValueError(
                f"Unknown Fashion-MNIST class {class_name!r}; expected one of {list(FASHION_MNIST_CLASS_NAMES)}"
            )
        class_idx = int(class_name_lookup[class_key])
        source_class_name = str(FASHION_MNIST_CLASS_NAMES[class_idx])
        train_mask = raw["y_train"] == class_idx
        eval_mask = raw["y_eval"] == class_idx
        x_train_split, y_train_local, x_eval_split, y_eval_local, component_names, _ = _apply_fashion_subclass_rule(
            images_train_raw=raw["x_train"][train_mask],
            images_eval_raw=raw["x_eval"][eval_mask],
            source_class_name=source_class_name,
            split_cfg=spec,
            train_seed=seed + spec_idx * 37,
            eval_seed=seed + spec_idx * 37 + 1,
        )
        x_train_raw_parts.append(np.asarray(x_train_split, dtype=np.float32))
        x_eval_raw_parts.append(np.asarray(x_eval_split, dtype=np.float32))
        y_train_parts.append(np.asarray(y_train_local, dtype=np.int64) + next_class_offset)
        y_eval_parts.append(np.asarray(y_eval_local, dtype=np.int64) + next_class_offset)
        next_class_offset += len(component_names)

    x_train_raw = np.concatenate(x_train_raw_parts, axis=0).astype(np.float32)
    x_eval_raw = np.concatenate(x_eval_raw_parts, axis=0).astype(np.float32)
    y_train_idx = np.concatenate(y_train_parts, axis=0).astype(np.int64)
    y_eval_idx = np.concatenate(y_eval_parts, axis=0).astype(np.int64)

    train_limit_raw = dataset_cfg.get("train_limit")
    eval_limit_raw = dataset_cfg.get("eval_limit")
    train_limit = None if train_limit_raw is None else int(train_limit_raw)
    eval_limit = None if eval_limit_raw is None else int(eval_limit_raw)
    train_subset = _subsample_split_indices(y_train_idx, limit=train_limit, seed=seed)
    eval_subset = _subsample_split_indices(y_eval_idx, limit=eval_limit, seed=seed + 1)
    x_train_raw = x_train_raw[train_subset]
    x_eval_raw = x_eval_raw[eval_subset]
    y_train_idx = y_train_idx[train_subset]
    y_eval_idx = y_eval_idx[eval_subset]

    avgpool_grid_size = int(dataset_cfg.get("avgpool_grid_size", 4))
    if avgpool_grid_size * avgpool_grid_size != num_branch_inputs:
        raise ValueError(
            "fashion_mnist_subclass_composite_dual requires avgpool_grid_size^2 == branch input_dim, "
            f"got grid_size={avgpool_grid_size}, branch_input_dim={num_branch_inputs}"
        )
    input_low = float(dataset_cfg.get("input_low", 0.0))
    input_high = float(dataset_cfg.get("input_high", 2.0))
    branch_b_feature_kind_raw = str(
        dataset_cfg.get("branch_b_feature_kind", f"avgpool{avgpool_grid_size}x{avgpool_grid_size}_grayscale")
    ).strip()
    branch_b_feature_kind = branch_b_feature_kind_raw.lower()
    if branch_b_feature_kind in {
        "avgpool",
        "avgpool16",
        f"avgpool{avgpool_grid_size}x{avgpool_grid_size}",
        f"avgpool{avgpool_grid_size}x{avgpool_grid_size}_grayscale",
    }:
        x_train_values, pool_block_shape = _make_mnist_avgpool_features(
            x_train_raw,
            grid_size=avgpool_grid_size,
        )
        x_eval_values, eval_pool_block_shape = _make_mnist_avgpool_features(
            x_eval_raw,
            grid_size=avgpool_grid_size,
        )
        if eval_pool_block_shape != pool_block_shape:
            raise ValueError(
                "Fashion-MNIST avg-pool block shape mismatch between train and eval: "
                f"train={pool_block_shape}, eval={eval_pool_block_shape}"
            )
        branch_b_metadata = {
            "feature_kind": f"avgpool{avgpool_grid_size}x{avgpool_grid_size}_grayscale",
            "pool_grid_shape": [avgpool_grid_size, avgpool_grid_size],
            "pool_block_shape": pool_block_shape,
        }
    elif branch_b_feature_kind in {
        "local_texture",
        "local_texture16",
        f"local_texture{avgpool_grid_size}x{avgpool_grid_size}",
        f"local_texture{avgpool_grid_size}x{avgpool_grid_size}_grayscale",
        "local_edge_contrast",
    }:
        edge_weight = float(dataset_cfg.get("local_texture_edge_weight", 0.7))
        contrast_weight = float(dataset_cfg.get("local_texture_contrast_weight", 0.3))
        x_train_values, texture_block_shape = _make_mnist_local_texture_features(
            x_train_raw,
            grid_size=avgpool_grid_size,
            edge_weight=edge_weight,
            contrast_weight=contrast_weight,
        )
        x_eval_values, eval_texture_block_shape = _make_mnist_local_texture_features(
            x_eval_raw,
            grid_size=avgpool_grid_size,
            edge_weight=edge_weight,
            contrast_weight=contrast_weight,
        )
        if eval_texture_block_shape != texture_block_shape:
            raise ValueError(
                "Fashion-MNIST local-texture block shape mismatch between train and eval: "
                f"train={texture_block_shape}, eval={eval_texture_block_shape}"
            )
        branch_b_metadata = {
            "feature_kind": f"local_texture{avgpool_grid_size}x{avgpool_grid_size}_grayscale",
            "texture_grid_shape": [avgpool_grid_size, avgpool_grid_size],
            "texture_block_shape": texture_block_shape,
            "local_texture_edge_weight": float(edge_weight),
            "local_texture_contrast_weight": float(contrast_weight),
        }
    elif branch_b_feature_kind in {
        "oriented_texture",
        "oriented_texture16",
        f"oriented_texture{avgpool_grid_size}x{avgpool_grid_size}",
        f"oriented_texture{avgpool_grid_size}x{avgpool_grid_size}_grayscale",
        "directional_texture",
        "directional_texture16",
        f"directional_texture{avgpool_grid_size}x{avgpool_grid_size}",
        f"directional_texture{avgpool_grid_size}x{avgpool_grid_size}_grayscale",
    }:
        edge_weight = float(dataset_cfg.get("oriented_texture_edge_weight", 0.55))
        orientation_weight = float(dataset_cfg.get("oriented_texture_orientation_weight", 0.65))
        contrast_weight = float(dataset_cfg.get("oriented_texture_contrast_weight", 0.20))
        x_train_values, texture_block_shape = _make_mnist_oriented_texture_features(
            x_train_raw,
            grid_size=avgpool_grid_size,
            edge_weight=edge_weight,
            orientation_weight=orientation_weight,
            contrast_weight=contrast_weight,
        )
        x_eval_values, eval_texture_block_shape = _make_mnist_oriented_texture_features(
            x_eval_raw,
            grid_size=avgpool_grid_size,
            edge_weight=edge_weight,
            orientation_weight=orientation_weight,
            contrast_weight=contrast_weight,
        )
        if eval_texture_block_shape != texture_block_shape:
            raise ValueError(
                "Fashion-MNIST oriented-texture block shape mismatch between train and eval: "
                f"train={texture_block_shape}, eval={eval_texture_block_shape}"
            )
        branch_b_metadata = {
            "feature_kind": f"oriented_texture{avgpool_grid_size}x{avgpool_grid_size}_grayscale",
            "texture_grid_shape": [avgpool_grid_size, avgpool_grid_size],
            "texture_block_shape": texture_block_shape,
            "oriented_texture_edge_weight": float(edge_weight),
            "oriented_texture_orientation_weight": float(orientation_weight),
            "oriented_texture_contrast_weight": float(contrast_weight),
        }
    elif branch_b_feature_kind in {
        "center_shape_blend",
        "center_shape_blend16",
        f"center_shape_blend{avgpool_grid_size}x{avgpool_grid_size}",
        f"center_shape_blend{avgpool_grid_size}x{avgpool_grid_size}_grayscale",
        "center_icon_shape_blend",
        "center_icon_shape_blend16",
        f"center_icon_shape_blend{avgpool_grid_size}x{avgpool_grid_size}",
        f"center_icon_shape_blend{avgpool_grid_size}x{avgpool_grid_size}_grayscale",
    }:
        crop_fraction = float(dataset_cfg.get("center_icon_crop_fraction", 0.58))
        deviation_weight = float(dataset_cfg.get("center_icon_deviation_weight", 0.85))
        edge_weight = float(dataset_cfg.get("center_icon_edge_weight", 0.45))
        contrast_weight = float(dataset_cfg.get("center_icon_contrast_weight", 0.20))
        center_weight = float(dataset_cfg.get("center_shape_blend_center_weight", 1.0))
        shape_weight = float(dataset_cfg.get("center_shape_blend_shape_weight", 0.65))
        shape_mass_weight = float(dataset_cfg.get("center_shape_descriptor_mass_weight", 0.80))
        shape_edge_weight = float(dataset_cfg.get("center_shape_descriptor_edge_weight", 0.95))
        shape_orientation_weight = float(dataset_cfg.get("center_shape_descriptor_orientation_weight", 0.70))
        shape_moment_weight = float(dataset_cfg.get("center_shape_descriptor_moment_weight", 0.55))
        x_train_values, texture_block_shape, crop_box = _make_mnist_center_shape_blend_features(
            x_train_raw,
            grid_size=avgpool_grid_size,
            crop_fraction=crop_fraction,
            deviation_weight=deviation_weight,
            edge_weight=edge_weight,
            contrast_weight=contrast_weight,
            center_weight=center_weight,
            shape_weight=shape_weight,
            shape_mass_weight=shape_mass_weight,
            shape_edge_weight=shape_edge_weight,
            shape_orientation_weight=shape_orientation_weight,
            shape_moment_weight=shape_moment_weight,
        )
        x_eval_values, eval_texture_block_shape, eval_crop_box = _make_mnist_center_shape_blend_features(
            x_eval_raw,
            grid_size=avgpool_grid_size,
            crop_fraction=crop_fraction,
            deviation_weight=deviation_weight,
            edge_weight=edge_weight,
            contrast_weight=contrast_weight,
            center_weight=center_weight,
            shape_weight=shape_weight,
            shape_mass_weight=shape_mass_weight,
            shape_edge_weight=shape_edge_weight,
            shape_orientation_weight=shape_orientation_weight,
            shape_moment_weight=shape_moment_weight,
        )
        if eval_texture_block_shape != texture_block_shape or eval_crop_box != crop_box:
            raise ValueError(
                "Fashion-MNIST center-shape blend block shape mismatch between train and eval: "
                f"train={texture_block_shape}/{crop_box}, eval={eval_texture_block_shape}/{eval_crop_box}"
            )
        branch_b_metadata = {
            "feature_kind": f"center_shape_blend{avgpool_grid_size}x{avgpool_grid_size}_grayscale",
            "texture_grid_shape": [avgpool_grid_size, avgpool_grid_size],
            "texture_block_shape": texture_block_shape,
            "center_icon_crop_box": crop_box,
            "center_icon_crop_fraction": float(crop_fraction),
            "center_icon_deviation_weight": float(deviation_weight),
            "center_icon_edge_weight": float(edge_weight),
            "center_icon_contrast_weight": float(contrast_weight),
            "center_shape_blend_center_weight": float(center_weight),
            "center_shape_blend_shape_weight": float(shape_weight),
            "center_shape_descriptor_mass_weight": float(shape_mass_weight),
            "center_shape_descriptor_edge_weight": float(shape_edge_weight),
            "center_shape_descriptor_orientation_weight": float(shape_orientation_weight),
            "center_shape_descriptor_moment_weight": float(shape_moment_weight),
            "center_shape_descriptor_dim": 16,
            "uses_target_labels_for_feature_routing": False,
        }
    elif branch_b_feature_kind in {
        "center_icon_local",
        "center_icon_local16",
        f"center_icon_local{avgpool_grid_size}x{avgpool_grid_size}",
        f"center_icon_local{avgpool_grid_size}x{avgpool_grid_size}_grayscale",
        "center_icon_texture",
        "center_icon_texture16",
        f"center_icon_texture{avgpool_grid_size}x{avgpool_grid_size}",
        f"center_icon_texture{avgpool_grid_size}x{avgpool_grid_size}_grayscale",
    }:
        crop_fraction = float(dataset_cfg.get("center_icon_crop_fraction", 0.58))
        deviation_weight = float(dataset_cfg.get("center_icon_deviation_weight", 0.85))
        edge_weight = float(dataset_cfg.get("center_icon_edge_weight", 0.45))
        contrast_weight = float(dataset_cfg.get("center_icon_contrast_weight", 0.20))
        x_train_values, texture_block_shape, crop_box = _make_mnist_center_icon_local_features(
            x_train_raw,
            grid_size=avgpool_grid_size,
            crop_fraction=crop_fraction,
            deviation_weight=deviation_weight,
            edge_weight=edge_weight,
            contrast_weight=contrast_weight,
        )
        x_eval_values, eval_texture_block_shape, eval_crop_box = _make_mnist_center_icon_local_features(
            x_eval_raw,
            grid_size=avgpool_grid_size,
            crop_fraction=crop_fraction,
            deviation_weight=deviation_weight,
            edge_weight=edge_weight,
            contrast_weight=contrast_weight,
        )
        if eval_texture_block_shape != texture_block_shape or eval_crop_box != crop_box:
            raise ValueError(
                "Fashion-MNIST center-icon local block shape mismatch between train and eval: "
                f"train={texture_block_shape}/{crop_box}, eval={eval_texture_block_shape}/{eval_crop_box}"
            )
        branch_b_metadata = {
            "feature_kind": f"center_icon_local{avgpool_grid_size}x{avgpool_grid_size}_grayscale",
            "texture_grid_shape": [avgpool_grid_size, avgpool_grid_size],
            "texture_block_shape": texture_block_shape,
            "center_icon_crop_box": crop_box,
            "center_icon_crop_fraction": float(crop_fraction),
            "center_icon_deviation_weight": float(deviation_weight),
            "center_icon_edge_weight": float(edge_weight),
            "center_icon_contrast_weight": float(contrast_weight),
        }
    else:
        raise ValueError(
            "Unsupported train.dataset.branch_b_feature_kind="
            f"{branch_b_feature_kind_raw!r}. Expected one of "
            f"{{'avgpool{avgpool_grid_size}x{avgpool_grid_size}_grayscale', "
            f"'local_texture{avgpool_grid_size}x{avgpool_grid_size}_grayscale', "
            f"'oriented_texture{avgpool_grid_size}x{avgpool_grid_size}_grayscale', "
            f"'center_shape_blend{avgpool_grid_size}x{avgpool_grid_size}_grayscale', "
            f"'center_icon_local{avgpool_grid_size}x{avgpool_grid_size}_grayscale'}}"
        )
    x_train_b, x_eval_b = _scale_feature_branch(
        x_train_values,
        x_eval_values,
        input_low=input_low,
        input_high=input_high,
    )

    dataset_summary = dict(single_summary)
    dataset_summary.update(
        {
            "dataset_name": "fashion_mnist_subclass_composite_dual",
            "component_split": "dual_branch",
            "branch_a_feature_kind": "pca16_flat",
            "branch_b_feature_kind": str(branch_b_metadata["feature_kind"]),
            "branch_a_explained_variance_ratio_sum": float(explained),
            "branch_b_train_min": float(np.min(x_train_b)),
            "branch_b_train_max": float(np.max(x_train_b)),
            "branch_b_train_mean": float(np.mean(x_train_b)),
            "branch_b_train_std": float(np.std(x_train_b)),
            "branch_a_train_min": float(np.min(x_train_a)),
            "branch_a_train_max": float(np.max(x_train_a)),
            "branch_a_train_mean": float(np.mean(x_train_a)),
            "branch_a_train_std": float(np.std(x_train_a)),
        }
    )
    if "pool_grid_shape" in branch_b_metadata:
        dataset_summary["branch_b_pool_grid_shape"] = list(branch_b_metadata["pool_grid_shape"])
    if "pool_block_shape" in branch_b_metadata:
        dataset_summary["branch_b_pool_block_shape"] = [int(v) for v in branch_b_metadata["pool_block_shape"]]
    if "texture_grid_shape" in branch_b_metadata:
        dataset_summary["branch_b_texture_grid_shape"] = list(branch_b_metadata["texture_grid_shape"])
    if "texture_block_shape" in branch_b_metadata:
        dataset_summary["branch_b_texture_block_shape"] = [int(v) for v in branch_b_metadata["texture_block_shape"]]
    if "local_texture_edge_weight" in branch_b_metadata:
        dataset_summary["branch_b_local_texture_edge_weight"] = float(
            branch_b_metadata["local_texture_edge_weight"]
        )
    if "local_texture_contrast_weight" in branch_b_metadata:
        dataset_summary["branch_b_local_texture_contrast_weight"] = float(
            branch_b_metadata["local_texture_contrast_weight"]
        )
    if "oriented_texture_edge_weight" in branch_b_metadata:
        dataset_summary["branch_b_oriented_texture_edge_weight"] = float(
            branch_b_metadata["oriented_texture_edge_weight"]
        )
    if "oriented_texture_orientation_weight" in branch_b_metadata:
        dataset_summary["branch_b_oriented_texture_orientation_weight"] = float(
            branch_b_metadata["oriented_texture_orientation_weight"]
        )
    if "oriented_texture_contrast_weight" in branch_b_metadata:
        dataset_summary["branch_b_oriented_texture_contrast_weight"] = float(
            branch_b_metadata["oriented_texture_contrast_weight"]
        )
    if "center_icon_crop_box" in branch_b_metadata:
        dataset_summary["branch_b_center_icon_crop_box"] = [
            int(v) for v in branch_b_metadata["center_icon_crop_box"]
        ]
    if "center_icon_crop_fraction" in branch_b_metadata:
        dataset_summary["branch_b_center_icon_crop_fraction"] = float(
            branch_b_metadata["center_icon_crop_fraction"]
        )
    if "center_icon_deviation_weight" in branch_b_metadata:
        dataset_summary["branch_b_center_icon_deviation_weight"] = float(
            branch_b_metadata["center_icon_deviation_weight"]
        )
    if "center_icon_edge_weight" in branch_b_metadata:
        dataset_summary["branch_b_center_icon_edge_weight"] = float(branch_b_metadata["center_icon_edge_weight"])
    if "center_icon_contrast_weight" in branch_b_metadata:
        dataset_summary["branch_b_center_icon_contrast_weight"] = float(
            branch_b_metadata["center_icon_contrast_weight"]
        )
    if "center_shape_blend_center_weight" in branch_b_metadata:
        dataset_summary["branch_b_center_shape_blend_center_weight"] = float(
            branch_b_metadata["center_shape_blend_center_weight"]
        )
    if "center_shape_blend_shape_weight" in branch_b_metadata:
        dataset_summary["branch_b_center_shape_blend_shape_weight"] = float(
            branch_b_metadata["center_shape_blend_shape_weight"]
        )
    if "center_shape_descriptor_mass_weight" in branch_b_metadata:
        dataset_summary["branch_b_center_shape_descriptor_mass_weight"] = float(
            branch_b_metadata["center_shape_descriptor_mass_weight"]
        )
    if "center_shape_descriptor_edge_weight" in branch_b_metadata:
        dataset_summary["branch_b_center_shape_descriptor_edge_weight"] = float(
            branch_b_metadata["center_shape_descriptor_edge_weight"]
        )
    if "center_shape_descriptor_orientation_weight" in branch_b_metadata:
        dataset_summary["branch_b_center_shape_descriptor_orientation_weight"] = float(
            branch_b_metadata["center_shape_descriptor_orientation_weight"]
        )
    if "center_shape_descriptor_moment_weight" in branch_b_metadata:
        dataset_summary["branch_b_center_shape_descriptor_moment_weight"] = float(
            branch_b_metadata["center_shape_descriptor_moment_weight"]
        )
    if "center_shape_descriptor_dim" in branch_b_metadata:
        dataset_summary["branch_b_center_shape_descriptor_dim"] = int(
            branch_b_metadata["center_shape_descriptor_dim"]
        )
    if "uses_target_labels_for_feature_routing" in branch_b_metadata:
        dataset_summary["branch_b_uses_target_labels_for_feature_routing"] = bool(
            branch_b_metadata["uses_target_labels_for_feature_routing"]
        )
    return x_train_a, x_train_b, x_eval_a, x_eval_b, y_train, y_eval, explained, dataset_summary


def _make_mnist_avgpool_features(
    inputs: np.ndarray,
    *,
    grid_size: int,
) -> tuple[np.ndarray, list[int]]:
    images = np.asarray(inputs, dtype=np.float32)
    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError(f"Expected flat MNIST inputs [N, 784], got {images.shape}")
    images = images.reshape(-1, 28, 28)
    if images.shape[1] % grid_size != 0 or images.shape[2] % grid_size != 0:
        raise ValueError(
            f"MNIST image shape {(images.shape[1], images.shape[2])} is not divisible by grid_size={grid_size}"
        )
    block_h = images.shape[1] // grid_size
    block_w = images.shape[2] // grid_size
    pooled = images.reshape(-1, grid_size, block_h, grid_size, block_w).mean(axis=(2, 4))
    return pooled.reshape(images.shape[0], grid_size * grid_size).astype(np.float32), [block_h, block_w]


def _make_mnist_local_texture_features(
    inputs: np.ndarray,
    *,
    grid_size: int,
    edge_weight: float,
    contrast_weight: float,
) -> tuple[np.ndarray, list[int]]:
    images = np.asarray(inputs, dtype=np.float32)
    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError(f"Expected flat MNIST inputs [N, 784], got {images.shape}")
    images = images.reshape(-1, 28, 28)
    if images.shape[1] % grid_size != 0 or images.shape[2] % grid_size != 0:
        raise ValueError(
            f"MNIST image shape {(images.shape[1], images.shape[2])} is not divisible by grid_size={grid_size}"
        )
    if edge_weight < 0.0 or contrast_weight < 0.0 or edge_weight + contrast_weight <= 0.0:
        raise ValueError("local texture feature weights must be non-negative and have a positive sum")

    block_h = images.shape[1] // grid_size
    block_w = images.shape[2] // grid_size
    gx = np.zeros_like(images, dtype=np.float32)
    gy = np.zeros_like(images, dtype=np.float32)
    gx[:, :, 1:] = np.abs(images[:, :, 1:] - images[:, :, :-1])
    gy[:, 1:, :] = np.abs(images[:, 1:, :] - images[:, :-1, :])
    edge = gx + gy

    image_blocks = images.reshape(-1, grid_size, block_h, grid_size, block_w)
    edge_blocks = edge.reshape(-1, grid_size, block_h, grid_size, block_w)
    block_edge = edge_blocks.mean(axis=(2, 4))
    block_mean = image_blocks.mean(axis=(2, 4))
    block_sq_mean = (image_blocks * image_blocks).mean(axis=(2, 4))
    block_contrast = np.sqrt(np.maximum(block_sq_mean - block_mean * block_mean, 0.0))
    features = float(edge_weight) * block_edge + float(contrast_weight) * block_contrast
    return features.reshape(images.shape[0], grid_size * grid_size).astype(np.float32), [block_h, block_w]


def _make_mnist_oriented_texture_features(
    inputs: np.ndarray,
    *,
    grid_size: int,
    edge_weight: float,
    orientation_weight: float,
    contrast_weight: float,
) -> tuple[np.ndarray, list[int]]:
    images = np.asarray(inputs, dtype=np.float32)
    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError(f"Expected flat MNIST inputs [N, 784], got {images.shape}")
    images = images.reshape(-1, 28, 28)
    if images.shape[1] % grid_size != 0 or images.shape[2] % grid_size != 0:
        raise ValueError(
            f"MNIST image shape {(images.shape[1], images.shape[2])} is not divisible by grid_size={grid_size}"
        )
    if edge_weight < 0.0 or orientation_weight < 0.0 or contrast_weight < 0.0:
        raise ValueError("oriented texture feature weights must be non-negative")
    if edge_weight + orientation_weight + contrast_weight <= 0.0:
        raise ValueError("oriented texture feature weights must have a positive sum")

    block_h = images.shape[1] // grid_size
    block_w = images.shape[2] // grid_size
    gx = np.zeros_like(images, dtype=np.float32)
    gy = np.zeros_like(images, dtype=np.float32)
    gx[:, :, 1:] = np.abs(images[:, :, 1:] - images[:, :, :-1])
    gy[:, 1:, :] = np.abs(images[:, 1:, :] - images[:, :-1, :])

    image_blocks = images.reshape(-1, grid_size, block_h, grid_size, block_w)
    gx_blocks = gx.reshape(-1, grid_size, block_h, grid_size, block_w)
    gy_blocks = gy.reshape(-1, grid_size, block_h, grid_size, block_w)
    block_gx = gx_blocks.mean(axis=(2, 4))
    block_gy = gy_blocks.mean(axis=(2, 4))
    block_mean = image_blocks.mean(axis=(2, 4))
    block_sq_mean = (image_blocks * image_blocks).mean(axis=(2, 4))
    block_contrast = np.sqrt(np.maximum(block_sq_mean - block_mean * block_mean, 0.0))

    edge_energy = block_gx + block_gy
    orientation_bias = block_gy - block_gx
    features = (
        float(edge_weight) * edge_energy
        + float(orientation_weight) * orientation_bias
        + float(contrast_weight) * block_contrast
    )
    return features.reshape(images.shape[0], grid_size * grid_size).astype(np.float32), [block_h, block_w]


def _make_mnist_center_icon_local_features(
    inputs: np.ndarray,
    *,
    grid_size: int,
    crop_fraction: float,
    deviation_weight: float,
    edge_weight: float,
    contrast_weight: float,
) -> tuple[np.ndarray, list[int], list[int]]:
    images = np.asarray(inputs, dtype=np.float32)
    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError(f"Expected flat MNIST inputs [N, 784], got {images.shape}")
    images = images.reshape(-1, 28, 28)
    if grid_size <= 0:
        raise ValueError("center icon local grid_size must be positive")
    if not (0.0 < float(crop_fraction) <= 1.0):
        raise ValueError("center_icon_crop_fraction must be in (0, 1]")
    if deviation_weight < 0.0 or edge_weight < 0.0 or contrast_weight < 0.0:
        raise ValueError("center icon local feature weights must be non-negative")
    if deviation_weight + edge_weight + contrast_weight <= 0.0:
        raise ValueError("center icon local feature weights must have a positive sum")

    h, w = images.shape[1:]
    crop_size = int(round(min(h, w) * float(crop_fraction)))
    crop_size = int(round(crop_size / grid_size)) * grid_size
    crop_size = max(grid_size, min(min(h, w), crop_size))
    if crop_size % grid_size != 0:
        crop_size = max(grid_size, (crop_size // grid_size) * grid_size)
    y0 = max(0, (h - crop_size) // 2)
    x0 = max(0, (w - crop_size) // 2)
    y1 = y0 + crop_size
    x1 = x0 + crop_size
    crop = images[:, y0:y1, x0:x1]

    baseline = np.median(crop.reshape(crop.shape[0], -1), axis=1).astype(np.float32).reshape(-1, 1, 1)
    deviation = np.abs(crop - baseline)
    gx = np.zeros_like(crop, dtype=np.float32)
    gy = np.zeros_like(crop, dtype=np.float32)
    gx[:, :, 1:] = np.abs(crop[:, :, 1:] - crop[:, :, :-1])
    gy[:, 1:, :] = np.abs(crop[:, 1:, :] - crop[:, :-1, :])
    edge = gx + gy

    block_h = crop.shape[1] // grid_size
    block_w = crop.shape[2] // grid_size
    crop_blocks = crop.reshape(-1, grid_size, block_h, grid_size, block_w)
    deviation_blocks = deviation.reshape(-1, grid_size, block_h, grid_size, block_w)
    edge_blocks = edge.reshape(-1, grid_size, block_h, grid_size, block_w)
    block_deviation = deviation_blocks.mean(axis=(2, 4))
    block_edge = edge_blocks.mean(axis=(2, 4))
    block_mean = crop_blocks.mean(axis=(2, 4))
    block_sq_mean = (crop_blocks * crop_blocks).mean(axis=(2, 4))
    block_contrast = np.sqrt(np.maximum(block_sq_mean - block_mean * block_mean, 0.0))
    features = (
        float(deviation_weight) * block_deviation
        + float(edge_weight) * block_edge
        + float(contrast_weight) * block_contrast
    )
    return (
        features.reshape(images.shape[0], grid_size * grid_size).astype(np.float32),
        [int(block_h), int(block_w)],
        [int(y0), int(y1), int(x0), int(x1)],
    )


def _make_mnist_center_shape_descriptor_features(
    inputs: np.ndarray,
    *,
    crop_fraction: float,
    mass_weight: float,
    edge_weight: float,
    orientation_weight: float,
    moment_weight: float,
) -> tuple[np.ndarray, list[int]]:
    images = np.asarray(inputs, dtype=np.float32)
    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError(f"Expected flat MNIST inputs [N, 784], got {images.shape}")
    images = images.reshape(-1, 28, 28)
    if not (0.0 < float(crop_fraction) <= 1.0):
        raise ValueError("center_shape_descriptor crop_fraction must be in (0, 1]")
    if mass_weight < 0.0 or edge_weight < 0.0 or orientation_weight < 0.0 or moment_weight < 0.0:
        raise ValueError("center shape descriptor weights must be non-negative")
    if mass_weight + edge_weight + orientation_weight + moment_weight <= 0.0:
        raise ValueError("center shape descriptor weights must have a positive sum")

    h, w = images.shape[1:]
    crop_size = int(round(min(h, w) * float(crop_fraction)))
    crop_size = max(4, min(min(h, w), crop_size))
    y0 = max(0, (h - crop_size) // 2)
    x0 = max(0, (w - crop_size) // 2)
    y1 = y0 + crop_size
    x1 = x0 + crop_size
    crop = images[:, y0:y1, x0:x1]
    n, crop_h, crop_w = crop.shape
    eps = np.float32(1e-6)

    baseline = np.median(crop.reshape(n, -1), axis=1).astype(np.float32).reshape(-1, 1, 1)
    deviation = np.abs(crop - baseline)
    gx = np.zeros_like(crop, dtype=np.float32)
    gy = np.zeros_like(crop, dtype=np.float32)
    gx[:, :, 1:] = np.abs(crop[:, :, 1:] - crop[:, :, :-1])
    gy[:, 1:, :] = np.abs(crop[:, 1:, :] - crop[:, :-1, :])
    edge = gx + gy

    mass = deviation + 0.5 * edge
    mass_flat = mass.reshape(n, -1)
    mass_sum = mass_flat.sum(axis=1, keepdims=True) + eps
    y_coords = np.linspace(0.0, 1.0, crop_h, dtype=np.float32).reshape(1, crop_h, 1)
    x_coords = np.linspace(0.0, 1.0, crop_w, dtype=np.float32).reshape(1, 1, crop_w)
    cy = (mass * y_coords).reshape(n, -1).sum(axis=1) / mass_sum[:, 0]
    cx = (mass * x_coords).reshape(n, -1).sum(axis=1) / mass_sum[:, 0]
    spread_y = np.sqrt(
        np.maximum((mass * (y_coords - cy.reshape(-1, 1, 1)) ** 2).reshape(n, -1).sum(axis=1) / mass_sum[:, 0], 0.0)
    )
    spread_x = np.sqrt(
        np.maximum((mass * (x_coords - cx.reshape(-1, 1, 1)) ** 2).reshape(n, -1).sum(axis=1) / mass_sum[:, 0], 0.0)
    )

    top_mask = y_coords < 0.5
    bottom_mask = ~top_mask
    left_mask = x_coords < 0.5
    right_mask = ~left_mask
    center_mask = (y_coords >= 0.25) & (y_coords <= 0.75) & (x_coords >= 0.25) & (x_coords <= 0.75)
    border_mask = ~center_mask
    diag_main_mask = np.abs(y_coords - x_coords) <= 0.22
    diag_anti_mask = np.abs(y_coords + x_coords - 1.0) <= 0.22

    def _mass_fraction(mask: np.ndarray) -> np.ndarray:
        return (mass * mask.astype(np.float32)).reshape(n, -1).sum(axis=1) / mass_sum[:, 0]

    mass_mean = deviation.reshape(n, -1).mean(axis=1)
    edge_mean = edge.reshape(n, -1).mean(axis=1)
    gx_sum = gx.reshape(n, -1).sum(axis=1)
    gy_sum = gy.reshape(n, -1).sum(axis=1)
    edge_sum = gx_sum + gy_sum + eps
    vertical_edge_ratio = gx_sum / edge_sum
    horizontal_edge_ratio = gy_sum / edge_sum
    solidness = mass_mean / (mass_mean + edge_mean + eps)

    features = np.stack(
        [
            float(mass_weight) * mass_mean,
            float(mass_weight) * solidness,
            float(edge_weight) * edge_mean,
            float(orientation_weight) * horizontal_edge_ratio,
            float(orientation_weight) * vertical_edge_ratio,
            float(moment_weight) * cy,
            float(moment_weight) * cx,
            float(moment_weight) * spread_y,
            float(moment_weight) * spread_x,
            float(mass_weight) * _mass_fraction(top_mask),
            float(mass_weight) * _mass_fraction(bottom_mask),
            float(mass_weight) * _mass_fraction(left_mask),
            float(mass_weight) * _mass_fraction(right_mask),
            float(mass_weight) * _mass_fraction(center_mask),
            float(orientation_weight) * _mass_fraction(diag_main_mask),
            float(orientation_weight) * _mass_fraction(diag_anti_mask),
        ],
        axis=1,
    )
    return features.astype(np.float32), [int(y0), int(y1), int(x0), int(x1)]


def _make_mnist_center_shape_blend_features(
    inputs: np.ndarray,
    *,
    grid_size: int,
    crop_fraction: float,
    deviation_weight: float,
    edge_weight: float,
    contrast_weight: float,
    center_weight: float,
    shape_weight: float,
    shape_mass_weight: float,
    shape_edge_weight: float,
    shape_orientation_weight: float,
    shape_moment_weight: float,
) -> tuple[np.ndarray, list[int], list[int]]:
    if grid_size != 4:
        raise ValueError("center_shape_blend requires avgpool_grid_size=4 so the unified vector is 16-D")
    if center_weight < 0.0 or shape_weight < 0.0:
        raise ValueError("center_shape_blend weights must be non-negative")
    if center_weight + shape_weight <= 0.0:
        raise ValueError("center_shape_blend weights must have a positive sum")

    center_features, block_shape, crop_box = _make_mnist_center_icon_local_features(
        inputs,
        grid_size=grid_size,
        crop_fraction=crop_fraction,
        deviation_weight=deviation_weight,
        edge_weight=edge_weight,
        contrast_weight=contrast_weight,
    )
    shape_features, shape_crop_box = _make_mnist_center_shape_descriptor_features(
        inputs,
        crop_fraction=crop_fraction,
        mass_weight=shape_mass_weight,
        edge_weight=shape_edge_weight,
        orientation_weight=shape_orientation_weight,
        moment_weight=shape_moment_weight,
    )
    if shape_crop_box != crop_box:
        raise ValueError(f"center-shape crop mismatch: center={crop_box}, shape={shape_crop_box}")
    blended = float(center_weight) * center_features + float(shape_weight) * shape_features
    return blended.astype(np.float32), block_shape, crop_box


def _scale_feature_branch(
    train_values: np.ndarray,
    eval_values: np.ndarray,
    *,
    input_low: float,
    input_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_min = train_values.min(axis=0, keepdims=True)
    train_max = train_values.max(axis=0, keepdims=True)
    scale = np.maximum(train_max - train_min, 1e-12)
    train_scaled = input_low + (train_values - train_min) * (input_high - input_low) / scale
    eval_scaled = input_low + (eval_values - train_min) * (input_high - input_low) / scale
    train_scaled = np.clip(train_scaled, input_low, input_high).astype(np.float32)
    eval_scaled = np.clip(eval_scaled, input_low, input_high).astype(np.float32)
    return train_scaled, eval_scaled


def _make_mnist_pca16_avgpool16_dataset(
    *,
    dataset_cfg: Mapping[str, Any],
    num_branch_inputs: int,
    num_outputs: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    Dict[str, Any],
]:
    if num_branch_inputs != 16:
        raise ValueError(
            f"mnist_pca16_avgpool16 requires model input_dim=16 per wavelength, got {num_branch_inputs}"
        )
    if num_outputs != 10:
        raise ValueError(f"mnist_pca16_avgpool16 requires model output_dim=10, got {num_outputs}")

    train_limit = int(dataset_cfg.get("train_limit", 0))
    eval_limit = int(dataset_cfg.get("eval_limit", 0))
    if train_limit <= 0 or eval_limit <= 0:
        raise ValueError("mnist_pca16_avgpool16 requires positive train_limit and eval_limit")

    pca_components = int(dataset_cfg.get("pca_components", num_branch_inputs))
    if pca_components != num_branch_inputs:
        raise ValueError(
            "mnist_pca16_avgpool16 requires "
            f"train.dataset.pca_components == branch input_dim={num_branch_inputs}, got {pca_components}"
        )

    avgpool_grid_size = int(dataset_cfg.get("avgpool_grid_size", 4))
    if avgpool_grid_size * avgpool_grid_size != num_branch_inputs:
        raise ValueError(
            "mnist_pca16_avgpool16 requires avgpool_grid_size^2 == branch input_dim, "
            f"got grid_size={avgpool_grid_size}, branch_input_dim={num_branch_inputs}"
        )

    input_low = float(dataset_cfg.get("input_low", 0.0))
    input_high = float(dataset_cfg.get("input_high", 2.0))
    if input_high <= input_low:
        raise ValueError("train.dataset.input_high must be greater than input_low")

    data_home = get_project_root() / str(dataset_cfg.get("data_home", "data/cache/openml"))
    x_all, y_all = _fetch_mnist_arrays(data_home)
    if train_limit + eval_limit > int(x_all.shape[0]):
        raise ValueError("train_limit + eval_limit exceeds MNIST dataset size")

    x_train_raw, x_eval_raw, y_train_idx, y_eval_idx = train_test_split(
        x_all,
        y_all,
        train_size=train_limit,
        test_size=eval_limit,
        stratify=y_all,
        random_state=seed,
    )

    pca = PCA(
        n_components=pca_components,
        svd_solver=str(dataset_cfg.get("pca_solver", "randomized")),
        whiten=bool(dataset_cfg.get("pca_whiten", False)),
        random_state=seed,
    )
    x_train_pca = pca.fit_transform(x_train_raw).astype(np.float32)
    x_eval_pca = pca.transform(x_eval_raw).astype(np.float32)
    x_train_a, x_eval_a = _scale_feature_branch(
        x_train_pca,
        x_eval_pca,
        input_low=input_low,
        input_high=input_high,
    )

    x_train_pool, pool_block_shape = _make_mnist_avgpool_features(
        x_train_raw,
        grid_size=avgpool_grid_size,
    )
    x_eval_pool, eval_pool_block_shape = _make_mnist_avgpool_features(
        x_eval_raw,
        grid_size=avgpool_grid_size,
    )
    if eval_pool_block_shape != pool_block_shape:
        raise ValueError(
            "Avg-pool block shape mismatch between train and eval: "
            f"train={pool_block_shape}, eval={eval_pool_block_shape}"
        )
    x_train_b, x_eval_b = _scale_feature_branch(
        x_train_pool,
        x_eval_pool,
        input_low=input_low,
        input_high=input_high,
    )

    y_train = np.eye(num_outputs, dtype=np.float32)[y_train_idx]
    y_eval = np.eye(num_outputs, dtype=np.float32)[y_eval_idx]
    explained_variance_ratio = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    explained = float(np.sum(explained_variance_ratio))
    dataset_summary = {
        "component_split": f"pca16_plus_avgpool{avgpool_grid_size}x{avgpool_grid_size}",
        "pca_components": int(pca_components),
        "branch_a_feature_kind": "pca16",
        "branch_b_feature_kind": f"avgpool{avgpool_grid_size}x{avgpool_grid_size}",
        "branch_a_component_indices": list(range(int(pca_components))),
        "branch_b_pool_grid_shape": [avgpool_grid_size, avgpool_grid_size],
        "branch_b_pool_block_shape": pool_block_shape,
        "scaled_input_low": float(input_low),
        "scaled_input_high": float(input_high),
        "branch_a_explained_variance_ratio_sum": explained,
        "branch_a_train_min": float(np.min(x_train_a)),
        "branch_a_train_max": float(np.max(x_train_a)),
        "branch_b_train_min": float(np.min(x_train_b)),
        "branch_b_train_max": float(np.max(x_train_b)),
        "branch_a_train_mean": float(np.mean(x_train_a)),
        "branch_b_train_mean": float(np.mean(x_train_b)),
        "branch_a_train_std": float(np.std(x_train_a)),
        "branch_b_train_std": float(np.std(x_train_b)),
    }
    return x_train_a, x_train_b, x_eval_a, x_eval_b, y_train, y_eval, explained, dataset_summary


def _make_cifar100_hierarchical_subset_dataset(
    *,
    dataset_cfg: Mapping[str, Any],
    num_branch_inputs: int,
    num_outputs: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    Dict[str, Any],
]:
    if num_branch_inputs != 16:
        raise ValueError(
            f"cifar100_hierarchical_subset requires model input_dim=16 per wavelength, got {num_branch_inputs}"
        )

    pca_components = int(dataset_cfg.get("pca_components", num_branch_inputs))
    if pca_components != num_branch_inputs:
        raise ValueError(
            "cifar100_hierarchical_subset requires "
            f"train.dataset.pca_components == branch input_dim={num_branch_inputs}, got {pca_components}"
        )

    avgpool_grid_size = int(dataset_cfg.get("avgpool_grid_size", 4))
    if avgpool_grid_size * avgpool_grid_size != num_branch_inputs:
        raise ValueError(
            "cifar100_hierarchical_subset requires avgpool_grid_size^2 == branch input_dim, "
            f"got grid_size={avgpool_grid_size}, branch_input_dim={num_branch_inputs}"
        )

    input_low = float(dataset_cfg.get("input_low", 0.0))
    input_high = float(dataset_cfg.get("input_high", 2.0))
    if input_high <= input_low:
        raise ValueError("train.dataset.input_high must be greater than input_low")

    data_home = get_project_root() / str(dataset_cfg.get("data_home", "data/cache/cifar100"))
    raw = _load_cifar100_raw_arrays(data_home)
    fine_names = list(raw["fine_names"])
    coarse_names = list(raw["coarse_names"])

    fine_candidates_by_coarse: Dict[int, list[int]] = {}
    for coarse_idx in range(len(coarse_names)):
        fine_idx = sorted(
            set(
                int(value)
                for value in raw["fine_train"][raw["coarse_train"] == coarse_idx].tolist()
            )
        )
        fine_candidates_by_coarse[coarse_idx] = fine_idx

    hierarchy = _resolve_cifar100_hierarchy(
        dataset_cfg=dataset_cfg,
        coarse_names=coarse_names,
        fine_names=fine_names,
        fine_candidates_by_coarse=fine_candidates_by_coarse,
    )
    target_level = str(dataset_cfg.get("target_level", "coarse")).strip().lower()
    if target_level not in {"coarse", "fine"}:
        raise ValueError("train.dataset.target_level must be 'coarse' or 'fine'")

    def _filter_split(
        images: np.ndarray,
        fine_idx: np.ndarray,
        coarse_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keep = np.zeros(fine_idx.shape[0], dtype=bool)
        for selected_coarse in hierarchy["coarse_indices"]:
            selected_fine = np.asarray(
                hierarchy["fine_indices_by_coarse"][selected_coarse],
                dtype=np.int64,
            )
            keep |= (coarse_idx == selected_coarse) & np.isin(fine_idx, selected_fine)
        return images[keep], fine_idx[keep], coarse_idx[keep]

    x_train_raw, fine_train, coarse_train = _filter_split(
        raw["x_train"],
        raw["fine_train"],
        raw["coarse_train"],
    )
    x_eval_raw, fine_eval, coarse_eval = _filter_split(
        raw["x_eval"],
        raw["fine_eval"],
        raw["coarse_eval"],
    )

    if target_level == "coarse":
        active_target_names = list(hierarchy["coarse_names"])
        coarse_to_local = {
            coarse_idx: local_idx
            for local_idx, coarse_idx in enumerate(hierarchy["coarse_indices"])
        }
        target_train_idx = np.asarray(
            [coarse_to_local[int(value)] for value in coarse_train.tolist()],
            dtype=np.int64,
        )
        target_eval_idx = np.asarray(
            [coarse_to_local[int(value)] for value in coarse_eval.tolist()],
            dtype=np.int64,
        )
    else:
        active_target_names = list(hierarchy["fine_display_flat"])
        fine_to_local: Dict[int, int] = {}
        ordered_fine: list[int] = []
        cursor = 0
        for coarse_idx in hierarchy["coarse_indices"]:
            for fine_idx in hierarchy["fine_indices_by_coarse"][coarse_idx]:
                fine_to_local[int(fine_idx)] = cursor
                ordered_fine.append(int(fine_idx))
                cursor += 1
        target_train_idx = np.asarray(
            [fine_to_local[int(value)] for value in fine_train.tolist()],
            dtype=np.int64,
        )
        target_eval_idx = np.asarray(
            [fine_to_local[int(value)] for value in fine_eval.tolist()],
            dtype=np.int64,
        )

    active_class_count = len(active_target_names)
    if active_class_count > num_outputs:
        raise ValueError(
            f"Selected CIFAR-100 subset exposes {active_class_count} target classes under target_level={target_level!r}, "
            f"but model output_dim={num_outputs}. Consider target_level='coarse' or rebuilding the map with more outputs."
        )

    train_limit_raw = dataset_cfg.get("train_limit")
    eval_limit_raw = dataset_cfg.get("eval_limit")
    train_limit = None if train_limit_raw is None else int(train_limit_raw)
    eval_limit = None if eval_limit_raw is None else int(eval_limit_raw)

    train_pick = _subsample_split_indices(target_train_idx, limit=train_limit, seed=seed)
    eval_pick = _subsample_split_indices(target_eval_idx, limit=eval_limit, seed=seed + 1)
    x_train_raw = x_train_raw[train_pick]
    x_eval_raw = x_eval_raw[eval_pick]
    target_train_idx = target_train_idx[train_pick]
    target_eval_idx = target_eval_idx[eval_pick]
    fine_train = fine_train[train_pick]
    fine_eval = fine_eval[eval_pick]
    coarse_train = coarse_train[train_pick]
    coarse_eval = coarse_eval[eval_pick]

    x_train_flat = x_train_raw.reshape(x_train_raw.shape[0], -1).astype(np.float32)
    x_eval_flat = x_eval_raw.reshape(x_eval_raw.shape[0], -1).astype(np.float32)
    pca = PCA(
        n_components=pca_components,
        svd_solver=str(dataset_cfg.get("pca_solver", "randomized")),
        whiten=bool(dataset_cfg.get("pca_whiten", False)),
        random_state=seed,
    )
    x_train_pca = pca.fit_transform(x_train_flat).astype(np.float32)
    x_eval_pca = pca.transform(x_eval_flat).astype(np.float32)
    x_train_a, x_eval_a = _scale_feature_branch(
        x_train_pca,
        x_eval_pca,
        input_low=input_low,
        input_high=input_high,
    )

    x_train_b, x_eval_b, branch_b_metadata = _make_cifar_branch_b_features(
        x_train_raw,
        x_eval_raw,
        dataset_cfg=dataset_cfg,
        num_branch_inputs=num_branch_inputs,
        pca_components=pca_components,
        avgpool_grid_size=avgpool_grid_size,
        input_low=input_low,
        input_high=input_high,
        seed=seed + 1540,
    )

    y_train = _one_hot_with_output_dim(target_train_idx, output_dim=num_outputs)
    y_eval = _one_hot_with_output_dim(target_eval_idx, output_dim=num_outputs)
    explained_variance_ratio = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    explained = float(np.sum(explained_variance_ratio))
    dataset_summary = {
        "dataset_name": "cifar100_hierarchical_subset",
        "target_level": target_level,
        "active_class_count": int(active_class_count),
        "output_dim": int(num_outputs),
        "class_names_active": list(active_target_names),
        "class_names_padded": _pad_class_names(active_target_names, output_dim=num_outputs),
        "selected_coarse_names": list(hierarchy["coarse_names"]),
        "selected_fine_names_by_coarse": dict(hierarchy["fine_names_by_coarse"]),
        "fine_per_coarse": int(hierarchy["fine_per_coarse"]),
        "branch_a_feature_kind": "pca16_rgb_flat",
        "branch_b_feature_kind": str(branch_b_metadata["feature_kind"]),
        "pca_components": int(pca_components),
        "branch_a_explained_variance_ratio_sum": explained,
        "scaled_input_low": float(input_low),
        "scaled_input_high": float(input_high),
        "train_samples": int(x_train_a.shape[0]),
        "eval_samples": int(x_eval_a.shape[0]),
        "train_limit_applied": None if train_limit is None else int(train_limit),
        "eval_limit_applied": None if eval_limit is None else int(eval_limit),
        "train_unique_coarse_count": int(len(set(int(value) for value in coarse_train.tolist()))),
        "eval_unique_coarse_count": int(len(set(int(value) for value in coarse_eval.tolist()))),
        "train_unique_fine_count": int(len(set(int(value) for value in fine_train.tolist()))),
        "eval_unique_fine_count": int(len(set(int(value) for value in fine_eval.tolist()))),
        "branch_a_train_min": float(np.min(x_train_a)),
        "branch_a_train_max": float(np.max(x_train_a)),
        "branch_b_train_min": float(np.min(x_train_b)),
        "branch_b_train_max": float(np.max(x_train_b)),
        "branch_a_train_mean": float(np.mean(x_train_a)),
        "branch_b_train_mean": float(np.mean(x_train_b)),
        "branch_a_train_std": float(np.std(x_train_a)),
        "branch_b_train_std": float(np.std(x_train_b)),
    }
    if "pool_grid_shape" in branch_b_metadata:
        dataset_summary["branch_b_pool_grid_shape"] = list(branch_b_metadata["pool_grid_shape"])
    if "pool_block_shape" in branch_b_metadata:
        dataset_summary["branch_b_pool_block_shape"] = list(branch_b_metadata["pool_block_shape"])
    if "residual_grid_shape" in branch_b_metadata:
        dataset_summary["branch_b_residual_grid_shape"] = list(branch_b_metadata["residual_grid_shape"])
    if "residual_block_shape" in branch_b_metadata:
        dataset_summary["branch_b_residual_block_shape"] = list(branch_b_metadata["residual_block_shape"])
    if "lowfreq_grid_shape" in branch_b_metadata:
        dataset_summary["branch_b_lowfreq_grid_shape"] = list(branch_b_metadata["lowfreq_grid_shape"])
    if "lowfreq_block_shape" in branch_b_metadata:
        dataset_summary["branch_b_lowfreq_block_shape"] = list(branch_b_metadata["lowfreq_block_shape"])
    if "explained_variance_ratio_sum" in branch_b_metadata:
        dataset_summary["branch_b_explained_variance_ratio_sum"] = float(
            branch_b_metadata["explained_variance_ratio_sum"]
        )
    return x_train_a, x_train_b, x_eval_a, x_eval_b, y_train, y_eval, explained, dataset_summary


def _make_cifar100_hierarchical_subset_single_dataset(
    *,
    dataset_cfg: Mapping[str, Any],
    num_inputs: int,
    num_outputs: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    Dict[str, Any],
]:
    feature_branch = str(dataset_cfg.get("feature_branch", "a")).strip().lower()
    if feature_branch not in {"a", "b"}:
        raise ValueError("train.dataset.feature_branch must be 'a' or 'b'")

    x_train_a, x_train_b, x_eval_a, x_eval_b, y_train, y_eval, explained, dataset_summary = _make_cifar100_hierarchical_subset_dataset(
        dataset_cfg=dataset_cfg,
        num_branch_inputs=num_inputs,
        num_outputs=num_outputs,
        seed=seed,
    )
    if feature_branch == "a":
        x_train = x_train_a
        x_eval = x_eval_a
        feature_kind = str(dataset_summary["branch_a_feature_kind"])
        explained_single = float(dataset_summary.get("branch_a_explained_variance_ratio_sum", explained))
        input_min = float(dataset_summary["branch_a_train_min"])
        input_max = float(dataset_summary["branch_a_train_max"])
        input_mean = float(dataset_summary["branch_a_train_mean"])
        input_std = float(dataset_summary["branch_a_train_std"])
    else:
        x_train = x_train_b
        x_eval = x_eval_b
        feature_kind = str(dataset_summary["branch_b_feature_kind"])
        explained_single = 0.0
        input_min = float(dataset_summary["branch_b_train_min"])
        input_max = float(dataset_summary["branch_b_train_max"])
        input_mean = float(dataset_summary["branch_b_train_mean"])
        input_std = float(dataset_summary["branch_b_train_std"])

    dataset_summary = dict(dataset_summary)
    dataset_summary.update(
        {
            "dataset_name": "cifar100_hierarchical_subset_single",
            "feature_branch": feature_branch,
            "single_feature_kind": feature_kind,
            "input_train_min": input_min,
            "input_train_max": input_max,
            "input_train_mean": input_mean,
            "input_train_std": input_std,
        }
    )
    return x_train, x_eval, y_train, y_eval, explained_single, dataset_summary


def _coerce_branch_output(raw: Any, *, device: torch.device) -> torch.Tensor:
    out = torch.as_tensor(raw, dtype=torch.float32, device=device)
    if out.ndim == 1:
        out = out.unsqueeze(0)
    if out.ndim != 2:
        raise ValueError(f"Expected single-wavelength output [batch, port], got shape {tuple(out.shape)}")
    return out


def _class_indices_from_one_hot(target: torch.Tensor) -> torch.Tensor:
    if target.ndim != 2:
        raise ValueError(f"classification target must be 2D [N, C], got shape {tuple(target.shape)}")
    return target.argmax(dim=-1)


def _confusion_matrix(pred_idx: torch.Tensor, target_idx: torch.Tensor, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    pred_np = pred_idx.detach().cpu().numpy().astype(np.int64)
    target_np = target_idx.detach().cpu().numpy().astype(np.int64)
    for truth, pred in zip(target_np.tolist(), pred_np.tolist()):
        matrix[truth, pred] += 1
    return matrix


def _resolve_task_wavelengths(
    model: torch.nn.Module,
    task_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    requested = task_cfg.get("wavelengths_um")
    if not isinstance(requested, Sequence) or len(requested) != 2:
        raise ValueError("train.task.wavelengths_um must contain exactly two wavelengths")
    available = np.asarray(
        getattr(model, "available_wavelengths_um", [getattr(model, "selected_wavelength_um", 1.55)]),
        dtype=float,
    ).reshape(-1)
    resolved = [
        resolve_wavelength_choice(available, wavelength_um=float(value), selection="nearest")
        for value in requested
    ]
    wavelengths_um = [float(item["selected_wavelength_um"]) for item in resolved]
    return {
        "wavelengths_um": wavelengths_um,
        "wavelength_keys": [canonical_wavelength_key(value) for value in wavelengths_um],
    }


def _resolve_single_task_wavelength(
    model: torch.nn.Module,
    task_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    requested = task_cfg.get("wavelength_um")
    if requested is None:
        requested_list = task_cfg.get("wavelengths_um")
        if isinstance(requested_list, Sequence) and not isinstance(requested_list, (str, bytes)) and len(requested_list) == 1:
            requested = requested_list[0]
    if requested is None:
        raise ValueError("single-wavelength training requires train.task.wavelength_um or a single-item wavelengths_um")
    available = np.asarray(
        getattr(model, "available_wavelengths_um", [getattr(model, "selected_wavelength_um", 1.55)]),
        dtype=float,
    ).reshape(-1)
    resolved = resolve_wavelength_choice(available, wavelength_um=float(requested), selection="nearest")
    wavelength_um = float(resolved["selected_wavelength_um"])
    return {
        "wavelength_um": wavelength_um,
        "wavelength_key": canonical_wavelength_key(wavelength_um),
    }


def _forward_two_wavelengths(
    model: torch.nn.Module,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    *,
    wavelength_a_um: float,
    wavelength_b_um: float,
    objective_cfg: Mapping[str, Any],
    device: torch.device,
    dual_hierarchy: Mapping[str, Any] | None = None,
    capture_fields: bool = False,
) -> Dict[str, Any]:
    raw_a = _coerce_branch_output(
        model(x_a, wavelength_um=wavelength_a_um),
        device=device,
    )
    fields_a = None
    if capture_fields:
        fields_a = [torch.as_tensor(field).detach().cpu() for field in getattr(model, "us_H", [])]

    raw_b = _coerce_branch_output(
        model(x_b, wavelength_um=wavelength_b_um),
        device=device,
    )
    fields_b = None
    if capture_fields:
        fields_b = [torch.as_tensor(field).detach().cpu() for field in getattr(model, "us_H", [])]

    pred_a_ports = _apply_output_transform(raw_a, objective_cfg)
    pred_b_ports = _apply_output_transform(raw_b, objective_cfg)
    if dual_hierarchy is not None:
        pred_a_active = _active_port_distribution(pred_a_ports, dual_hierarchy)
        pred_a = _sum_group_ranges(pred_a_active, dual_hierarchy)
        pred_b_grouped = _extract_branch_b_local_distribution(pred_b_ports, dual_hierarchy)
        pred_sum = _compose_hierarchical_fine_distribution(pred_a, pred_b_grouped, dual_hierarchy)
        if bool(dual_hierarchy.get("branch_b_uses_coarse_context", False)):
            pred_b = _contextualize_branch_b_local_distribution(pred_b_grouped, pred_a, dual_hierarchy)
        else:
            pred_b = pred_b_grouped
        raw_a_display = pred_a
        raw_b_display = pred_b
        raw_sum = pred_sum
        raw_a_flatten = raw_a_display
        raw_b_flatten = raw_b_display
        pred_a_flatten = pred_a
        pred_b_flatten = pred_b
    else:
        raw_sum = raw_a + raw_b
        pred_a = pred_a_ports
        pred_b = pred_b_ports
        pred_sum = _apply_output_transform(raw_sum, objective_cfg)
        raw_a_display = raw_a
        raw_b_display = raw_b
        raw_a_flatten = raw_a
        raw_b_flatten = raw_b
        pred_a_flatten = pred_a_ports
        pred_b_flatten = pred_b_ports
    return {
        "raw_a": raw_a_display,
        "raw_b": raw_b_display,
        "raw_sum": raw_sum,
        "pred_a": pred_a,
        "pred_b": pred_b,
        "pred_sum": pred_sum,
        "pred_b_grouped": (pred_b_grouped if dual_hierarchy is not None else pred_b),
        "raw_a_flatten": raw_a_flatten,
        "raw_b_flatten": raw_b_flatten,
        "pred_a_flatten": pred_a_flatten,
        "pred_b_flatten": pred_b_flatten,
        "fields_a": fields_a,
        "fields_b": fields_b,
    }


def _forward_single_wavelength(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    wavelength_um: float,
    objective_cfg: Mapping[str, Any],
    device: torch.device,
    capture_fields: bool = False,
) -> Dict[str, Any]:
    raw = _coerce_branch_output(
        model(x, wavelength_um=wavelength_um),
        device=device,
    )
    fields = None
    if capture_fields:
        fields = [torch.as_tensor(field).detach().cpu() for field in getattr(model, "us_H", [])]
    pred = _apply_output_transform(raw, objective_cfg)
    return {
        "raw": raw,
        "pred": pred,
        "fields": fields,
    }


def _compute_losses(
    outputs: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    *,
    loss_fn,
    branch_weights: torch.Tensor,
    combined_weight: float,
    dual_hierarchy: Mapping[str, Any] | None = None,
    include_combined_in_total_loss: bool = True,
) -> Dict[str, torch.Tensor]:
    if dual_hierarchy is None:
        loss_a = loss_fn(outputs["pred_a"], target)
        loss_b = loss_fn(outputs["pred_b"], target)
    else:
        hierarchy_targets = _extract_dual_hierarchy_targets(target, dual_hierarchy)
        pred_b_local, branch_b_mask = _resolve_branch_b_task_predictions(
            outputs,
            hierarchy_targets,
            dual_hierarchy,
        )
        loss_a = loss_fn(outputs["pred_a"], hierarchy_targets["coarse_target"])
        loss_b = _masked_loss_mean(loss_fn, pred_b_local, hierarchy_targets["local_target"], branch_b_mask)
    loss_sum = loss_fn(outputs["pred_sum"], target)
    if include_combined_in_total_loss:
        numerator = (
            branch_weights[0] * loss_a
            + branch_weights[1] * loss_b
            + float(combined_weight) * loss_sum
        )
        denominator = branch_weights.sum() + float(combined_weight)
    else:
        numerator = branch_weights[0] * loss_a + branch_weights[1] * loss_b
        denominator = branch_weights.sum()
    total = numerator / denominator.clamp_min(1e-12)
    return {
        "loss_a": loss_a,
        "loss_b": loss_b,
        "loss_sum": loss_sum,
        "loss_total": total,
    }


def _interp_complex_along_axis(
    values: np.ndarray,
    old_axis: np.ndarray,
    new_axis: np.ndarray,
    *,
    axis: int,
) -> np.ndarray:
    arr = np.asarray(values)
    moved = np.moveaxis(arr, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    out = np.empty((new_axis.size, flat.shape[1]), dtype=np.complex128)
    for col in range(flat.shape[1]):
        real = np.interp(new_axis, old_axis, np.real(flat[:, col]))
        imag = np.interp(new_axis, old_axis, np.imag(flat[:, col]))
        out[:, col] = real + 1j * imag
    out = out.reshape((new_axis.size,) + moved.shape[1:])
    return np.moveaxis(out, 0, axis)


def _downsample_monitor_payload(payload: Mapping[str, Any], *, target_samples: int) -> Dict[str, Any]:
    out = copy.deepcopy(dict(payload))
    y_um = np.asarray(out.get("y_um", []), dtype=float).reshape(-1)
    if target_samples <= 0 or y_um.size <= target_samples:
        return out
    new_y_um = np.linspace(float(y_um.min()), float(y_um.max()), int(target_samples), dtype=float)
    e = np.asarray(out["E"])
    try:
        axis = [idx for idx, size in enumerate(e.shape) if size == y_um.size][-1]
    except IndexError as exc:
        raise ValueError(
            f"Cannot locate y axis of size {y_um.size} in monitor E shape {e.shape}"
        ) from exc
    out["E"] = _interp_complex_along_axis(e, y_um, new_y_um, axis=axis)
    out["y_um"] = new_y_um
    return out


def _build_training_package(
    mapped_network_package: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    package = copy.deepcopy(dict(mapped_network_package))

    local_shape_cfg = train_cfg.get("shape")
    local_mapping_cfg = train_cfg.get("mapping")
    if local_shape_cfg is None and local_mapping_cfg is None:
        return package
    raise ValueError(
        "This recovered MNIST bundle does not allow train.shape or train.mapping overrides. "
        "Changing physical geometry or mapping settings requires regenerating the shared map/ "
        "package instead of mutating it inside train."
    )


def _downsample_training_package(
    mapped_network_package: Mapping[str, Any],
    *,
    target_samples: int | None,
) -> Dict[str, Any]:
    package = copy.deepcopy(dict(mapped_network_package))
    if target_samples is None or target_samples <= 0:
        return package

    calibration = dict(package.get("calibration_package") or {})
    monitors = calibration.get("monitors") or []
    calibration["monitors"] = [
        _downsample_monitor_payload(monitor, target_samples=int(target_samples))
        if isinstance(monitor, Mapping)
        else np.asarray(monitor)
        for monitor in monitors
    ]
    package["calibration_package"] = calibration
    return package


def _apply_slot_length_constraints(model: torch.nn.Module, constraints_cfg: Mapping[str, Any]) -> None:
    low = constraints_cfg.get("slot_length_min_um")
    high = constraints_cfg.get("slot_length_max_um")
    if low is None and high is None:
        return
    low_value = None if low is None else float(low)
    high_value = None if high is None else float(high)
    if low_value is not None and high_value is not None and high_value < low_value:
        raise ValueError(
            "train.constraints.slot_length_max_um must be >= slot_length_min_um, "
            f"got {high_value} < {low_value}"
        )
    for param in getattr(model, "slot_length_um", []):
        if not getattr(param, "requires_grad", False):
            continue
        if low_value is not None and high_value is not None:
            param.data.clamp_(min=low_value, max=high_value)
        elif low_value is not None:
            param.data.clamp_(min=low_value)
        else:
            param.data.clamp_(max=high_value)


def _resolve_exploration_state(exploration_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    enabled = bool(exploration_cfg.get("enabled", False))
    metric = str(exploration_cfg.get("metric", "eval_accuracy_combined")).strip()
    goal = str(exploration_cfg.get("goal", "max")).strip().lower()
    if goal not in {"min", "max"}:
        raise ValueError(f"train.exploration.goal must be 'min' or 'max', got {goal!r}")

    plateau_patience = int(exploration_cfg.get("plateau_patience", 0))
    warmup_epochs = int(exploration_cfg.get("warmup_epochs", 0))
    cooldown_epochs = int(exploration_cfg.get("cooldown_epochs", 0))
    jitter_std_um = float(exploration_cfg.get("jitter_std_um", 0.0))
    jitter_decay = float(exploration_cfg.get("jitter_decay", 1.0))
    min_delta = float(exploration_cfg.get("min_delta", 0.0))
    max_events = int(exploration_cfg.get("max_events", 0))

    if enabled:
        if not metric:
            raise ValueError("train.exploration.metric must be a non-empty string when exploration is enabled")
        if plateau_patience <= 0:
            raise ValueError("train.exploration.plateau_patience must be > 0 when exploration is enabled")
        if warmup_epochs < 0:
            raise ValueError("train.exploration.warmup_epochs must be >= 0")
        if cooldown_epochs < 0:
            raise ValueError("train.exploration.cooldown_epochs must be >= 0")
        if jitter_std_um <= 0.0:
            raise ValueError("train.exploration.jitter_std_um must be > 0 when exploration is enabled")
        if jitter_decay <= 0.0:
            raise ValueError("train.exploration.jitter_decay must be > 0")
        if min_delta < 0.0:
            raise ValueError("train.exploration.min_delta must be >= 0")

    return {
        "enabled": enabled,
        "metric": metric,
        "goal": goal,
        "plateau_patience": plateau_patience,
        "warmup_epochs": warmup_epochs,
        "cooldown_epochs": cooldown_epochs,
        "jitter_std_um": jitter_std_um,
        "jitter_decay": jitter_decay,
        "min_delta": min_delta,
        "max_events": max_events,
        "best_value": None,
        "best_epoch": 0,
        "bad_epochs": 0,
        "events_applied": 0,
        "cooldown_until_epoch": 0,
        "pending_jitter_std_um": None,
        "pending_trigger_epoch": 0,
        "last_applied_epoch": 0,
        "last_applied_std_um": 0.0,
    }


def _metric_improved(
    value: float,
    best_value: float | None,
    *,
    goal: str,
    min_delta: float,
) -> bool:
    if best_value is None:
        return True
    if goal == "min":
        return value < best_value - min_delta
    return value > best_value + min_delta


def _apply_slot_length_jitter(model: torch.nn.Module, *, std_um: float) -> int:
    if std_um <= 0.0:
        return 0
    updated = 0
    with torch.no_grad():
        for param in getattr(model, "slot_length_um", []):
            if not getattr(param, "requires_grad", False):
                continue
            if int(param.numel()) <= 0:
                continue
            param.add_(torch.randn_like(param) * float(std_um))
            updated += int(param.numel())
    return updated


def _maybe_apply_pending_exploration_jitter(
    state: Mapping[str, Any],
    *,
    epoch: int,
) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {"exploration_jitter_applied": False}
    exploration = state.get("exploration_state")
    if not isinstance(exploration, dict) or not bool(exploration.get("enabled", False)):
        return {"exploration_jitter_applied": False}

    pending_std_um = exploration.get("pending_jitter_std_um")
    if pending_std_um is None:
        return {
            "exploration_jitter_applied": False,
            "exploration_events_applied": int(exploration.get("events_applied", 0)),
        }

    std_um = float(pending_std_um)
    param_count = _apply_slot_length_jitter(state["model"], std_um=std_um)
    _apply_slot_length_constraints(state["model"], state["constraints_cfg"])
    exploration["pending_jitter_std_um"] = None
    exploration["pending_trigger_epoch"] = 0
    exploration["events_applied"] = int(exploration.get("events_applied", 0)) + 1
    exploration["last_applied_epoch"] = int(epoch)
    exploration["last_applied_std_um"] = std_um
    print(
        "exploration_jitter_applied "
        f"epoch={epoch:04d} std_um={std_um:.6f} updated_params={param_count}"
    )
    return {
        "exploration_jitter_applied": True,
        "exploration_jitter_std_um": std_um,
        "exploration_jitter_updated_params": int(param_count),
        "exploration_events_applied": int(exploration["events_applied"]),
    }


def _update_exploration_state_after_eval(
    state: Mapping[str, Any],
    *,
    epoch: int,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return metrics
    exploration = state.get("exploration_state")
    if not isinstance(exploration, dict) or not bool(exploration.get("enabled", False)):
        return metrics

    metric_name = str(exploration["metric"])
    if metric_name not in metrics:
        raise KeyError(
            f"train.exploration.metric={metric_name!r} was not found in eval metrics: {sorted(metrics.keys())}"
        )

    metric_value = float(metrics[metric_name])
    best_value = exploration.get("best_value")
    improved = _metric_improved(
        metric_value,
        None if best_value is None else float(best_value),
        goal=str(exploration["goal"]),
        min_delta=float(exploration["min_delta"]),
    )
    if improved:
        exploration["best_value"] = metric_value
        exploration["best_epoch"] = int(epoch)
        exploration["bad_epochs"] = 0
    else:
        exploration["bad_epochs"] = int(exploration.get("bad_epochs", 0)) + 1

    metrics["exploration_metric_name"] = metric_name
    metrics["exploration_metric_value"] = metric_value
    metrics["exploration_bad_epochs"] = int(exploration["bad_epochs"])
    metrics["exploration_best_epoch"] = int(exploration.get("best_epoch", 0))
    metrics["exploration_best_metric_value"] = (
        metric_value if exploration.get("best_value") is None else float(exploration["best_value"])
    )
    metrics["exploration_events_applied"] = int(exploration.get("events_applied", 0))
    metrics["exploration_jitter_queued"] = False

    if exploration.get("pending_jitter_std_um") is not None:
        metrics["exploration_pending_jitter_std_um"] = float(exploration["pending_jitter_std_um"])
        return metrics

    warmup_epochs = int(exploration["warmup_epochs"])
    plateau_patience = int(exploration["plateau_patience"])
    cooldown_until_epoch = int(exploration.get("cooldown_until_epoch", 0))
    max_events = int(exploration.get("max_events", 0))
    events_applied = int(exploration.get("events_applied", 0))

    if improved or epoch < warmup_epochs or int(exploration["bad_epochs"]) < plateau_patience:
        return metrics
    if epoch <= cooldown_until_epoch:
        return metrics
    if max_events > 0 and events_applied >= max_events:
        return metrics

    jitter_std_um = float(exploration["jitter_std_um"]) * (
        float(exploration["jitter_decay"]) ** float(events_applied)
    )
    if jitter_std_um <= 0.0:
        return metrics

    exploration["pending_jitter_std_um"] = jitter_std_um
    exploration["pending_trigger_epoch"] = int(epoch)
    exploration["cooldown_until_epoch"] = int(epoch) + int(exploration["cooldown_epochs"])
    exploration["bad_epochs"] = 0
    metrics["exploration_jitter_queued"] = True
    metrics["exploration_pending_jitter_std_um"] = jitter_std_um
    print(
        "exploration_jitter_queued "
        f"epoch={epoch:04d} metric={metric_name} value={metric_value:.6f} std_um={jitter_std_um:.6f}"
    )
    return metrics


def _accuracy_from_predictions(pred: torch.Tensor, target: torch.Tensor) -> int:
    return int((pred.argmax(dim=-1) == target.argmax(dim=-1)).sum().item())


def _flatten_reference_branch_predictions(outputs: Mapping[str, torch.Tensor]) -> Dict[str, list[float]]:
    return {
        "raw_output": torch.cat(
            (
                outputs.get("raw_a_flatten", outputs["raw_a"]),
                outputs.get("raw_b_flatten", outputs["raw_b"]),
            ),
            dim=-1,
        ).detach().cpu().reshape(-1).tolist(),
        "transformed_output": torch.cat(
            (
                outputs.get("pred_a_flatten", outputs["pred_a"]),
                outputs.get("pred_b_flatten", outputs["pred_b"]),
            ),
            dim=-1,
        ).detach().cpu().reshape(-1).tolist(),
    }


def _resolve_dual_selection_metric(
    *,
    mode: str,
    eval_accuracy_a: float,
    eval_accuracy_b: float,
    eval_accuracy_sum: float,
) -> float:
    mode_key = str(mode).strip().lower()
    if mode_key in {"", "combined", "eval_accuracy_combined"}:
        return -float(eval_accuracy_sum)
    if mode_key in {"branch_mean", "mean", "branches"}:
        return -0.5 * (float(eval_accuracy_a) + float(eval_accuracy_b))
    if mode_key in {"branch_a", "a", "1.56", "eval_accuracy_1.56"}:
        return -float(eval_accuracy_a)
    if mode_key in {"branch_b", "b", "1.54", "eval_accuracy_1.54"}:
        return -float(eval_accuracy_b)
    raise ValueError(
        "train.task.selection_metric_mode must be one of "
        "{'combined', 'branch_mean', 'branch_a', 'branch_b'}"
    )


def _make_shuffled_train_loader(
    dataset: TensorDataset,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def _make_weighted_train_loader(
    dataset: TensorDataset,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
    sample_weights: np.ndarray,
) -> DataLoader:
    weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if weights.ndim != 1 or weights.size != len(dataset):
        raise ValueError(
            "sample_weights must be a 1D array with one entry per training sample"
        )
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=int(weights.size),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def _resolve_dual_phase_schedule(
    train_cfg: Mapping[str, Any],
    *,
    loop_epochs: int,
    branch_loss_weights: np.ndarray,
    combined_loss_weight: float,
    include_combined_in_total_loss: bool,
    selection_metric_mode: str,
) -> list[Dict[str, Any]] | None:
    phases_cfg = train_cfg.get("phases")
    if phases_cfg is None:
        return None
    if (
        not isinstance(phases_cfg, Sequence)
        or isinstance(phases_cfg, (str, bytes))
        or len(phases_cfg) == 0
    ):
        raise ValueError("train.phases must be a non-empty list when provided")

    schedule: list[Dict[str, Any]] = []
    cursor = 1
    for idx, raw_phase in enumerate(phases_cfg):
        if not isinstance(raw_phase, Mapping):
            raise ValueError(
                f"train.phases[{idx}] must be a mapping, got {type(raw_phase).__name__}"
            )
        name = str(raw_phase.get("name", f"phase_{idx + 1}")).strip() or f"phase_{idx + 1}"
        epochs = int(raw_phase.get("epochs", 0))
        if epochs <= 0:
            raise ValueError(f"train.phases[{idx}].epochs must be > 0")
        phase_branch_loss_weights = np.asarray(
            raw_phase.get("branch_loss_weights", branch_loss_weights.tolist()),
            dtype=np.float32,
        ).reshape(-1)
        if phase_branch_loss_weights.shape != (2,):
            raise ValueError(
                f"train.phases[{idx}].branch_loss_weights must contain exactly two values"
            )
        phase_combined_loss_weight = float(
            raw_phase.get("combined_loss_weight", combined_loss_weight)
        )
        if phase_combined_loss_weight < 0.0:
            raise ValueError(
                f"train.phases[{idx}].combined_loss_weight must be >= 0"
            )
        phase_oversample_factor = float(
            raw_phase.get("branch_b_focus_oversample_factor", 1.0)
        )
        if phase_oversample_factor < 1.0:
            raise ValueError(
                f"train.phases[{idx}].branch_b_focus_oversample_factor must be >= 1"
            )
        schedule.append(
            {
                "name": name,
                "start_epoch": int(cursor),
                "end_epoch": int(cursor + epochs - 1),
                "branch_loss_weights": phase_branch_loss_weights.tolist(),
                "combined_loss_weight": phase_combined_loss_weight,
                "include_combined_in_total_loss": bool(
                    raw_phase.get(
                        "include_combined_in_total_loss",
                        include_combined_in_total_loss,
                    )
                ),
                "selection_metric_mode": str(
                    raw_phase.get("selection_metric_mode", selection_metric_mode)
                ),
                "branch_b_focus_oversample_factor": phase_oversample_factor,
            }
        )
        cursor += epochs
    if cursor - 1 != int(loop_epochs):
        raise ValueError(
            "train.phases epoch counts must sum to train.loop.epochs, "
            f"got {cursor - 1} vs {int(loop_epochs)}"
        )
    return schedule


def _resolve_active_dual_phase(
    state: Mapping[str, Any],
    epoch: int,
) -> Dict[str, Any] | None:
    phase_schedule = state.get("phase_schedule") or []
    for phase in phase_schedule:
        if int(phase["start_epoch"]) <= int(epoch) <= int(phase["end_epoch"]):
            return dict(phase)
    return None


def _resolve_active_dual_training_settings(
    state: Mapping[str, Any],
    epoch: int,
) -> Dict[str, Any]:
    phase = _resolve_active_dual_phase(state, epoch)
    if phase is None:
        return {
            "phase_name": "default",
            "phase_epoch": int(epoch),
            "branch_loss_weights": state["branch_loss_weights"],
            "combined_loss_weight": float(state["combined_loss_weight"]),
            "include_combined_in_total_loss": bool(
                state.get("include_combined_in_total_loss", True)
            ),
            "selection_metric_mode": str(state.get("selection_metric_mode", "combined")),
            "train_loader": state["train_loader"],
        }
    start_epoch = int(phase["start_epoch"])
    loader_key = str(phase.get("train_loader_key", "default"))
    return {
        "phase_name": str(phase["name"]),
        "phase_epoch": int(epoch - start_epoch + 1),
        "branch_loss_weights": torch.tensor(
            phase["branch_loss_weights"],
            dtype=torch.float32,
            device=state["device"],
        ),
        "combined_loss_weight": float(phase["combined_loss_weight"]),
        "include_combined_in_total_loss": bool(phase["include_combined_in_total_loss"]),
        "selection_metric_mode": str(phase["selection_metric_mode"]),
        "train_loader": state["train_loaders_by_key"][loader_key],
    }


def build_training_state(
    mapped_network_package: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    *,
    train_dir: str | Path,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_dir = Path(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = dict(train_cfg.get("model") or {})
    dataset_cfg = dict(train_cfg.get("dataset") or {})
    task_cfg = dict(train_cfg.get("task") or {})
    objective_cfg = dict(train_cfg.get("objective") or {})
    optimizer_cfg = dict(train_cfg.get("optimizer") or {})
    loop_cfg = dict(train_cfg.get("loop") or {})
    constraints_cfg = dict(train_cfg.get("constraints") or {})
    exploration_cfg = dict(train_cfg.get("exploration") or {})

    if str(task_cfg.get("kind", "")) != "supervised_dataset":
        raise ValueError("mnist32_dualwavelength_localmap_w500 only supports train.task.kind='supervised_dataset'")
    dataset_kind = str(dataset_cfg.get("kind", ""))
    if dataset_kind not in {
        "mnist_pca32_split16",
        "mnist_pca_split16",
        "mnist_pca16_single",
        "mnist_pca16_avgpool16",
        "fashion_mnist_subset_single",
        "fashion_mnist_subclass_composite_single",
        "fashion_mnist_subclass_composite_dual",
        "imagenet_scene_subset_single",
        "cifar100_hierarchical_subset",
        "cifar100_hierarchical_subset_single",
    }:
        raise ValueError(
            "mnist32_dualwavelength_localmap_w500 only supports "
            "train.dataset.kind in {'mnist_pca32_split16', 'mnist_pca_split16', "
            "'mnist_pca16_single', 'mnist_pca16_avgpool16', "
            "'fashion_mnist_subset_single', 'fashion_mnist_subclass_composite_single', "
            "'fashion_mnist_subclass_composite_dual', 'imagenet_scene_subset_single', "
            "'cifar100_hierarchical_subset', 'cifar100_hierarchical_subset_single'}"
        )

    training_package = _downsample_training_package(
        _build_training_package(mapped_network_package, train_cfg),
        target_samples=(
            None
            if model_cfg.get("monitor_downsample_to") is None
            else int(model_cfg.get("monitor_downsample_to"))
        ),
    )
    device = _resolve_device(
        model_cfg.get(
            "device",
            training_package.get("optical_donn_config", {}).get("device", "cpu"),
        )
    )
    model = instantiate_optical_donn_from_mapped_package(
        training_package,
        work_dir=train_dir / "model_workspace",
        device=str(device),
    ).to(device)
    if bool(model_cfg.get("reinitialize_parameters", True)):
        if "initcanshu" in model_cfg:
            model.initcanshu = model_cfg["initcanshu"]
        model.init_parameters()

    calibration_monitors = list(
        (training_package.get("calibration_package") or {}).get("monitors") or []
    )
    if calibration_monitors:
        first_monitor = calibration_monitors[0]
        if isinstance(first_monitor, Mapping):
            monitor_samples = int(np.asarray(first_monitor.get("y_um", [])).reshape(-1).size)
        else:
            monitor_samples = int(np.asarray(first_monitor).reshape(-1).size)
    else:
        monitor_samples = 0
    requested_monitor_downsample_to = model_cfg.get("monitor_downsample_to")

    num_inputs = len(getattr(model, "inputposition"))
    num_outputs = len(getattr(model, "outputposition"))
    batch_size = int(loop_cfg.get("batch_size", 32))
    if batch_size <= 0:
        raise ValueError("train.loop.batch_size must be positive")
    augment_noise_std = float(dataset_cfg.get("augment_noise_std", 0.0))
    if augment_noise_std < 0.0:
        raise ValueError("train.dataset.augment_noise_std must be >= 0")
    input_low = float(dataset_cfg.get("input_low", 0.0))
    input_high = float(dataset_cfg.get("input_high", 2.0))
    optimizer = _build_optimizer(
        optimizer_cfg.get("name", "adam"),
        list(model.parameters()),
        lr=float(optimizer_cfg.get("lr", 1e-3)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        momentum=float(optimizer_cfg.get("momentum", 0.0)),
        nesterov=bool(optimizer_cfg.get("nesterov", False)),
    )
    scheduler = _build_scheduler(optimizer, optimizer_cfg)
    loss_fn = _build_loss(objective_cfg.get("loss", "cross_entropy_prob"))
    exploration_state = _resolve_exploration_state(exploration_cfg)

    if dataset_kind in {
        "mnist_pca16_single",
        "fashion_mnist_subset_single",
        "fashion_mnist_subclass_composite_single",
        "imagenet_scene_subset_single",
        "cifar100_hierarchical_subset_single",
    }:
        wavelength = _resolve_single_task_wavelength(model, task_cfg)
        if dataset_kind == "mnist_pca16_single":
            x_train, x_eval, y_train, y_eval, explained, dataset_summary = _make_mnist_pca16_single_dataset(
                dataset_cfg=dataset_cfg,
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                seed=seed,
            )
        elif dataset_kind == "fashion_mnist_subset_single":
            x_train, x_eval, y_train, y_eval, explained, dataset_summary = _make_fashion_mnist_subset_single_dataset(
                dataset_cfg=dataset_cfg,
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                seed=seed,
            )
        elif dataset_kind == "fashion_mnist_subclass_composite_single":
            x_train, x_eval, y_train, y_eval, explained, dataset_summary = _make_fashion_mnist_subclass_composite_single_dataset(
                dataset_cfg=dataset_cfg,
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                seed=seed,
            )
        elif dataset_kind == "imagenet_scene_subset_single":
            x_train, x_eval, y_train, y_eval, explained, dataset_summary = _make_imagenet_scene_subset_single_dataset(
                dataset_cfg=dataset_cfg,
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                seed=seed,
            )
        else:
            x_train, x_eval, y_train, y_eval, explained, dataset_summary = _make_cifar100_hierarchical_subset_single_dataset(
                dataset_cfg=dataset_cfg,
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                seed=seed,
            )
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        eval_loader = DataLoader(
            TensorDataset(
                torch.tensor(x_eval, dtype=torch.float32),
                torch.tensor(y_eval, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        if dataset_kind == "mnist_pca16_single":
            print(
                "task=mnist_pca16_single_wavelength "
                f"wavelength_um={wavelength['wavelength_um']} "
                f"train_samples={x_train.shape[0]} eval_samples={x_eval.shape[0]} "
                f"input_dim={x_train.shape[1]} output_dim={y_train.shape[1]} device={device}"
            )
            print(f"pca_explained_variance_ratio_sum={explained:.6f}")
        elif dataset_kind == "fashion_mnist_subset_single":
            print(
                "task=fashion_mnist_subset_single_wavelength "
                f"wavelength_um={wavelength['wavelength_um']} "
                f"train_samples={x_train.shape[0]} eval_samples={x_eval.shape[0]} "
                f"input_dim={x_train.shape[1]} output_dim={y_train.shape[1]} device={device}"
            )
            print(
                "classes="
                f"{dataset_summary['class_names_active']} "
                f"active_classes={len(dataset_summary['class_names_active'])}"
            )
            print(f"pca_explained_variance_ratio_sum={explained:.6f}")
        elif dataset_kind == "fashion_mnist_subclass_composite_single":
            print(
                "task=fashion_mnist_subclass_composite_single_wavelength "
                f"wavelength_um={wavelength['wavelength_um']} "
                f"train_samples={x_train.shape[0]} eval_samples={x_eval.shape[0]} "
                f"input_dim={x_train.shape[1]} output_dim={y_train.shape[1]} device={device}"
            )
            print(
                "classes="
                f"{dataset_summary['class_names_active']} "
                f"active_classes={dataset_summary['active_class_count']} "
                f"base_classes={dataset_summary['selected_base_class_names']}"
            )
            print(f"pca_explained_variance_ratio_sum={explained:.6f}")
        elif dataset_kind == "imagenet_scene_subset_single":
            print(
                "task=imagenet_scene_subset_single_wavelength "
                f"wavelength_um={wavelength['wavelength_um']} "
                f"train_samples={x_train.shape[0]} eval_samples={x_eval.shape[0]} "
                f"input_dim={x_train.shape[1]} output_dim={y_train.shape[1]} device={device}"
            )
            print(
                "classes="
                f"{dataset_summary['class_names_active']} "
                f"active_classes={len(dataset_summary['class_names_active'])} "
                f"split_source={dataset_summary['split_source']}"
            )
            print(
                "image_features="
                f"{dataset_summary['image_color_mode']}{tuple(dataset_summary['image_size'])} "
                f"kind={dataset_summary['feature_kind']} "
                f"pca_explained_variance_ratio_sum={explained:.6f}"
            )
        else:
            print(
                "task=cifar100_hierarchical_subset_single_wavelength "
                f"wavelength_um={wavelength['wavelength_um']} "
                f"train_samples={x_train.shape[0]} eval_samples={x_eval.shape[0]} "
                f"input_dim={x_train.shape[1]} output_dim={y_train.shape[1]} device={device}"
            )
            print(
                "hierarchy="
                f"target_level={dataset_summary['target_level']} "
                f"coarse_classes={dataset_summary['selected_coarse_names']} "
                f"fine_per_coarse={dataset_summary['fine_per_coarse']} "
                f"active_classes={dataset_summary['active_class_count']}"
            )
            print(
                "input_feature="
                f"{str(dataset_summary['feature_branch']).upper()}[{dataset_summary['single_feature_kind']}]"
            )
            if explained > 0.0:
                print(f"pca_explained_variance_ratio_sum={explained:.6f}")
        print(
            "scaled_input_range="
            f"[{dataset_summary['input_train_min']:.3f}, {dataset_summary['input_train_max']:.3f}] "
            f"mean={dataset_summary['input_train_mean']:.3f} std={dataset_summary['input_train_std']:.3f}"
        )
        if model_cfg.get("monitor_downsample_to") is not None:
            print(f"monitor_downsample_to={int(model_cfg['monitor_downsample_to'])}")
        print(f"monitor_samples_used={monitor_samples}")
        print(
            "train_shape="
            f"total_length_um={training_package['shape_spec']['shape_config']['total_length_um']} "
            f"inter_layer_spacings_um={training_package['shape_spec']['shape_config']['inter_layer_spacings_um']}"
        )
        if scheduler is not None:
            print(
                "lr_schedule=multistep "
                f"milestones={list(optimizer_cfg.get('lr_milestones') or [])} "
                f"gamma={float(optimizer_cfg.get('lr_gamma', 0.1)):.6f}"
            )
        if augment_noise_std > 0.0:
            print(
                f"train_input_noise_std={augment_noise_std:.6f} "
                f"clamp_range=[{input_low:.3f}, {input_high:.3f}]"
            )
        if exploration_state["enabled"]:
            print(
                "exploration=plateau_jitter "
                f"metric={exploration_state['metric']} "
                f"goal={exploration_state['goal']} "
                f"patience={exploration_state['plateau_patience']} "
                f"warmup_epochs={exploration_state['warmup_epochs']} "
                f"jitter_std_um={exploration_state['jitter_std_um']:.6f} "
                f"jitter_decay={exploration_state['jitter_decay']:.6f}"
            )
        return {
            "training_mode": "single",
            "model": model,
            "mapped_network_package_for_training": training_package,
            "device": device,
            "objective_cfg": objective_cfg,
            "constraints_cfg": constraints_cfg,
            "loss_fn": loss_fn,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "train_loader": train_loader,
            "eval_loader": eval_loader,
            "task_wavelengths_um": [wavelength["wavelength_um"]],
            "task_wavelength_keys": [wavelength["wavelength_key"]],
            "reference_input": torch.tensor(x_eval[:1], dtype=torch.float32, device=device),
            "reference_target": y_eval[:1].reshape(-1).tolist(),
            "task_kind": "supervised_dataset",
            "dataset_kind": dataset_kind,
            "train_samples": int(x_train.shape[0]),
            "eval_samples": int(x_eval.shape[0]),
            "input_dim_per_wavelength": int(x_train.shape[1]),
            "output_dim": int(y_train.shape[1]),
            "class_names": list(
                dataset_summary.get(
                    "class_names_padded",
                    _pad_class_names(CLASS_NAMES, output_dim=int(y_train.shape[1])),
                )
            ),
            "pca_explained_variance_ratio_sum": explained,
            "dataset_summary": dataset_summary,
            "monitor_samples_used": int(monitor_samples),
            "requested_monitor_downsample_to": (
                None
                if requested_monitor_downsample_to is None
                else int(requested_monitor_downsample_to)
            ),
            "augment_noise_std": augment_noise_std,
            "input_low": input_low,
            "input_high": input_high,
            "exploration_state": copy.deepcopy(exploration_state),
        }

    wavelengths = _resolve_task_wavelengths(model, task_cfg)
    if dataset_kind == "mnist_pca16_avgpool16":
        x_train_a, x_train_b, x_eval_a, x_eval_b, y_train, y_eval, explained, dataset_summary = _make_mnist_pca16_avgpool16_dataset(
            dataset_cfg=dataset_cfg,
            num_branch_inputs=num_inputs,
            num_outputs=num_outputs,
            seed=seed,
        )
    elif dataset_kind == "cifar100_hierarchical_subset":
        x_train_a, x_train_b, x_eval_a, x_eval_b, y_train, y_eval, explained, dataset_summary = _make_cifar100_hierarchical_subset_dataset(
            dataset_cfg=dataset_cfg,
            num_branch_inputs=num_inputs,
            num_outputs=num_outputs,
            seed=seed,
        )
    elif dataset_kind == "fashion_mnist_subclass_composite_dual":
        x_train_a, x_train_b, x_eval_a, x_eval_b, y_train, y_eval, explained, dataset_summary = _make_fashion_mnist_subclass_composite_dual_dataset(
            dataset_cfg=dataset_cfg,
            num_branch_inputs=num_inputs,
            num_outputs=num_outputs,
            seed=seed,
        )
    else:
        x_train_a, x_train_b, x_eval_a, x_eval_b, y_train, y_eval, explained, dataset_summary = _make_mnist_pca_split_dataset(
            dataset_cfg=dataset_cfg,
            num_branch_inputs=num_inputs,
            num_outputs=num_outputs,
            seed=seed,
        )
    train_dataset = TensorDataset(
        torch.tensor(x_train_a, dtype=torch.float32),
        torch.tensor(x_train_b, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    eval_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_eval_a, dtype=torch.float32),
            torch.tensor(x_eval_b, dtype=torch.float32),
            torch.tensor(y_eval, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    branch_loss_weights = np.asarray(task_cfg.get("branch_loss_weights", [1.0, 1.0]), dtype=np.float32).reshape(-1)
    if branch_loss_weights.shape != (2,):
        raise ValueError("train.task.branch_loss_weights must contain exactly two values")
    combined_loss_weight = float(task_cfg.get("combined_loss_weight", 1.0))
    if combined_loss_weight < 0.0:
        raise ValueError("train.task.combined_loss_weight must be >= 0")
    include_combined_in_total_loss = bool(task_cfg.get("include_combined_in_total_loss", True))
    selection_metric_mode = str(task_cfg.get("selection_metric_mode", "combined"))
    dual_hierarchy = _resolve_dual_hierarchy_config(
        task_cfg=task_cfg,
        objective_cfg=objective_cfg,
        dataset_summary=dataset_summary,
        output_dim=int(y_train.shape[1]),
    )
    phase_schedule = _resolve_dual_phase_schedule(
        train_cfg=train_cfg,
        loop_epochs=int(loop_cfg.get("epochs", 1)),
        branch_loss_weights=branch_loss_weights,
        combined_loss_weight=combined_loss_weight,
        include_combined_in_total_loss=include_combined_in_total_loss,
        selection_metric_mode=selection_metric_mode,
    )
    train_loaders_by_key: Dict[str, DataLoader] = {
        "default": _make_shuffled_train_loader(
            train_dataset,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
    }
    if phase_schedule is not None:
        oversample_factors = sorted(
            {
                float(phase["branch_b_focus_oversample_factor"])
                for phase in phase_schedule
                if float(phase["branch_b_focus_oversample_factor"]) > 1.0
            }
        )
        if oversample_factors:
            focus_coarse_index = (
                None if dual_hierarchy is None else dual_hierarchy.get("branch_b_focus_coarse_index")
            )
            if focus_coarse_index is None:
                raise ValueError(
                    "train.phases uses branch_b_focus_oversample_factor > 1, but "
                    "train.task.branch_b_focus_coarse is not configured"
                )
            focus_mask = (
                np.argmax(y_train[:, : int(dual_hierarchy["active_class_count"])], axis=1)
                // int(dual_hierarchy["fine_per_coarse"])
            ) == int(focus_coarse_index)
            for factor_idx, factor in enumerate(oversample_factors, start=1):
                loader_key = f"focus_x{factor:g}".replace(".", "p")
                sample_weights = np.ones(int(focus_mask.shape[0]), dtype=np.float64)
                sample_weights[focus_mask] = float(factor)
                train_loaders_by_key[loader_key] = _make_weighted_train_loader(
                    train_dataset,
                    batch_size=batch_size,
                    seed=seed + factor_idx,
                    device=device,
                    sample_weights=sample_weights,
                )
            for phase in phase_schedule:
                factor = float(phase["branch_b_focus_oversample_factor"])
                phase["train_loader_key"] = (
                    "default"
                    if factor <= 1.0
                    else f"focus_x{factor:g}".replace(".", "p")
                )
        else:
            for phase in phase_schedule:
                phase["train_loader_key"] = "default"
    train_loader = train_loaders_by_key["default"]

    if dataset_kind == "mnist_pca16_avgpool16":
        branch_b_shape_info = (
            f"pool_block_shape={dataset_summary['branch_b_pool_block_shape']}"
            if "branch_b_pool_block_shape" in dataset_summary
            else f"residual_block_shape={dataset_summary['branch_b_residual_block_shape']}"
            if "branch_b_residual_block_shape" in dataset_summary
            else f"lowfreq_block_shape={dataset_summary['branch_b_lowfreq_block_shape']}"
            if "branch_b_lowfreq_block_shape" in dataset_summary
            else "shape_info=unavailable"
        )
        print(
            "task=mnist_pca16_avgpool16_dual_wavelength "
            f"wavelengths_um={wavelengths['wavelengths_um']} "
            f"train_samples={x_train_a.shape[0]} eval_samples={x_eval_a.shape[0]} "
            f"branch_input_dim={x_train_a.shape[1]} output_dim={y_train.shape[1]} device={device}"
        )
        print(
            "branch_features="
            f"A[pca16 explained_variance_ratio_sum={dataset_summary['branch_a_explained_variance_ratio_sum']:.6f}] "
            f"B[{dataset_summary['branch_b_feature_kind']} {branch_b_shape_info}]"
        )
    elif dataset_kind == "cifar100_hierarchical_subset":
        branch_b_shape_info = (
            f"pool_block_shape={dataset_summary['branch_b_pool_block_shape']}"
            if "branch_b_pool_block_shape" in dataset_summary
            else f"residual_block_shape={dataset_summary['branch_b_residual_block_shape']}"
            if "branch_b_residual_block_shape" in dataset_summary
            else f"lowfreq_block_shape={dataset_summary['branch_b_lowfreq_block_shape']}"
            if "branch_b_lowfreq_block_shape" in dataset_summary
            else "shape_info=unavailable"
        )
        print(
            "task=cifar100_hierarchical_subset_dual_wavelength "
            f"wavelengths_um={wavelengths['wavelengths_um']} "
            f"train_samples={x_train_a.shape[0]} eval_samples={x_eval_a.shape[0]} "
            f"branch_input_dim={x_train_a.shape[1]} output_dim={y_train.shape[1]} device={device}"
        )
        print(
            "hierarchy="
            f"target_level={dataset_summary['target_level']} "
            f"coarse_classes={dataset_summary['selected_coarse_names']} "
            f"fine_per_coarse={dataset_summary['fine_per_coarse']} "
            f"active_classes={dataset_summary['active_class_count']}"
        )
        print(
            "branch_features="
            f"A[{dataset_summary['branch_a_feature_kind']} explained_variance_ratio_sum={dataset_summary['branch_a_explained_variance_ratio_sum']:.6f}] "
            f"B[{dataset_summary['branch_b_feature_kind']} {branch_b_shape_info}]"
        )
    elif dataset_kind == "fashion_mnist_subclass_composite_dual":
        print(
            "task=fashion_mnist_subclass_composite_dual_wavelength "
            f"wavelengths_um={wavelengths['wavelengths_um']} "
            f"train_samples={x_train_a.shape[0]} eval_samples={x_eval_a.shape[0]} "
            f"branch_input_dim={x_train_a.shape[1]} output_dim={y_train.shape[1]} device={device}"
        )
        print(
            "classes="
            f"{dataset_summary['class_names_active']} "
            f"active_classes={dataset_summary['active_class_count']} "
            f"base_classes={dataset_summary['selected_base_class_names']}"
        )
        if "branch_b_pool_block_shape" in dataset_summary:
            branch_b_shape_info = f"pool_block_shape={dataset_summary['branch_b_pool_block_shape']}"
        elif "branch_b_texture_block_shape" in dataset_summary:
            if "branch_b_center_shape_blend_shape_weight" in dataset_summary:
                branch_b_shape_info = (
                    f"texture_block_shape={dataset_summary['branch_b_texture_block_shape']} "
                    f"center_crop_box={dataset_summary['branch_b_center_icon_crop_box']} "
                    f"center_weight={dataset_summary.get('branch_b_center_shape_blend_center_weight', 0.0):.3f} "
                    f"shape_weight={dataset_summary.get('branch_b_center_shape_blend_shape_weight', 0.0):.3f} "
                    f"shape_descriptor_dim={dataset_summary.get('branch_b_center_shape_descriptor_dim', 0)} "
                    f"label_routing={dataset_summary.get('branch_b_uses_target_labels_for_feature_routing', False)}"
                )
            elif "branch_b_center_icon_crop_box" in dataset_summary:
                branch_b_shape_info = (
                    f"texture_block_shape={dataset_summary['branch_b_texture_block_shape']} "
                    f"center_crop_box={dataset_summary['branch_b_center_icon_crop_box']} "
                    f"deviation_weight={dataset_summary.get('branch_b_center_icon_deviation_weight', 0.0):.3f} "
                    f"edge_weight={dataset_summary.get('branch_b_center_icon_edge_weight', 0.0):.3f} "
                    f"contrast_weight={dataset_summary.get('branch_b_center_icon_contrast_weight', 0.0):.3f}"
                )
            elif "branch_b_oriented_texture_orientation_weight" in dataset_summary:
                branch_b_shape_info = (
                    f"texture_block_shape={dataset_summary['branch_b_texture_block_shape']} "
                    f"edge_weight={dataset_summary.get('branch_b_oriented_texture_edge_weight', 0.0):.3f} "
                    f"orientation_weight={dataset_summary.get('branch_b_oriented_texture_orientation_weight', 0.0):.3f} "
                    f"contrast_weight={dataset_summary.get('branch_b_oriented_texture_contrast_weight', 0.0):.3f}"
                )
            else:
                branch_b_shape_info = (
                    f"texture_block_shape={dataset_summary['branch_b_texture_block_shape']} "
                    f"edge_weight={dataset_summary.get('branch_b_local_texture_edge_weight', 0.0):.3f} "
                    f"contrast_weight={dataset_summary.get('branch_b_local_texture_contrast_weight', 0.0):.3f}"
                )
        else:
            branch_b_shape_info = "shape_info=unavailable"
        print(
            "branch_features="
            f"A[{dataset_summary['branch_a_feature_kind']} explained_variance_ratio_sum={dataset_summary['branch_a_explained_variance_ratio_sum']:.6f}] "
            f"B[{dataset_summary['branch_b_feature_kind']} {branch_b_shape_info}]"
        )
    else:
        print(
            "task=mnist_pca_split16_dual_wavelength "
            f"wavelengths_um={wavelengths['wavelengths_um']} "
            f"train_samples={x_train_a.shape[0]} eval_samples={x_eval_a.shape[0]} "
            f"branch_input_dim={x_train_a.shape[1]} output_dim={y_train.shape[1]} device={device}"
        )
        print(f"pca_explained_variance_ratio_sum={explained:.6f}")
        print(
            "component_split={component_split} "
            "branch_explained_variance_ratio=({branch_a_explained_variance_ratio_sum:.6f}, "
            "{branch_b_explained_variance_ratio_sum:.6f})".format(**dataset_summary)
        )
        print(
            "shared_component_count={shared_component_count} unique_component_count={unique_component_count} "
            "shared_explained_variance_ratio_sum={shared_explained_variance_ratio_sum:.6f}".format(**dataset_summary)
        )
    print(
        "scaled_input_ranges="
        f"A[{dataset_summary['branch_a_train_min']:.3f}, {dataset_summary['branch_a_train_max']:.3f}] "
        f"B[{dataset_summary['branch_b_train_min']:.3f}, {dataset_summary['branch_b_train_max']:.3f}]"
    )
    if model_cfg.get("monitor_downsample_to") is not None:
        print(f"monitor_downsample_to={int(model_cfg['monitor_downsample_to'])}")
    print(f"monitor_samples_used={monitor_samples}")
    print(
        "train_shape="
        f"total_length_um={training_package['shape_spec']['shape_config']['total_length_um']} "
        f"inter_layer_spacings_um={training_package['shape_spec']['shape_config']['inter_layer_spacings_um']}"
    )
    if scheduler is not None:
        print(
            "lr_schedule=multistep "
            f"milestones={list(optimizer_cfg.get('lr_milestones') or [])} "
            f"gamma={float(optimizer_cfg.get('lr_gamma', 0.1)):.6f}"
        )
    if augment_noise_std > 0.0:
        print(
            f"train_input_noise_std={augment_noise_std:.6f} "
            f"clamp_range=[{input_low:.3f}, {input_high:.3f}]"
        )
    if dual_hierarchy is not None:
        print(
            "objective=hierarchical_dual "
            f"mode={dual_hierarchy['mode']} "
            f"coarse_branch=A({wavelengths['wavelength_keys'][0]}) "
            f"local_fine_branch=B({wavelengths['wavelength_keys'][1]}) "
            f"shared_local_ports_b={bool(dual_hierarchy.get('shared_local_ports_b', False))} "
            f"branch_b_uses_coarse_context={bool(dual_hierarchy.get('branch_b_uses_coarse_context', False))} "
            f"branch_b_focus_local_ports_only={bool(dual_hierarchy.get('branch_b_focus_local_ports_only', False))} "
            f"branch_b_focus_coarse={dual_hierarchy.get('branch_b_focus_coarse_name')} "
            f"fine_per_coarse={dual_hierarchy['fine_per_coarse']} "
            f"active_fine_classes={dual_hierarchy['active_class_count']}"
        )
    if not include_combined_in_total_loss:
        print("loss_mode=branch_only (combined branch is eval-only, no gradient in loss_total)")
    if phase_schedule is not None:
        for phase in phase_schedule:
            print(
                "phase_schedule="
                f"{phase['name']} epochs={phase['start_epoch']}-{phase['end_epoch']} "
                f"branch_loss_weights={phase['branch_loss_weights']} "
                f"branch_b_focus_oversample_factor={phase['branch_b_focus_oversample_factor']:.3f} "
                f"selection_metric_mode={phase['selection_metric_mode']}"
            )
    if exploration_state["enabled"]:
        print(
            "exploration=plateau_jitter "
            f"metric={exploration_state['metric']} "
            f"goal={exploration_state['goal']} "
            f"patience={exploration_state['plateau_patience']} "
            f"warmup_epochs={exploration_state['warmup_epochs']} "
            f"jitter_std_um={exploration_state['jitter_std_um']:.6f} "
            f"jitter_decay={exploration_state['jitter_decay']:.6f}"
        )

    return {
        "training_mode": "dual",
        "model": model,
        "mapped_network_package_for_training": training_package,
        "device": device,
        "objective_cfg": objective_cfg,
        "constraints_cfg": constraints_cfg,
        "loss_fn": loss_fn,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "train_loaders_by_key": train_loaders_by_key,
        "eval_loader": eval_loader,
        "task_wavelengths_um": wavelengths["wavelengths_um"],
        "task_wavelength_keys": wavelengths["wavelength_keys"],
        "branch_loss_weights": torch.tensor(branch_loss_weights, dtype=torch.float32, device=device),
        "combined_loss_weight": combined_loss_weight,
        "include_combined_in_total_loss": include_combined_in_total_loss,
        "reference_input": torch.tensor(x_eval_a[:1], dtype=torch.float32, device=device),
        "reference_input_a": torch.tensor(x_eval_a[:1], dtype=torch.float32, device=device),
        "reference_input_b": torch.tensor(x_eval_b[:1], dtype=torch.float32, device=device),
        "reference_target": y_eval[:1].reshape(-1).tolist(),
        "task_kind": "supervised_dataset",
        "dataset_kind": dataset_kind,
        "train_samples": int(x_train_a.shape[0]),
        "eval_samples": int(x_eval_a.shape[0]),
        "input_dim_per_wavelength": int(x_train_a.shape[1]),
        "output_dim": int(y_train.shape[1]),
        "class_names": list(dataset_summary.get("class_names_padded") or _pad_class_names(CLASS_NAMES, output_dim=int(y_train.shape[1]))),
        "pca_explained_variance_ratio_sum": explained,
        "dataset_summary": dataset_summary,
        "monitor_samples_used": int(monitor_samples),
        "requested_monitor_downsample_to": (
            None
            if requested_monitor_downsample_to is None
            else int(requested_monitor_downsample_to)
        ),
        "augment_noise_std": augment_noise_std,
        "input_low": input_low,
        "input_high": input_high,
        "selection_metric_mode": selection_metric_mode,
        "exploration_state": copy.deepcopy(exploration_state),
        "phase_schedule": copy.deepcopy(phase_schedule),
        "dual_hierarchy": copy.deepcopy(dual_hierarchy),
        "reference_output_layout_by_wavelength": {
            wavelengths["wavelength_keys"][0]: (
                list(dual_hierarchy["wl_a_output_layout"])
                if dual_hierarchy is not None
                else [f"port_{port_idx}" for port_idx in range(int(y_train.shape[1]))]
            ),
            wavelengths["wavelength_keys"][1]: (
                list(dual_hierarchy["wl_b_output_layout"])
                if dual_hierarchy is not None
                else [f"port_{port_idx}" for port_idx in range(int(y_train.shape[1]))]
            ),
        },
        "reference_combined_output_layout": (
            list(dual_hierarchy["combined_output_layout"])
            if dual_hierarchy is not None
            else [f"combined_port_{port_idx}" for port_idx in range(int(y_train.shape[1]))]
        ),
    }


def train_one_epoch(state: Mapping[str, Any], epoch: int) -> Dict[str, Any]:
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state.get("scheduler")
    loss_fn = state["loss_fn"]
    objective_cfg = state["objective_cfg"]
    constraints_cfg = state["constraints_cfg"]
    if state.get("training_mode") == "single":
        wl = float(state["task_wavelengths_um"][0])
        wl_key = str(state["task_wavelength_keys"][0])

        model.train()
        exploration_metrics = _maybe_apply_pending_exploration_jitter(state, epoch=epoch)
        total_loss = 0.0
        total_count = 0
        correct = 0

        for x, y in state["train_loader"]:
            x = x.to(state["device"])
            y = y.to(state["device"])
            x = _apply_train_input_noise(
                x,
                std=float(state.get("augment_noise_std", 0.0)),
                input_low=float(state.get("input_low", 0.0)),
                input_high=float(state.get("input_high", 2.0)),
            )

            optimizer.zero_grad(set_to_none=True)
            outputs = _forward_single_wavelength(
                model,
                x,
                wavelength_um=wl,
                objective_cfg=objective_cfg,
                device=state["device"],
            )
            loss = loss_fn(outputs["pred"], y)
            loss.backward()
            optimizer.step()
            _apply_slot_length_constraints(model, constraints_cfg)

            batch = x.shape[0]
            total_count += batch
            total_loss += float(loss.item()) * batch
            correct += _accuracy_from_predictions(outputs["pred"], y)

        if scheduler is not None:
            scheduler.step()
        accuracy = correct / max(total_count, 1)
        return {
            "train_loss": total_loss / max(total_count, 1),
            f"train_loss_{wl_key}": total_loss / max(total_count, 1),
            "train_loss_combined": total_loss / max(total_count, 1),
            "train_accuracy": accuracy,
            f"train_accuracy_{wl_key}": accuracy,
            "train_accuracy_combined": accuracy,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "slot_length_min_um": float(
                min(float(param.detach().min().cpu()) for param in getattr(model, "slot_length_um", []) if param.numel())
            ),
            "slot_length_max_um": float(
                max(float(param.detach().max().cpu()) for param in getattr(model, "slot_length_um", []) if param.numel())
            ),
            **exploration_metrics,
        }

    wl_a, wl_b = state["task_wavelengths_um"]
    wl_key_a, wl_key_b = state["task_wavelength_keys"]
    dual_hierarchy = state.get("dual_hierarchy")
    active_settings = _resolve_active_dual_training_settings(state, epoch)

    model.train()
    exploration_metrics = _maybe_apply_pending_exploration_jitter(state, epoch=epoch)
    total_loss = 0.0
    total_loss_a = 0.0
    total_loss_b = 0.0
    total_loss_sum = 0.0
    total_count = 0
    total_count_b = 0
    correct_a = 0
    correct_b = 0
    correct_sum = 0

    for x_a, x_b, y in active_settings["train_loader"]:
        x_a = x_a.to(state["device"])
        x_b = x_b.to(state["device"])
        y = y.to(state["device"])
        x_a = _apply_train_input_noise(
            x_a,
            std=float(state.get("augment_noise_std", 0.0)),
            input_low=float(state.get("input_low", 0.0)),
            input_high=float(state.get("input_high", 2.0)),
        )
        x_b = _apply_train_input_noise(
            x_b,
            std=float(state.get("augment_noise_std", 0.0)),
            input_low=float(state.get("input_low", 0.0)),
            input_high=float(state.get("input_high", 2.0)),
        )

        optimizer.zero_grad(set_to_none=True)
        outputs = _forward_two_wavelengths(
            model,
            x_a,
            x_b,
            wavelength_a_um=wl_a,
            wavelength_b_um=wl_b,
            objective_cfg=objective_cfg,
            device=state["device"],
            dual_hierarchy=dual_hierarchy,
        )
        losses = _compute_losses(
            outputs,
            y,
            loss_fn=loss_fn,
            branch_weights=active_settings["branch_loss_weights"],
            combined_weight=active_settings["combined_loss_weight"],
            dual_hierarchy=dual_hierarchy,
            include_combined_in_total_loss=bool(active_settings["include_combined_in_total_loss"]),
        )
        losses["loss_total"].backward()
        optimizer.step()
        _apply_slot_length_constraints(model, constraints_cfg)

        batch = x_a.shape[0]
        total_count += batch
        total_loss += float(losses["loss_total"].item()) * batch
        total_loss_a += float(losses["loss_a"].item()) * batch
        total_loss_b += float(losses["loss_b"].item()) * batch
        total_loss_sum += float(losses["loss_sum"].item()) * batch
        if dual_hierarchy is None:
            correct_a += _accuracy_from_predictions(outputs["pred_a"], y)
            correct_b += _accuracy_from_predictions(outputs["pred_b"], y)
            correct_sum += _accuracy_from_predictions(outputs["pred_sum"], y)
        else:
            hierarchy_targets = _extract_dual_hierarchy_targets(y, dual_hierarchy)
            pred_b_local, branch_b_mask = _resolve_branch_b_task_predictions(
                outputs,
                hierarchy_targets,
                dual_hierarchy,
            )
            correct_a += int(
                (outputs["pred_a"].argmax(dim=-1) == hierarchy_targets["coarse_idx"]).sum().item()
            )
            if branch_b_mask is None:
                correct_b += int(
                    (pred_b_local.argmax(dim=-1) == hierarchy_targets["local_idx"]).sum().item()
                )
                total_count_b += batch
            else:
                total_count_b += int(branch_b_mask.sum().item())
                if bool(branch_b_mask.any().item()):
                    correct_b += int(
                        (
                            pred_b_local[branch_b_mask].argmax(dim=-1)
                            == hierarchy_targets["local_idx"][branch_b_mask]
                        ).sum().item()
                    )
            correct_sum += int(
                (outputs["pred_sum"].argmax(dim=-1) == hierarchy_targets["fine_idx"]).sum().item()
            )

    if scheduler is not None:
        scheduler.step()
    train_accuracy_a = correct_a / max(total_count, 1)
    train_accuracy_b = correct_b / max(total_count_b, 1)
    train_accuracy_sum = correct_sum / max(total_count, 1)
    return {
        "train_loss": total_loss / max(total_count, 1),
        f"train_loss_{wl_key_a}": total_loss_a / max(total_count, 1),
        f"train_loss_{wl_key_b}": total_loss_b / max(total_count, 1),
        "train_loss_combined": total_loss_sum / max(total_count, 1),
        "train_accuracy": train_accuracy_sum,
        f"train_accuracy_{wl_key_a}": train_accuracy_a,
        f"train_accuracy_{wl_key_b}": train_accuracy_b,
        "train_accuracy_branch_mean": 0.5 * (train_accuracy_a + train_accuracy_b),
        "train_accuracy_combined": train_accuracy_sum,
        "lr": float(optimizer.param_groups[0]["lr"]),
        "phase_name": active_settings["phase_name"],
        "phase_epoch": int(active_settings["phase_epoch"]),
        "slot_length_min_um": float(
            min(float(param.detach().min().cpu()) for param in getattr(model, "slot_length_um", []) if param.numel())
        ),
        "slot_length_max_um": float(
            max(float(param.detach().max().cpu()) for param in getattr(model, "slot_length_um", []) if param.numel())
        ),
        **exploration_metrics,
    }


def evaluate_model(state: Mapping[str, Any], epoch: int) -> Dict[str, Any]:
    model = state["model"]
    loss_fn = state["loss_fn"]
    objective_cfg = state["objective_cfg"]
    if state.get("training_mode") == "single":
        wl = float(state["task_wavelengths_um"][0])
        wl_key = str(state["task_wavelength_keys"][0])

        model.eval()
        total_loss = 0.0
        total_count = 0
        correct = 0
        confusion = np.zeros((int(state["output_dim"]), int(state["output_dim"])), dtype=np.int64)

        with torch.no_grad():
            for x, y in state["eval_loader"]:
                x = x.to(state["device"])
                y = y.to(state["device"])
                outputs = _forward_single_wavelength(
                    model,
                    x,
                    wavelength_um=wl,
                    objective_cfg=objective_cfg,
                    device=state["device"],
                )
                loss = loss_fn(outputs["pred"], y)
                batch = x.shape[0]
                total_count += batch
                total_loss += float(loss.item()) * batch
                correct += _accuracy_from_predictions(outputs["pred"], y)

                target_idx = _class_indices_from_one_hot(y)
                confusion += _confusion_matrix(outputs["pred"].argmax(dim=-1), target_idx, int(state["output_dim"]))

            reference_outputs = _forward_single_wavelength(
                model,
                state["reference_input"],
                wavelength_um=wl,
                objective_cfg=objective_cfg,
                device=state["device"],
            )

        eval_accuracy = correct / max(total_count, 1)
        raw_output = reference_outputs["raw"].detach().cpu().reshape(-1).tolist()
        transformed_output = reference_outputs["pred"].detach().cpu().reshape(-1).tolist()
        metrics = {
            "eval_loss": total_loss / max(total_count, 1),
            f"eval_loss_{wl_key}": total_loss / max(total_count, 1),
            "eval_loss_combined": total_loss / max(total_count, 1),
            "eval_accuracy": eval_accuracy,
            f"eval_accuracy_{wl_key}": eval_accuracy,
            "eval_accuracy_combined": eval_accuracy,
            "selection_metric": -eval_accuracy,
            "raw_output": raw_output,
            "transformed_output": transformed_output,
            "target_output": list(state["reference_target"]),
            "reference_prediction_by_wavelength": {
                wl_key: transformed_output,
            },
            "reference_raw_prediction_by_wavelength": {
                wl_key: raw_output,
            },
            "reference_prediction_combined": transformed_output,
            "reference_raw_prediction_combined": raw_output,
            "confusion_matrix": confusion.tolist(),
            "confusion_matrix_by_wavelength": {
                wl_key: confusion.tolist(),
            },
        }
        return _update_exploration_state_after_eval(state, epoch=epoch, metrics=metrics)

    wl_a, wl_b = state["task_wavelengths_um"]
    wl_key_a, wl_key_b = state["task_wavelength_keys"]
    dual_hierarchy = state.get("dual_hierarchy")
    active_settings = _resolve_active_dual_training_settings(state, epoch)

    model.eval()
    total_loss = 0.0
    total_loss_a = 0.0
    total_loss_b = 0.0
    total_loss_sum = 0.0
    total_count = 0
    total_count_b = 0
    correct_a = 0
    correct_b = 0
    correct_sum = 0
    if dual_hierarchy is None:
        confusion_a = np.zeros((int(state["output_dim"]), int(state["output_dim"])), dtype=np.int64)
        confusion_b = np.zeros((int(state["output_dim"]), int(state["output_dim"])), dtype=np.int64)
    else:
        confusion_a = np.zeros(
            (int(dual_hierarchy["coarse_count"]), int(dual_hierarchy["coarse_count"])),
            dtype=np.int64,
        )
        confusion_b = np.zeros(
            (int(dual_hierarchy["fine_per_coarse"]), int(dual_hierarchy["fine_per_coarse"])),
            dtype=np.int64,
        )
    confusion_sum = np.zeros((int(state["output_dim"]), int(state["output_dim"])), dtype=np.int64)

    with torch.no_grad():
        for x_a, x_b, y in state["eval_loader"]:
            x_a = x_a.to(state["device"])
            x_b = x_b.to(state["device"])
            y = y.to(state["device"])
            outputs = _forward_two_wavelengths(
                model,
                x_a,
                x_b,
                wavelength_a_um=wl_a,
                wavelength_b_um=wl_b,
                objective_cfg=objective_cfg,
                device=state["device"],
                dual_hierarchy=dual_hierarchy,
            )
            losses = _compute_losses(
                outputs,
                y,
                loss_fn=loss_fn,
                branch_weights=active_settings["branch_loss_weights"],
                combined_weight=active_settings["combined_loss_weight"],
                dual_hierarchy=dual_hierarchy,
                include_combined_in_total_loss=bool(active_settings["include_combined_in_total_loss"]),
            )
            batch = x_a.shape[0]
            total_count += batch
            total_loss += float(losses["loss_total"].item()) * batch
            total_loss_a += float(losses["loss_a"].item()) * batch
            total_loss_b += float(losses["loss_b"].item()) * batch
            total_loss_sum += float(losses["loss_sum"].item()) * batch
            if dual_hierarchy is None:
                correct_a += _accuracy_from_predictions(outputs["pred_a"], y)
                correct_b += _accuracy_from_predictions(outputs["pred_b"], y)
                correct_sum += _accuracy_from_predictions(outputs["pred_sum"], y)

                target_idx = _class_indices_from_one_hot(y)
                confusion_a += _confusion_matrix(outputs["pred_a"].argmax(dim=-1), target_idx, int(state["output_dim"]))
                confusion_b += _confusion_matrix(outputs["pred_b"].argmax(dim=-1), target_idx, int(state["output_dim"]))
                confusion_sum += _confusion_matrix(outputs["pred_sum"].argmax(dim=-1), target_idx, int(state["output_dim"]))
            else:
                hierarchy_targets = _extract_dual_hierarchy_targets(y, dual_hierarchy)
                pred_b_local, branch_b_mask = _resolve_branch_b_task_predictions(
                    outputs,
                    hierarchy_targets,
                    dual_hierarchy,
                )
                pred_a_idx = outputs["pred_a"].argmax(dim=-1)
                pred_b_idx = pred_b_local.argmax(dim=-1)
                pred_sum_idx = outputs["pred_sum"].argmax(dim=-1)
                correct_a += int((pred_a_idx == hierarchy_targets["coarse_idx"]).sum().item())
                if branch_b_mask is None:
                    correct_b += int((pred_b_idx == hierarchy_targets["local_idx"]).sum().item())
                    total_count_b += batch
                else:
                    total_count_b += int(branch_b_mask.sum().item())
                    if bool(branch_b_mask.any().item()):
                        correct_b += int(
                            (
                                pred_b_idx[branch_b_mask]
                                == hierarchy_targets["local_idx"][branch_b_mask]
                            ).sum().item()
                        )
                correct_sum += int((pred_sum_idx == hierarchy_targets["fine_idx"]).sum().item())

                confusion_a += _confusion_matrix(
                    pred_a_idx,
                    hierarchy_targets["coarse_idx"],
                    int(dual_hierarchy["coarse_count"]),
                )
                if branch_b_mask is None:
                    confusion_b += _confusion_matrix(
                        pred_b_idx,
                        hierarchy_targets["local_idx"],
                        int(dual_hierarchy["fine_per_coarse"]),
                    )
                elif bool(branch_b_mask.any().item()):
                    confusion_b += _confusion_matrix(
                        pred_b_idx[branch_b_mask],
                        hierarchy_targets["local_idx"][branch_b_mask],
                        int(dual_hierarchy["fine_per_coarse"]),
                    )
                confusion_sum += _confusion_matrix(
                    pred_sum_idx,
                    hierarchy_targets["fine_idx"],
                    int(state["output_dim"]),
                )

        reference_outputs = _forward_two_wavelengths(
            model,
            state["reference_input_a"],
            state["reference_input_b"],
            wavelength_a_um=wl_a,
            wavelength_b_um=wl_b,
            objective_cfg=objective_cfg,
            device=state["device"],
            dual_hierarchy=dual_hierarchy,
        )

    flattened = _flatten_reference_branch_predictions(reference_outputs)
    eval_accuracy_a = correct_a / max(total_count, 1)
    eval_accuracy_b = correct_b / max(total_count_b, 1)
    eval_accuracy_sum = correct_sum / max(total_count, 1)
    eval_accuracy_branch_mean = 0.5 * (eval_accuracy_a + eval_accuracy_b)
    metrics = {
        "eval_loss": total_loss / max(total_count, 1),
        f"eval_loss_{wl_key_a}": total_loss_a / max(total_count, 1),
        f"eval_loss_{wl_key_b}": total_loss_b / max(total_count, 1),
        "eval_loss_combined": total_loss_sum / max(total_count, 1),
        "eval_accuracy": eval_accuracy_sum,
        f"eval_accuracy_{wl_key_a}": eval_accuracy_a,
        f"eval_accuracy_{wl_key_b}": eval_accuracy_b,
        "eval_accuracy_branch_mean": eval_accuracy_branch_mean,
        "eval_accuracy_combined": eval_accuracy_sum,
        "selection_metric": _resolve_dual_selection_metric(
            mode=str(active_settings["selection_metric_mode"]),
            eval_accuracy_a=eval_accuracy_a,
            eval_accuracy_b=eval_accuracy_b,
            eval_accuracy_sum=eval_accuracy_sum,
        ),
        "phase_name": active_settings["phase_name"],
        "phase_epoch": int(active_settings["phase_epoch"]),
        "raw_output": flattened["raw_output"],
        "transformed_output": flattened["transformed_output"],
        "target_output": list(state["reference_target"]),
        "reference_prediction_by_wavelength": {
            wl_key_a: reference_outputs["pred_a"].detach().cpu().reshape(-1).tolist(),
            wl_key_b: reference_outputs["pred_b"].detach().cpu().reshape(-1).tolist(),
        },
        "reference_raw_prediction_by_wavelength": {
            wl_key_a: reference_outputs["raw_a"].detach().cpu().reshape(-1).tolist(),
            wl_key_b: reference_outputs["raw_b"].detach().cpu().reshape(-1).tolist(),
        },
        "reference_prediction_combined": reference_outputs["pred_sum"].detach().cpu().reshape(-1).tolist(),
        "reference_raw_prediction_combined": reference_outputs["raw_sum"].detach().cpu().reshape(-1).tolist(),
        "confusion_matrix": confusion_sum.tolist(),
        "confusion_matrix_by_wavelength": {
            wl_key_a: confusion_a.tolist(),
            wl_key_b: confusion_b.tolist(),
        },
    }
    return _update_exploration_state_after_eval(state, epoch=epoch, metrics=metrics)


def _save_reference_field_summary(
    *,
    last_field_a: torch.Tensor,
    last_field_b: torch.Tensor,
    out_dir: Path,
) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    intensity_a = np.abs(last_field_a.detach().cpu().numpy().reshape(-1)) ** 2
    intensity_b = np.abs(last_field_b.detach().cpu().numpy().reshape(-1)) ** 2
    intensity_sum = intensity_a + intensity_b

    intensity_sum_path = out_dir / "reference_last_layer_intensity_sum.npy"
    np.save(intensity_sum_path, intensity_sum)

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    axis = np.arange(intensity_sum.size)
    ax.plot(axis, intensity_a, linewidth=1.1, label="lambda_a")
    ax.plot(axis, intensity_b, linewidth=1.1, label="lambda_b")
    ax.plot(axis, intensity_sum, linewidth=1.4, label="lambda_a + lambda_b")
    ax.set_xlabel("lateral sample index")
    ax.set_ylabel("intensity")
    ax.set_title("Reference Last-Layer Intensity")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    figure_path = out_dir / "reference_last_layer_intensity_sum.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return {
        "reference_last_layer_intensity_sum_path": str(intensity_sum_path),
        "reference_last_layer_intensity_sum_figure": str(figure_path),
    }


def dump_reference_artifacts(
    state: Mapping[str, Any],
    *,
    train_dir: str | Path,
) -> Dict[str, Any]:
    train_dir = Path(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    if state.get("training_mode") == "single":
        wl = float(state["task_wavelengths_um"][0])
        wl_key = str(state["task_wavelength_keys"][0])

        for stale_name in (
            "reference_input_lambda_a.npy",
            "reference_input_lambda_b.npy",
            "reference_output_lambda_a.npy",
            "reference_output_lambda_b.npy",
            "reference_output_combined.npy",
            "reference_last_layer_intensity_sum.npy",
            "reference_last_layer_intensity_sum.png",
        ):
            stale_path = train_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()

        state["model"].eval()
        with torch.no_grad():
            outputs = _forward_single_wavelength(
                state["model"],
                state["reference_input"],
                wavelength_um=wl,
                objective_cfg=state["objective_cfg"],
                device=state["device"],
                capture_fields=True,
            )

        ref_input_path = train_dir / "reference_input.npy"
        ref_output_path = train_dir / "reference_output.npy"
        ref_output_combined_path = train_dir / "reference_output_combined.npy"
        labels_path = train_dir / "reference_output_labels.json"

        np.save(ref_input_path, state["reference_input"].detach().cpu().numpy())
        np.save(ref_output_path, outputs["pred"].detach().cpu().numpy())
        np.save(ref_output_combined_path, outputs["pred"].detach().cpu().numpy())

        field_root = train_dir / "reference_fields"
        field_root.mkdir(parents=True, exist_ok=True)
        field_paths: list[str] = []
        wavelength_dir = field_root / wl_key
        wavelength_dir.mkdir(parents=True, exist_ok=True)
        if outputs["fields"] is not None:
            for layer_idx, field in enumerate(outputs["fields"]):
                layer_path = wavelength_dir / f"layer_{layer_idx}.npy"
                np.save(layer_path, torch.as_tensor(field).detach().cpu().numpy())
                field_paths.append(str(layer_path))

        labels_path.write_text(
            json.dumps(
                {
                    "task_wavelengths_um": list(state["task_wavelengths_um"]),
                    "flattened_output_layout": [
                        f"{wl_key}_port_{port_idx}" for port_idx in range(int(state["output_dim"]))
                    ],
                    "combined_output_layout": [
                        f"combined_port_{port_idx}" for port_idx in range(int(state["output_dim"]))
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "reference_input_path": str(ref_input_path),
            "reference_output_path": str(ref_output_path),
            "reference_output_combined_path": str(ref_output_combined_path),
            "reference_output_labels_path": str(labels_path),
            "reference_field_paths": field_paths,
        }

    wl_a, wl_b = state["task_wavelengths_um"]
    wl_key_a, wl_key_b = state["task_wavelength_keys"]
    dual_hierarchy = state.get("dual_hierarchy")

    state["model"].eval()
    with torch.no_grad():
        outputs = _forward_two_wavelengths(
            state["model"],
            state["reference_input_a"],
            state["reference_input_b"],
            wavelength_a_um=wl_a,
            wavelength_b_um=wl_b,
            objective_cfg=state["objective_cfg"],
            device=state["device"],
            dual_hierarchy=dual_hierarchy,
            capture_fields=True,
        )

    ref_input_a_path = train_dir / "reference_input_lambda_a.npy"
    ref_input_b_path = train_dir / "reference_input_lambda_b.npy"
    ref_output_a_path = train_dir / "reference_output_lambda_a.npy"
    ref_output_b_path = train_dir / "reference_output_lambda_b.npy"
    ref_output_sum_path = train_dir / "reference_output_combined.npy"
    labels_path = train_dir / "reference_output_labels.json"

    np.save(ref_input_a_path, state["reference_input_a"].detach().cpu().numpy())
    np.save(ref_input_b_path, state["reference_input_b"].detach().cpu().numpy())
    np.save(ref_output_a_path, outputs["pred_a"].detach().cpu().numpy())
    np.save(ref_output_b_path, outputs["pred_b"].detach().cpu().numpy())
    np.save(ref_output_sum_path, outputs["pred_sum"].detach().cpu().numpy())

    field_root = train_dir / "reference_fields"
    field_root.mkdir(parents=True, exist_ok=True)
    field_paths: list[str] = []
    for field_key, fields in ((wl_key_a, outputs["fields_a"]), (wl_key_b, outputs["fields_b"])):
        wavelength_dir = field_root / field_key
        wavelength_dir.mkdir(parents=True, exist_ok=True)
        if fields is None:
            continue
        for layer_idx, field in enumerate(fields):
            layer_path = wavelength_dir / f"layer_{layer_idx}.npy"
            np.save(layer_path, torch.as_tensor(field).detach().cpu().numpy())
            field_paths.append(str(layer_path))

    layout_by_wavelength = dict(state.get("reference_output_layout_by_wavelength") or {})
    layout_a = list(
        layout_by_wavelength.get(
            wl_key_a,
            [f"port_{port_idx}" for port_idx in range(int(outputs["pred_a"].shape[-1]))],
        )
    )
    layout_b = list(
        layout_by_wavelength.get(
            wl_key_b,
            [f"port_{port_idx}" for port_idx in range(int(outputs["pred_b"].shape[-1]))],
        )
    )
    combined_layout = list(
        state.get("reference_combined_output_layout")
        or [f"combined_port_{port_idx}" for port_idx in range(int(outputs["pred_sum"].shape[-1]))]
    )
    labels_path.write_text(
        json.dumps(
            {
                "task_wavelengths_um": list(state["task_wavelengths_um"]),
                "output_layout_by_wavelength": {
                    wl_key_a: layout_a,
                    wl_key_b: layout_b,
                },
                "flattened_output_layout": [f"{wl_key_a}:{label}" for label in layout_a]
                + [f"{wl_key_b}:{label}" for label in layout_b],
                "combined_output_layout": combined_layout,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    extra = {}
    if outputs["fields_a"] and outputs["fields_b"]:
        extra = _save_reference_field_summary(
            last_field_a=torch.as_tensor(outputs["fields_a"][-1]),
            last_field_b=torch.as_tensor(outputs["fields_b"][-1]),
            out_dir=train_dir,
        )

    return {
        "reference_input_lambda_a_path": str(ref_input_a_path),
        "reference_input_lambda_b_path": str(ref_input_b_path),
        "reference_output_lambda_a_path": str(ref_output_a_path),
        "reference_output_lambda_b_path": str(ref_output_b_path),
        "reference_output_combined_path": str(ref_output_sum_path),
        "reference_output_labels_path": str(labels_path),
        "reference_field_paths": field_paths,
        **extra,
    }


def summarize_training(
    state: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    *,
    best_epoch: int,
    best_metric: float,
) -> Dict[str, Any]:
    last = dict(history[-1])
    best_record = dict(history[best_epoch - 1])
    exploration_state = state.get("exploration_state")
    if state.get("training_mode") == "single":
        wl_key = str(state["task_wavelength_keys"][0])
        best_accuracy_record = max(history, key=lambda record: float(record.get("eval_accuracy", 0.0)))
        summary = {
            "task_kind": state["task_kind"],
            "dataset_kind": state["dataset_kind"],
            "epochs": int(len(history)),
            "best_epoch": int(best_epoch),
            "best_selection_metric": float(best_metric),
            "best_eval_accuracy": float(best_accuracy_record.get("eval_accuracy", 0.0)),
            "best_eval_accuracy_combined": float(best_accuracy_record.get("eval_accuracy", 0.0)),
            "best_eval_accuracy_combined_epoch": int(best_accuracy_record["epoch"]),
            f"best_eval_accuracy_{wl_key}": float(best_record.get(f"eval_accuracy_{wl_key}", 0.0)),
            "final_eval_accuracy": float(last.get("eval_accuracy", 0.0)),
            "final_eval_accuracy_combined": float(last.get("eval_accuracy", 0.0)),
            f"final_eval_accuracy_{wl_key}": float(last.get(f"eval_accuracy_{wl_key}", 0.0)),
            "final_eval_loss": float(last["eval_loss"]),
            "train_samples": int(state["train_samples"]),
            "eval_samples": int(state["eval_samples"]),
            "input_dim_per_wavelength": int(state["input_dim_per_wavelength"]),
            "output_dim": int(state["output_dim"]),
            "monitor_samples_used": int(state.get("monitor_samples_used", 0)),
            "requested_monitor_downsample_to": state.get("requested_monitor_downsample_to"),
            "class_names": list(state["class_names"]),
            "pca_explained_variance_ratio_sum": float(state["pca_explained_variance_ratio_sum"]),
            "dataset_summary": copy.deepcopy(dict(state.get("dataset_summary") or {})),
            "task_wavelengths_um": list(state["task_wavelengths_um"]),
            "reference_target": list(state["reference_target"]),
            "reference_prediction_by_wavelength": dict(best_record["reference_prediction_by_wavelength"]),
            "reference_prediction_combined": list(best_record["reference_prediction_combined"]),
            "decision_rule": "predicted_class = argmax(port_normalized_output)",
            "decision_rule_combined": "single-wavelength output is reused as combined output for compatibility",
            "final_confusion_matrix_combined": last.get("confusion_matrix"),
            "final_confusion_matrix_by_wavelength": last.get("confusion_matrix_by_wavelength"),
        }
        if isinstance(exploration_state, Mapping) and bool(exploration_state.get("enabled", False)):
            summary["exploration"] = {
                "enabled": True,
                "metric": str(exploration_state["metric"]),
                "goal": str(exploration_state["goal"]),
                "events_applied": int(exploration_state.get("events_applied", 0)),
                "best_metric_value": (
                    None
                    if exploration_state.get("best_value") is None
                    else float(exploration_state["best_value"])
                ),
                "best_metric_epoch": int(exploration_state.get("best_epoch", 0)),
                "last_applied_epoch": int(exploration_state.get("last_applied_epoch", 0)),
                "last_applied_std_um": float(exploration_state.get("last_applied_std_um", 0.0)),
            }
        return summary

    wl_key_a, wl_key_b = state["task_wavelength_keys"]
    best_combined_record = max(history, key=lambda record: float(record.get("eval_accuracy_combined", 0.0)))
    best_branch_mean_record = max(history, key=lambda record: float(record.get("eval_accuracy_branch_mean", 0.0)))
    dual_hierarchy = state.get("dual_hierarchy")
    summary = {
        "task_kind": state["task_kind"],
        "dataset_kind": state["dataset_kind"],
        "epochs": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_selection_metric": float(best_metric),
        "best_eval_accuracy_branch_mean": float(best_branch_mean_record.get("eval_accuracy_branch_mean", 0.0)),
        "best_eval_accuracy_branch_mean_epoch": int(best_branch_mean_record["epoch"]),
        "best_eval_accuracy_combined": float(best_combined_record.get("eval_accuracy_combined", 0.0)),
        "best_eval_accuracy_combined_epoch": int(best_combined_record["epoch"]),
        f"final_eval_accuracy_branch_mean": float(last.get("eval_accuracy_branch_mean", 0.0)),
        f"best_eval_accuracy_{wl_key_a}": float(best_record.get(f"eval_accuracy_{wl_key_a}", 0.0)),
        f"best_eval_accuracy_{wl_key_b}": float(best_record.get(f"eval_accuracy_{wl_key_b}", 0.0)),
        "final_eval_accuracy_combined": float(last.get("eval_accuracy_combined", 0.0)),
        f"final_eval_accuracy_{wl_key_a}": float(last.get(f"eval_accuracy_{wl_key_a}", 0.0)),
        f"final_eval_accuracy_{wl_key_b}": float(last.get(f"eval_accuracy_{wl_key_b}", 0.0)),
        "final_eval_loss": float(last["eval_loss"]),
        "train_samples": int(state["train_samples"]),
        "eval_samples": int(state["eval_samples"]),
        "input_dim_per_wavelength": int(state["input_dim_per_wavelength"]),
        "output_dim": int(state["output_dim"]),
        "monitor_samples_used": int(state.get("monitor_samples_used", 0)),
        "requested_monitor_downsample_to": state.get("requested_monitor_downsample_to"),
        "class_names": list(state["class_names"]),
        "pca_explained_variance_ratio_sum": float(state["pca_explained_variance_ratio_sum"]),
        "dataset_summary": copy.deepcopy(dict(state.get("dataset_summary") or {})),
        "task_wavelengths_um": list(state["task_wavelengths_um"]),
        "reference_target": list(state["reference_target"]),
        "reference_prediction_by_wavelength": dict(best_record["reference_prediction_by_wavelength"]),
        "reference_prediction_combined": list(best_record["reference_prediction_combined"]),
        "decision_rule": (
            "predicted_class = argmax(port_normalized_output)"
            if dual_hierarchy is None
            else (
                (
                    f"{wl_key_a}: argmax over grouped coarse probabilities; "
                    f"{wl_key_b}: argmax over dedicated 3-port local head "
                    f"restricted to coarse='{dual_hierarchy['branch_b_focus_coarse_name']}'"
                )
                if bool(dual_hierarchy.get("branch_b_focus_local_ports_only", False))
                else (
                    (
                    f"{wl_key_a}: argmax over grouped coarse probabilities; "
                    f"{wl_key_b}: argmax over coarse-guided local probabilities"
                )
                if bool(dual_hierarchy.get("branch_b_uses_coarse_context", False))
                else (
                    (
                        f"{wl_key_a}: argmax over grouped coarse probabilities; "
                        f"{wl_key_b}: argmax over within-coarse conditional probabilities "
                        f"restricted to coarse='{dual_hierarchy['branch_b_focus_coarse_name']}'"
                    )
                    if dual_hierarchy.get("branch_b_focus_coarse_name") is not None
                    else (
                        f"{wl_key_a}: argmax over grouped coarse probabilities; "
                        f"{wl_key_b}: argmax over within-coarse conditional probabilities using the true coarse group"
                    )
                )
                )
            )
        ),
        "decision_rule_combined": (
            "combined output uses port-wise summed raw responses from the two wavelengths, then port_normalized"
            if dual_hierarchy is None
            else (
                (
                    f"combined fine probability = P(coarse|{wl_key_a}) * "
                    f"P(fine within coarse='{dual_hierarchy['branch_b_focus_coarse_name']}'|{wl_key_b}); "
                    "non-focused coarse groups fall back to uniform within-group fine probabilities"
                )
                if bool(dual_hierarchy.get("branch_b_focus_local_ports_only", False))
                else (
                f"combined fine probability = P(coarse|{wl_key_a}) * "
                f"P(fine within coarse|{wl_key_b})"
                )
            )
        ),
        "final_confusion_matrix_combined": last.get("confusion_matrix"),
        "final_confusion_matrix_by_wavelength": last.get("confusion_matrix_by_wavelength"),
    }
    if dual_hierarchy is not None:
        summary["dual_hierarchy"] = copy.deepcopy(dict(dual_hierarchy))
    if state.get("phase_schedule") is not None:
        summary["phase_schedule"] = copy.deepcopy(list(state["phase_schedule"]))
    if isinstance(exploration_state, Mapping) and bool(exploration_state.get("enabled", False)):
        summary["exploration"] = {
            "enabled": True,
            "metric": str(exploration_state["metric"]),
            "goal": str(exploration_state["goal"]),
            "events_applied": int(exploration_state.get("events_applied", 0)),
            "best_metric_value": (
                None
                if exploration_state.get("best_value") is None
                else float(exploration_state["best_value"])
            ),
            "best_metric_epoch": int(exploration_state.get("best_epoch", 0)),
            "last_applied_epoch": int(exploration_state.get("last_applied_epoch", 0)),
            "last_applied_std_um": float(exploration_state.get("last_applied_std_um", 0.0)),
        }
    return summary


__all__ = [
    "build_training_state",
    "train_one_epoch",
    "evaluate_model",
    "dump_reference_artifacts",
    "summarize_training",
]
