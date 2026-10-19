"""Train one physical slot structure for joint 1310/1610 nm nine-class inference.

The two carriers have equal input power, encode the same sample and contribute
incoherently to the same class ports. Only one (3, 555) slot-length tensor is
optimized. Propagating the 16 input basis modes gives exactly the same linear
field operator as sample-wise propagation, with fewer FFTs per training step.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.nn import functional as F

from run_comparison import (BASE, HERE, PROJECT, CAL, CONFIG, FINE,
                            CalibratedDONN, dataset, probabilities, save_json, sha)

OUT = BASE / "joint_fine9_shared_20261019"
WAVELENGTHS = (1310, 1610)
POWER_FRACTIONS = (.5, .5)


class SharedDualDONN(nn.Module):
    def __init__(self, seed=42, n=10000, pad=2, slots=None):
        super().__init__()
        self.channels = nn.ModuleList([
            CalibratedDONN(wl, seed=seed, n=n, pad=pad, slots=slots)
            for wl in WAVELENGTHS
        ])
        # Both references point to one Parameter, not two independently trained masks.
        self.channels[1].lengths = self.channels[0].lengths
        assert len(list(self.parameters())) == 1
        assert self.lengths.numel() == 1665
        assert self.channels[0].lengths is self.channels[1].lengths

    @property
    def lengths(self):
        return self.channels[0].lengths

    def transfer_matrices(self):
        """Return shape (wavelength, input, output), for unit-power input modes."""
        matrices = []
        for channel in self.channels:
            masks = channel.masks()
            u = F.pad(channel.input_modes.T,
                      (channel.start, channel.n*channel.pad-channel.n-channel.start))
            for gap in range(4):
                if gap:
                    u = u * masks[gap-1]
                u = channel.propagate(u)
            matrix = u[:, channel.start:channel.start+channel.n] @ channel.output_modes.conj() * channel.dy
            matrices.append(matrix)
        return torch.stack(matrices)

    def forward(self, x, matrices=None):
        matrices = self.transfer_matrices() if matrices is None else matrices
        amplitudes = torch.polar(torch.ones_like(x), np.pi*x) / 4
        # Per-channel powers here use unit input power. Fusion weights ensure
        # TOTAL two-carrier input power is one, matching single-carrier baselines.
        fields = torch.einsum("bi,wio->bwo", amplitudes, matrices)
        channel_ports = fields.abs().square()
        fused = .5*channel_ports[:, 0] + .5*channel_ports[:, 1]
        return fused, channel_ports


def metrics(ports, truth):
    p = ports[:, :9] / np.maximum(ports[:, :9].sum(1, keepdims=True), 1e-12)
    pred = p.argmax(1)
    record = dict(accuracy=float((pred == truth).mean()),
                  ce=float(-np.log(np.maximum(p[np.arange(len(truth)), truth], 1e-12)).mean()),
                  mean_active_collection_fraction=float(ports[:, :9].sum(1).mean()),
                  mean_10port_collection_fraction=float(ports.sum(1).mean()))
    return record, p


@torch.no_grad()
def evaluate_joint(model, x, truth):
    model.eval()
    matrices = model.transfer_matrices()
    fused, channels = model(x, matrices=matrices)
    fused, channels = fused.numpy(), channels.numpy()
    result, p = metrics(fused, truth)
    ablations = {}
    for index, wl in enumerate(WAVELENGTHS):
        ablations[str(wl)] = metrics(channels[:, index], truth)[0]
    return result, p, fused, channels, ablations, matrices.numpy()


def preflight(xt, truth):
    model = SharedDualDONN()
    x, y = xt[:5], torch.as_tensor(truth[:5])
    fused, channel_ports = model(x)
    reference = torch.stack([channel(x) for channel in model.channels], dim=1)
    torch.testing.assert_close(channel_ports, reference, rtol=5e-4, atol=2e-8)
    fast_loss = F.nll_loss(probabilities(fused, 9).clamp_min(1e-12).log(), y)
    fast_grad = torch.autograd.grad(fast_loss, model.lengths)[0]
    reference_loss = F.nll_loss(probabilities(reference.mean(1), 9).clamp_min(1e-12).log(), y)
    reference_grad = torch.autograd.grad(reference_loss, model.lengths)[0]
    torch.testing.assert_close(fast_grad, reference_grad, rtol=3e-3, atol=2e-5)
    grad_relative_error = float((fast_grad-reference_grad).norm()/reference_grad.norm())
    if not torch.isfinite(fast_grad).all() or grad_relative_error > 1e-3:
        raise AssertionError(f"Invalid shared-structure gradient: relative error {grad_relative_error}")
    branch_norms, energy = {}, {}
    for index, channel in enumerate(model.channels):
        branch = channel(x)
        target_power_loss = -branch[torch.arange(len(y)), y].clamp_min(1e-12).log().mean()
        grad = torch.autograd.grad(target_power_loss, model.lengths)[0]
        if not torch.isfinite(grad).all() or float(grad.norm()) <= 0:
            raise AssertionError(f"Missing branch gradient at {channel.wl_nm} nm")
        branch_norms[str(channel.wl_nm)] = float(grad.norm())
        with torch.no_grad():
            _, fields = channel(x, trace=True)
            energies = np.array([(u.abs().square().sum(1)*channel.dy).numpy() for u in fields])
            np.testing.assert_allclose(energies[0], 1., atol=2e-6)
            if np.any(np.diff(energies, axis=0) > 1e-5):
                raise AssertionError("Passive propagation unexpectedly increased scalar power")
            energy[str(channel.wl_nm)] = energies.tolist()
    np.testing.assert_allclose(fused.detach().numpy(), reference.detach().mean(1).numpy(), rtol=5e-4, atol=2e-8)
    record = dict(shared_parameter_identity=True, trainable_slot_lengths=1665,
                  input_power_fractions=list(POWER_FRACTIONS), total_input_power=1.,
                  basis_vs_sample_max_port_abs_error=float((channel_ports-reference).detach().abs().max()),
                  basis_vs_sample_gradient_relative_error=grad_relative_error,
                  branch_gradient_norms=branch_norms, per_unit_input_plane_energies=energy,
                  fusion="sum physical powers before normalization; no cross-wavelength field interference")
    save_json(OUT / "preflight.json", record)
    print("PREFLIGHT " + json.dumps({k:v for k,v in record.items() if k != "per_unit_input_plane_energies"}), flush=True)


def train(seed, epochs, arrays, config):
    destination = OUT / f"seed{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.json"
    if summary_path.exists():
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if saved["experiment_config"] != config:
            raise RuntimeError(f"Completed run has different configuration: {destination}")
        print(f"SKIP completed {destination}", flush=True)
        return
    xt, xv, yt, yv = arrays
    torch.manual_seed(seed)
    model = SharedDualDONN(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0008, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=.00003)
    generator = torch.Generator().manual_seed(seed+1000)
    history, best_key, best_epoch = [], (-1., -float("inf")), 0
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        total_loss, correct = 0., 0
        permutation = torch.randperm(len(yt), generator=generator)
        for indices in permutation.split(64):
            x = xt[indices] + torch.randn(xt[indices].shape, generator=generator)*.005
            optimizer.zero_grad(set_to_none=True)
            ports, _ = model(x)
            p = probabilities(ports, 9)
            loss = F.nll_loss(p.clamp_min(1e-12).log(), yt[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at seed {seed}, epoch {epoch}")
            loss.backward()
            if model.lengths.grad is None or not torch.isfinite(model.lengths.grad).all():
                raise FloatingPointError(f"Invalid shared slot gradients at seed {seed}, epoch {epoch}")
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            with torch.no_grad():
                model.lengths.clamp_(0., 2.)
            total_loss += float(loss.detach())*len(indices)
            correct += int((p.argmax(1) == yt[indices]).sum())
        validation, _, _, _, branches, _ = evaluate_joint(model, xv, yv.numpy())
        row = dict(epoch=epoch, train_ce=total_loss/len(yt), train_noisy_accuracy=correct/len(yt),
                   val_accuracy=validation["accuracy"], val_ce=validation["ce"],
                   val_active_collection_fraction=validation["mean_active_collection_fraction"],
                   val_1310_only_accuracy=branches["1310"]["accuracy"],
                   val_1610_only_accuracy=branches["1610"]["accuracy"],
                   learning_rate=optimizer.param_groups[0]["lr"], elapsed_s=time.perf_counter()-started)
        history.append(row)
        key = (validation["accuracy"], -validation["ce"])
        if key > best_key:
            best_key, best_epoch = key, epoch
            torch.save(dict(lengths_um=model.lengths.detach().clone(), seed=seed, epoch=epoch,
                            experiment_config=config), destination / "best_model.pt")
        save_json(destination / "history.json", history)
        print(json.dumps(dict(task="joint_fine9_shared", seed=seed, **row)), flush=True)
        scheduler.step()
    torch.save(dict(lengths_um=model.lengths.detach().clone(), seed=seed, epoch=epochs,
                    experiment_config=config), destination / "final_model.pt")
    final = evaluate_joint(model, xv, yv.numpy())[0]
    checkpoint = torch.load(destination / "best_model.pt", weights_only=False)
    with torch.no_grad():
        model.lengths.copy_(checkpoint["lengths_um"])
    best, p, ports, branch_ports, ablations, matrices = evaluate_joint(model, xv, yv.numpy())
    training = evaluate_joint(model, xt, yt.numpy())[0]
    prediction = p.argmax(1)
    np.savez_compressed(destination / "validation_predictions.npz", probabilities=p, ports=ports,
                        channel_ports_unit_input=branch_ports, truth=yv.numpy(), prediction=prediction,
                        input_power_fractions=np.array(POWER_FRACTIONS), wavelengths_nm=np.array(WAVELENGTHS))
    np.save(destination / "slot_lengths_um.npy", model.lengths.detach().numpy())
    # Export the conventional output-by-input matrix, b = T @ a.
    np.savez_compressed(destination / "complex_transfer_matrices.npz",
                        matrices=matrices.transpose(0, 2, 1), wavelengths_nm=np.array(WAVELENGTHS),
                        input_centers_um=np.arange(-90., 91., 12.), output_centers_um=np.arange(-90., 91., 20.),
                        slot_lengths_um=model.lengths.detach().numpy())
    a, b = matrices
    fidelity = float(abs(np.vdot(a, b))**2/(np.vdot(a, a).real*np.vdot(b, b).real))
    summary = dict(task="joint_fine9_shared", wavelengths_nm=list(WAVELENGTHS), seed=seed,
                   epochs=epochs, best_epoch=best_epoch, trainable_parameters=1665, shared_structure=True,
                   best_validation=best, final_validation=final, best_training=training,
                   same_structure_single_channel_ablations=ablations,
                   coarse_from_9ports_accuracy=float((p.reshape(-1, 3, 3).sum(2).argmax(1) == yv.numpy()//3).mean()),
                   classes=FINE, confusion_matrix=confusion_matrix(yv, prediction, labels=np.arange(9)).tolist(),
                   classification_report=classification_report(yv, prediction, labels=np.arange(9),
                                                               target_names=FINE, output_dict=True, zero_division=0),
                   complex_matrix_normalized_fidelity=fidelity,
                   slot_min_um=float(model.lengths.detach().min()), slot_max_um=float(model.lengths.detach().max()),
                   elapsed_s=time.perf_counter()-started, whole_device_fdtd=False, experiment_config=config)
    save_json(summary_path, summary)
    print("COMPLETED " + json.dumps({k:summary[k] for k in ("seed", "best_epoch", "best_validation", "elapsed_s")}), flush=True)


@torch.no_grad()
def convergence(seeds, arrays):
    x, truth = arrays[1], arrays[3].numpy()
    records = []
    for seed in seeds:
        destination = OUT / f"seed{seed}"
        slots = np.load(destination / "slot_lengths_um.npy")
        saved = np.load(destination / "validation_predictions.npz")
        settings = [(20000, 2)] + ([(10000, 4)] if seed == 42 else [])
        for n, pad in settings:
            model = SharedDualDONN(seed=seed, n=n, pad=pad, slots=slots)
            result, p, _, _, _, _ = evaluate_joint(model, x, truth)
            records.append(dict(seed=seed, grid_n=n, padding_factor=pad, accuracy=result["accuracy"],
                                accuracy_delta=float(result["accuracy"]-(saved["prediction"] == truth).mean()),
                                probability_mean_abs_delta=float(abs(p-saved["probabilities"]).mean()),
                                probability_max_abs_delta=float(abs(p-saved["probabilities"]).max()),
                                prediction_changes=int((p.argmax(1) != saved["prediction"]).sum())))
    save_json(OUT / "convergence.json", records)
    return records


def aggregate(seeds, arrays):
    truth = arrays[3].numpy()
    summaries = [json.loads((OUT/f"seed{s}/summary.json").read_text(encoding="utf-8")) for s in seeds]
    rows, baseline_checks = [], []
    for seed, joint in zip(seeds, summaries):
        row = dict(seed=seed, joint_accuracy=joint["best_validation"]["accuracy"], best_epoch=joint["best_epoch"],
                   joint_final_accuracy=joint["final_validation"]["accuracy"],
                   joint_active_collection_fraction=joint["best_validation"]["mean_active_collection_fraction"],
                   same_structure_1310_only=joint["same_structure_single_channel_ablations"]["1310"]["accuracy"],
                   same_structure_1610_only=joint["same_structure_single_channel_ablations"]["1610"]["accuracy"])
        saved_branches = []
        for wl in WAVELENGTHS:
            directory = HERE/f"fine9/seed{seed}/wl{wl}nm"
            baseline = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
            saved = np.load(directory/"validation_predictions.npz")
            np.testing.assert_array_equal(saved["truth"], truth)
            slots = np.load(directory/"slot_lengths_um.npy")
            model = SharedDualDONN(seed, slots=slots)
            with torch.no_grad():
                _, channels = model(arrays[1])
            current = channels[:, WAVELENGTHS.index(wl)].numpy()
            np.testing.assert_allclose(current, saved["ports"], rtol=1e-3, atol=3e-8)
            current_metrics, _ = metrics(current, truth)
            if current_metrics["accuracy"] != baseline["best_validation"]["accuracy"]:
                raise AssertionError(f"Baseline prediction mismatch for {wl} nm, seed {seed}")
            baseline_checks.append(dict(seed=seed, wavelength_nm=wl,
                                        max_port_abs_delta=float(abs(current-saved["ports"]).max()),
                                        accuracy_reproduced=True))
            row[f"separately_trained_{wl}_accuracy"] = baseline["best_validation"]["accuracy"]
            saved_branches.append(saved["ports"])
        # Reference ensemble needs TWO different slot structures (3330 parameters).
        ensemble, _ = metrics(.5*saved_branches[0]+.5*saved_branches[1], truth)
        row["two_structure_power_sum_reference"] = ensemble["accuracy"]
        rows.append(row)
    comparisons = {}
    for key in ("separately_trained_1310_accuracy", "separately_trained_1610_accuracy",
                "joint_accuracy", "same_structure_1310_only", "same_structure_1610_only",
                "two_structure_power_sum_reference", "joint_active_collection_fraction", "joint_final_accuracy"):
        values = [row[key] for row in rows]
        comparisons[key] = dict(mean=float(np.mean(values)), sample_sd=float(np.std(values, ddof=1)) if len(values)>1 else None,
                                per_seed=values)
    deltas = {}
    for wl in WAVELENGTHS:
        values = [r["joint_accuracy"]-r[f"separately_trained_{wl}_accuracy"] for r in rows]
        deltas[str(wl)] = dict(mean=float(np.mean(values)), per_seed=values)
    record = dict(seeds=seeds, runs=rows, comparisons=comparisons, joint_minus_separate_single=deltas,
                  baseline_reproduction=baseline_checks, trainable_parameters=1665, total_input_power=1.,
                  independent_test=False, shared_structure=True, whole_device_fdtd=False)
    save_json(OUT / "summary.json", record)
    plot_results(seeds, summaries, comparisons)
    return record


def plot_results(seeds, summaries, comparisons):
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    keys = ["separately_trained_1310_accuracy", "separately_trained_1610_accuracy", "joint_accuracy"]
    colors = ["#1679ab", "#d05b24", "#43894c"]
    for i, (key, color) in enumerate(zip(keys, colors)):
        item = comparisons[key]
        axes[0].bar(i, item["mean"]*100, color=color, alpha=.75)
        axes[0].errorbar(i, item["mean"]*100, yerr=(item["sample_sd"] or 0)*100, color="black", capsize=4)
        axes[0].scatter(np.linspace(i-.1, i+.1, len(seeds)), np.array(item["per_seed"])*100, c="black", s=17)
        axes[0].text(i, item["mean"]*100+1.2, f'{item["mean"]*100:.2f}%', ha="center")
    axes[0].set_xticks(range(3), ["1310 only\nseparate training", "1610 only\nseparate training", "Joint carriers\nshared slots"])
    axes[0].set(ylabel="Best validation accuracy (%)", ylim=(65, 95), title="Same encoder, 1665 trainable lengths")
    for seed in seeds:
        history = json.loads((OUT/f"seed{seed}/history.json").read_text())
        axes[1].plot([r["epoch"] for r in history], [r["val_accuracy"]*100 for r in history], label=f"seed {seed}")
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy (%)", title="Joint training histories")
    axes[1].legend()
    ablation_keys = ["same_structure_1310_only", "same_structure_1610_only", "joint_accuracy"]
    for i, (key, color) in enumerate(zip(ablation_keys, colors)):
        values = comparisons[key]
        axes[2].bar(i, values["mean"]*100, color=color, alpha=.75)
        axes[2].errorbar(i, values["mean"]*100, yerr=(values["sample_sd"] or 0)*100, color="black", capsize=4)
        axes[2].text(i, values["mean"]*100+1.2, f'{values["mean"]*100:.2f}%', ha="center")
    axes[2].set_xticks(range(3), ["1310 only", "1610 only", "Both"])
    axes[2].set(ylabel="Best-checkpoint validation accuracy (%)", ylim=(0, 100), title="Ablations of the SAME joint-trained slots")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("1310 + 1610 nm: equal total input power; calibrated scalar surrogate")
    fig.tight_layout()
    fig.savefig(OUT / "joint_comparison.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, len(seeds), figsize=(6*len(seeds), 5.8), squeeze=False)
    labels = [x.split("_", 1)[1] for x in FINE]
    for ax, summary in zip(axes[0], summaries):
        cm = np.array(summary["confusion_matrix"])
        ax.imshow(cm/140, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(9), labels, rotation=55, ha="right")
        ax.set_yticks(range(9), labels)
        for i in range(9):
            for j in range(9):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if cm[i, j]>70 else "black")
        ax.set(title=f'Shared dual, seed {summary["seed"]}: {summary["best_validation"]["accuracy"]*100:.2f}%',
               xlabel="Predicted", ylabel="True")
    fig.tight_layout()
    fig.savefig(OUT / "joint_confusion.png")
    plt.close(fig)


def write_report(record, checks):
    comp = record["comparisons"]
    def percent(key):
        item = comp[key]
        sd = f' ± {item["sample_sd"]*100:.2f} 百分点' if item["sample_sd"] is not None else ""
        return f'{item["mean"]*100:.2f}%{sd}'
    rows = "\n".join(f'| {r["seed"]} | {r["best_epoch"]} | {r["joint_accuracy"]*100:.2f}% | {r["same_structure_1310_only"]*100:.2f}% | {r["same_structure_1610_only"]*100:.2f}% |'
                     for r in record["runs"])
    max_delta = max(abs(r["accuracy_delta"]) for r in checks)
    report = f'''# 共享槽结构双波长联合九分类

日期：2026-10-19。该实验实际联合训练了一套共享槽长，同时利用 1310 nm、1610 nm 完成九分类。模型为局部电磁标定驱动的标量角谱代理，尚未进行完整器件 FDTD 或实物双波长验证。

## 联合方式

两个波长加载同一样本、相同的 16 维九类相位编码。在这个理想输入模型中，同一数值相位在两个载波上分别施加，未模拟共用移相器的波长相关驱动。两个分支共享一个 3×555 的槽长张量，共 1665 个可训练参数。各波长使用自己的有效折射率、槽复透射查表和传播算子。输入、输出模式沿用解析余弦模式。

各分支按单位输入功率计算 P_1310、P_1610，实际总输入功率固定为 1，各占 0.5。第 j 个端口的读出为 P_j = 0.5 P_1310,j + 0.5 P_1610,j，随后仅在前九个端口归一化，取最大者作为类别。不同光频率的复场不直接相加，不引入可训练电子分类器。假设慢速探测器平均掉载波拍频，且两波长探测响应相同；真实探测器的波长响应尚未标定。

同一套槽结构上的两个复传输矩阵均已保存。矩阵尺寸为 10×16，文件维度为“波长、输出、输入”，定义 b = T a。每步从 16 个输入基模式提取矩阵，再作用于样本；这利用线性叠加减少计算，预检查已与原逐样本传播核对端口功率及槽长梯度。

## 数据与训练

固定 7200 个训练样本、1260 个验证样本；九类各有 800/140 个样本。复用原正式九分类的训练集拟合编码 RBF384→LDA8+PCA8，两个载波接收同一份输入。未更换数据集，未重新选择特征；数据是合成可见光场景图，波长是计算载波。

种子：{record["seeds"]}。每次 20 轮，batch64，Adam lr=8e-4、weight_decay=1e-4，余弦最低学习率 3e-5，梯度裁剪 1，输入相位噪声 0.005π。槽长初始化 0.5±0.02 μm，限制在已标定的 0–2 μm。3 层、555 组/层、4 段 125 μm、500 μm 训练区域，10000 点、0.05 μm 网格，2 倍填充。最佳轮次按验证准确率优先、交叉熵次之选择。

## 实际结果

| 模型 | 最佳验证准确率，均值 ± 种子间样本标准差 |
| --- | --- |
| 单波长 1310 nm，原分别训练基线 | {percent("separately_trained_1310_accuracy")} |
| 单波长 1610 nm，原分别训练基线 | {percent("separately_trained_1610_accuracy")} |
| 双波长、共享槽长、联合训练 | {percent("joint_accuracy")} |
| 两套独立槽结构的固定等功率输出求和，仅作参考 | {percent("two_structure_power_sum_reference")} |

联合训练相对原 1310 nm 基线的平均变化为 {record["joint_minus_separate_single"]["1310"]["mean"]*100:+.3f} 百分点，相对原 1610 nm 为 {record["joint_minus_separate_single"]["1610"]["mean"]*100:+.3f} 百分点。两套结构求和参考需要 3330 个槽长，未作为共享单器件结果；该参考复用已有单波长检查点，没有另行训练。

| 种子 | 联合最佳轮次 | 双波长 | 同一联合结构仅开 1310 nm | 同一联合结构仅开 1610 nm |
| --- | --- | --- | --- | --- |
{rows}

联合模型最终轮次的平均准确率为 {percent("joint_final_accuracy")}。分支消融不重新训练或重新选检查点；关掉另一路后观察分类准确率，不能据此推断噪声或信噪比优势。

![联合与单波长比较](joint_comparison.png)

![三个种子的混淆矩阵](joint_confusion.png)

## 检查与限制

- 所有种子用 20000 点复核；seed42 另用 4 倍填充复核。最大准确率绝对变化为 {max_delta*100:.3f} 百分点，详见 convergence.json。
- 六个原单波长基线均在当前代码环境重算端口输出，与存档预测核对，准确率一致。
- 预检查验证两波长使用同一个 Parameter、只有 1665 个可训练量，两路对该参数均有有限非零梯度，且传播符合输入归一化与被动功率检查。
- 前九个端口的平均收光比例为 {comp["joint_active_collection_fraction"]["mean"]*100:.4f}%，相对于固定总输入功率。归一化交叉熵不约束效率，准确率不能代表高收光效率，也未建模探测噪声。
- 同一验证集用于检查点选择；只有三个种子，没有独立测试集，不作普遍优越性或统计显著性断言。
- 本次是双波长共同九分类，没有规定一个波长粗分、另一个细分，也没有建立两级路由。
- 当前编码经监督降维，不保留原图二维空间坐标；当前数据未保证细分类只能依赖局部细节。
- 局部周期槽查表忽略非局部槽耦合、反射往返、角度依赖和真实端口色散，需进一步全器件验证。

## 复现与文件

在项目根目录运行：

```powershell
python comparison_1310_1610_scene3_20261016/run_joint_fine9.py
```

默认跑三个种子、20 轮，并输出检查和报告；同配置的已完成结果会跳过。`--analysis-only` 从已完成检查点重新验证并生成报告；`--preflight-only` 只运行物理和梯度检查。完整复现需恢复原数据集；小型检查点、编码和标定数据随仓库保存。

- config.json：运行参数、输入及代码 SHA256、软件版本。
- preflight.json：共享参数、功率、梯度及矩阵法等价性检查。
- seed*/best_model.pt、final_model.pt、slot_lengths_um.npy：共享槽长检查点。
- seed*/history.json、summary.json：逐轮训练和最佳结果。
- seed*/validation_predictions.npz：联合概率、端口功率、各波长单位输入功率下的读出、真值。
- seed*/complex_transfer_matrices.npz：同一物理槽结构下两个波长的 10×16 复传输矩阵。
- convergence.json、summary.json：数值复核、基线重现、对比结果。
'''
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], choices=[42, 43, 44])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    data, names = dataset("fine9")
    assert names == FINE
    arrays = tuple(torch.as_tensor(a) for a in data)
    assert arrays[0].shape == (7200, 16) and arrays[1].shape == (1260, 16)
    np.testing.assert_array_equal(np.bincount(data[2], minlength=9), np.full(9, 800))
    np.testing.assert_array_equal(np.bincount(data[3], minlength=9), np.full(9, 140))
    sources = [Path(__file__), BASE / "run_comparison.py", HERE / "fine_encoded16.npz",
               HERE / "fine_encoding.json", PROJECT / "feature_upgrade_20261011/diagnostics.json"]
    sources += [CAL / f"wl{wl}nm/{name}" for wl in WAVELENGTHS
                for name in ("calibration_response.npz", "mode_probe.json")]
    config = {k:CONFIG[k] for k in ("width_um", "gap_um", "gaps", "trainable_layers", "groups_per_layer",
                                   "group_pitch_um", "slot_pitch_um", "repeats", "slot_width_um", "grid_n", "dy_um",
                                   "padding_factor", "input_centers_um", "input_width_um", "output_centers_um", "output_width_um",
                                   "slot_bounds_um", "interpolation", "epochs", "batch_size", "cpu_threads", "optimizer",
                                   "learning_rate", "weight_decay", "cosine_eta_min", "gradient_clip", "input_noise_pi",
                                   "initialization", "selection", "train_samples", "val_samples", "independent_test")}
    config.update(task="joint_fine9_shared", wavelengths_nm=list(WAVELENGTHS), input_power_fractions=list(POWER_FRACTIONS),
                  total_input_power=1., shared_structure=True, trainable_parameters=1665, whole_device_fdtd=False,
                  model="calibrated scalar angular spectrum with analytic input/output modes",
                  fusion="physical intensity sum before nine-port normalization; no learned electronic readout",
                  source_sha256={p.relative_to(PROJECT).as_posix():sha(p) for p in sources},
                  software=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__))
    if not args.analysis_only:
        existing = OUT / "config.json"
        if existing.exists() and json.loads(existing.read_text(encoding="utf-8"))["experiment"] != config:
            raise RuntimeError("Output directory contains another experiment configuration")
        save_json(existing, dict(experiment=config, seeds=args.seeds))
    preflight(arrays[0], data[2])
    if args.preflight_only:
        return
    if not args.analysis_only:
        for seed in args.seeds:
            train(seed, config["epochs"], arrays, config)
    checks = convergence(args.seeds, arrays)
    record = aggregate(args.seeds, arrays)
    write_report(record, checks)
    print("ALL COMPLETED " + json.dumps(record["comparisons"]), flush=True)


if __name__ == "__main__":
    main()
