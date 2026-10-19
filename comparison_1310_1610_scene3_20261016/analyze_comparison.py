"""Analysis of actual saved runs; no invented or extrapolated training results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

from run_comparison import (BASE, HERE, PROJECT, CAL, CONFIG, COARSE, FINE,
                            CalibratedDONN, dataset, evaluate, save_json, sha)

plt.rcParams.update({"font.size": 10, "figure.dpi": 140, "savefig.dpi": 180})
COLORS = {1310: "#1679ab", 1610: "#d05b24"}
OUT = HERE / "analysis"


def result_dir(task, seed, wl):
    return HERE / task / f"seed{seed}" / f"wl{wl}nm"


def checkpoint_model(task, seed, wl, **kwargs):
    slots = np.load(result_dir(task, seed, wl) / "slot_lengths_um.npy")
    return CalibratedDONN(wl, seed, slots=slots, **kwargs)


def paired_comparison(a, b, truth):
    ca, cb = a == truth, b == truth
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return dict(only_1310_correct=only_a, only_1610_correct=only_b,
                paired_delta_1610_minus_1310=float(cb.mean()-ca.mean()),
                exact_mcnemar_p=float(binomtest(only_a, only_a+only_b, .5).pvalue) if only_a+only_b else 1.)


def aggregate():
    rows, comparison = [], {}
    for task in ("coarse3", "fine9"):
        stats = {}
        for wl in (1310, 1610):
            records = [json.loads((result_dir(task, s, wl) / "summary.json").read_text(encoding="utf-8")) for s in (42, 43, 44)]
            for r in records:
                rows.append(dict(task=task, wavelength_nm=wl, seed=r["seed"], best_epoch=r["best_epoch"],
                                 best_validation_accuracy=r["best_validation"]["accuracy"], final_validation_accuracy=r["final_validation"]["accuracy"],
                                 validation_ce=r["best_validation"]["ce"], active_collection_fraction=r["best_validation"]["mean_active_collection_fraction"],
                                 **({k:r[k] for k in ("within_true_coarse_fine_accuracy", "coarse_from_9ports_accuracy")} if task == "fine9" else {})))
            accuracies = [r["best_validation"]["accuracy"] for r in records]
            stats[str(wl)] = dict(mean_accuracy=float(np.mean(accuracies)), sample_sd=float(np.std(accuracies, ddof=1)), seed_accuracies=accuracies,
                                 final_epoch_mean_accuracy=float(np.mean([r["final_validation"]["accuracy"] for r in records])),
                                 mean_active_collection_fraction=float(np.mean([r["best_validation"]["mean_active_collection_fraction"] for r in records])),
                                 per_class_recall_mean={name: float(np.mean([r["classification_report"][name]["recall"] for r in records])) for name in records[0]["classes"]})
            if task == "fine9":
                for key in ("within_true_coarse_fine_accuracy", "coarse_from_9ports_accuracy"):
                    stats[str(wl)][key+"_mean"] = float(np.mean([r[key] for r in records]))
                stats[str(wl)]["within_true_coarse_per_group_mean"] = {name: float(np.mean([r["within_true_coarse_fine_accuracy_per_group"][name] for r in records])) for name in COARSE}
        paired = []
        for seed in (42, 43, 44):
            a = np.load(result_dir(task, seed, 1310) / "validation_predictions.npz")
            b = np.load(result_dir(task, seed, 1610) / "validation_predictions.npz")
            np.testing.assert_array_equal(a["truth"], b["truth"])
            paired.append(dict(seed=seed, **paired_comparison(a["prediction"], b["prediction"], a["truth"])))
        stats["paired_by_seed"] = paired
        stats["caution"] = "Exploratory validation comparisons, checkpoint selection on same validation; no independent test and no multiple-comparison correction. Seeds are not independent samples of the validation set."
        comparison[task] = stats
    save_json(OUT / "summary.json", comparison)
    save_json(OUT / "runs.json", rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    for ax, task in zip(axes[:2], ("coarse3", "fine9")):
        for i, wl in enumerate((1310, 1610)):
            ss = comparison[task][str(wl)]
            ax.bar(i, ss["mean_accuracy"]*100, color=COLORS[wl], alpha=.7, width=.6)
            ax.errorbar(i, ss["mean_accuracy"]*100, yerr=ss["sample_sd"]*100, color="black", capsize=5)
            ax.scatter(np.array([-.12, 0, .12])+i, np.array(ss["seed_accuracies"])*100, c="black", s=17)
            ax.text(i, ss["mean_accuracy"]*100+1.2, f'{ss["mean_accuracy"]*100:.2f}%', ha="center")
        ax.set_xticks([0, 1], ["1310 nm", "1610 nm"])
        ax.set_ylim(65 if task == "fine9" else 80, 95)
        ax.set_ylabel("Validation accuracy (%)")
        ax.set_title("3 coarse classes" if task == "coarse3" else "9 fine subclasses")
        ax.grid(axis="y", alpha=.2)
    for wl in (1310, 1610):
        for task, style in (("coarse3", "-"), ("fine9", "--")):
            histories = [json.loads((result_dir(task, s, wl) / "history.json").read_text()) for s in (42, 43, 44)]
            acc = np.array([[row["val_accuracy"] for row in h] for h in histories])*100
            axes[2].plot(np.arange(1, 21), acc.mean(0), style, color=COLORS[wl], label=f"{wl} {task}")
            axes[2].fill_between(np.arange(1, 21), acc.min(0), acc.max(0), color=COLORS[wl], alpha=.08)
    axes[2].set(xlabel="Epoch", ylabel="Validation accuracy (%)", title="Same 20-epoch budget; 3 seeds")
    axes[2].legend(fontsize=8)
    fig.suptitle("Calibrated scalar surrogate; analytic ports; not full-device FDTD", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "classification_comparison.png")
    plt.close(fig)
    for task in ("coarse3", "fine9"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, wl in zip(axes, (1310, 1610)):
            ss = json.loads((result_dir(task, 42, wl)/"summary.json").read_text())
            cm = np.array(ss["confusion_matrix"])
            im = ax.imshow(cm/cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
            labels = ["landform", "water/coast", "built"] if task == "coarse3" else [x.split("_", 1)[1] for x in FINE]
            ax.set_xticks(np.arange(len(labels)), labels, rotation=50, ha="right")
            ax.set_yticks(np.arange(len(labels)), labels)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if cm[i, j] or task == "coarse3":
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8, color="white" if cm[i,j]/cm[i].sum()>.5 else "black")
            ax.set(title=f"{wl} nm, seed 42", xlabel="Predicted", ylabel="True")
        fig.tight_layout()
        fig.savefig(OUT / f"confusion_{task}.png")
        plt.close(fig)
    return comparison


@torch.no_grad()
def convergence(tasks=("coarse3", "fine9")):
    cache = OUT / "convergence.json"
    records = json.loads(cache.read_text()) if cache.exists() else []
    for task in tasks:
        data, names = dataset(task)
        x, y = torch.tensor(data[1]), torch.tensor(data[3])
        for seed in (42, 43, 44):
            for wl in (1310, 1610):
                saved = np.load(result_dir(task, seed, wl) / "validation_predictions.npz")
                for n, pad in ([(20000, 2), (10000, 4)] if seed == 42 else [(20000, 2)]):
                    if any(r["task"] == task and r["seed"] == seed and r["wavelength_nm"] == wl and r["grid_n"] == n and r["pad_factor"] == pad for r in records):
                        continue
                    model = checkpoint_model(task, seed, wl, n=n, pad=pad)
                    metrics, p, _ = evaluate(model, x, y, len(names))
                    record = dict(task=task, seed=seed, wavelength_nm=wl, grid_n=n, pad_factor=pad,
                                  probability_mean_abs_delta=float(abs(p-saved["probabilities"]).mean()),
                                  probability_max_abs_delta=float(abs(p-saved["probabilities"]).max()),
                                  accuracy=metrics["accuracy"], accuracy_delta=float(metrics["accuracy"]-(saved["prediction"]==saved["truth"]).mean()),
                                  prediction_changes=int((p.argmax(1)!=saved["prediction"]).sum()))
                    records.append(record)
                    save_json(cache, records)
                    print("CONVERGENCE " + json.dumps(record), flush=True)
    save_json(OUT / "convergence.json", records)
    return records


def field_metrics(model, field):
    intensity = field.abs().square().numpy()
    y = (np.arange(model.n*model.pad) - model.n*model.pad/2 + .5)*model.dy
    weight = intensity / intensity.sum()
    center = (y*weight).sum()
    sigma = np.sqrt(((y-center)**2*weight).sum())
    spatial_frequency = np.fft.fftfreq(len(y), d=model.dy)
    spectrum = abs(np.fft.fft(field.numpy()))**2
    return dict(total_scalar_power=float(intensity.sum()*model.dy), centroid_um=float(center), intensity_sigma_um=float(sigma),
                spectral_fraction_above_0p25_cycles_per_um=float(spectrum[abs(spatial_frequency)>.25].sum()/spectrum.sum()))


@torch.no_grad()
def propagation():
    data, _ = dataset("coarse3")
    x = torch.tensor(data[1][0:1])
    assert data[3][0] == 0
    records, raw = {}, {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for row, wl in enumerate((1310, 1610)):
        model = checkpoint_model("coarse3", 42, wl)
        same_slots = CalibratedDONN(wl, 42).lengths.detach()
        records[str(wl)] = dict(neff=model.neff, slab_wavelength_um=wl*.001/model.neff,
                                propagating_cutoff_cycles_per_um=model.neff/(wl*.001),
                                input_note="First validation landform sample, exactly identical 16-channel phases at both wavelengths.")
        for col, (name, lengths) in enumerate((("same_physical_mask", same_slots), ("separately_trained_mask", model.lengths.detach()))):
            u = model.input_field(x)
            masks = model.masks(lengths)
            slices, zvalues, planes = [], [], [field_metrics(model, u[0])]
            for gap in range(4):
                if gap:
                    u = u*masks[gap-1]
                for d in np.linspace(0, 125, 51):
                    ud = u if d == 0 else model.propagate(u, float(d))
                    central = ud[0, model.start:model.start+model.n]
                    slices.append(central.numpy())
                    zvalues.append(gap*125+float(d))
                u = model.propagate(u)
                planes.append(field_metrics(model, u[0]))
            field = np.asarray(slices)
            raw[f"{wl}_{name}"] = field
            y = model.y.numpy()
            raw["y_um"], raw["z_um"] = y, np.array(zvalues)
            intensity = abs(field)**2
            # Common input normalization, retaining actual wavelength-dependent attenuation.
            reference = float(abs(model.input_field(x)[0]).square().max())
            log_i = 10*np.log10(np.maximum(intensity/reference, 1e-6))
            select = abs(y) <= 150
            ax = axes[row, col]
            im = ax.imshow(log_i[:, select].T, origin="lower", aspect="auto", extent=[0, 500, y[select][0], y[select][-1]], cmap="magma", vmin=-40, vmax=0)
            for z in (125, 250, 375):
                ax.axvline(z, color="cyan", alpha=.5, lw=.7)
            ax.set(title=f"{wl} nm: {'identical initial slot lengths' if col == 0 else 'individually trained slots'}", xlabel="Propagation x (um)", ylabel="Transverse y (um)")
            records[str(wl)][name] = dict(planes=planes, output_ports=model.ports(u)[0].numpy().tolist())
        # Single narrow pulse: no masks, same physical beam in both wavelengths.
        y_full = (torch.arange(model.n*model.pad)-model.n*model.pad/2+.5)*model.dy
        pulse = torch.exp(-y_full.square()/4).to(torch.complex64)[None, :]
        pulse /= torch.sqrt(pulse.abs().square().sum()*model.dy)
        pulse_rows = []
        for d in (0., 125., 500.):
            field = pulse if d == 0 else model.propagate(pulse, d)
            pulse_rows.append(dict(distance_um=d, **field_metrics(model, field[0])))
        records[str(wl)]["single_gaussian_no_slots_sigma0_1um"] = pulse_rows
    fig.suptitle("Angular-spectrum fields: identical-mask control vs trained designs", fontsize=13)
    fig.subplots_adjust(left=.08, right=.83, bottom=.08, top=.90, hspace=.25, wspace=.15)
    cax = fig.add_axes([.87, .16, .018, .67])
    fig.colorbar(im, cax=cax, label="Intensity relative to common input peak (dB)")
    fig.savefig(OUT / "propagation_comparison.png")
    plt.close(fig)
    # The same layout is essential when attributing a difference to wavelength.
    a, b = raw["1310_same_physical_mask"][-1], raw["1610_same_physical_mask"][-1]
    records["same_mask_output_intensity_pearson"] = float(np.corrcoef(abs(a)**2, abs(b)**2)[0,1])
    records["same_mask_output_complex_fidelity"] = float(abs(np.vdot(a,b))**2/(np.vdot(a,a).real*np.vdot(b,b).real))
    save_json(OUT / "propagation_metrics.json", records)
    np.savez_compressed(OUT / "propagation_fields.npz", **raw)
    # Plot actual calibrated phase/amplitude; don't force linear phase-only behavior.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for wl in (1310, 1610):
        r = np.load(CAL / f"wl{wl}nm/calibration_response.npz")
        axes[0].plot(r["slot_length_um"], r["phase_pi"], "o-", ms=3, color=COLORS[wl], label=f"{wl} nm")
        axes[1].plot(r["slot_length_um"], abs(r["relative_complex_transmission"]), "o-", ms=3, color=COLORS[wl], label=f"{wl} nm")
        pulse = records[str(wl)]["single_gaussian_no_slots_sigma0_1um"]
        axes[2].plot([r["distance_um"] for r in pulse], [r["intensity_sigma_um"] for r in pulse], "o-", color=COLORS[wl], label=f"{wl} nm")
    axes[0].set(xlabel="Slot length (um)", ylabel="Unwrapped relative phase / pi", title="Lumerical local calibration")
    axes[1].set(xlabel="Slot length (um)", ylabel="Relative transmitted field amplitude", title="Amplitude is included in training")
    axes[2].set(xlabel="Propagation distance (um)", ylabel="Intensity standard deviation (um)", title="Same narrow beam, no slots")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "calibration_and_spreading.png")
    plt.close(fig)
    return records


@torch.no_grad()
def robustness_and_transfer():
    tolerance_path = OUT / "slot_tolerance.json"
    transfer_path = OUT / "cross_wavelength_same_mask.json"
    if tolerance_path.exists() and transfer_path.exists():
        return json.loads(tolerance_path.read_text()), json.loads(transfer_path.read_text())
    robustness, transfer = [], []
    generator = torch.Generator().manual_seed(20261016)
    # Matched physical fabrication errors, same errors at both wavelengths.
    perturbations = {sigma: [torch.randn(3, 555, generator=generator)*sigma for _ in range(3)] for sigma in (.005, .01, .02)}
    for task in ("coarse3", "fine9"):
        data, names = dataset(task)
        x, y = torch.tensor(data[1]), torch.tensor(data[3])
        for wl in (1310, 1610):
            model = checkpoint_model(task, 42, wl)
            for sigma, errors in perturbations.items():
                metrics = [evaluate(model, x, y, len(names), lengths=(model.lengths.detach()+error).clamp(0,2))[0] for error in errors]
                robustness.append(dict(task=task, wavelength_nm=wl, seed=42, length_noise_sigma_nm=sigma*1000,
                                       accuracies=[m["accuracy"] for m in metrics], mean_accuracy=float(np.mean([m["accuracy"] for m in metrics]))))
            other = 1610 if wl == 1310 else 1310
            swapped = checkpoint_model(task, 42, other)
            metrics, _, _ = evaluate(swapped, x, y, len(names), lengths=model.lengths.detach())
            transfer.append(dict(task=task, trained_wavelength_nm=wl, evaluated_wavelength_nm=other, validation=metrics))
    save_json(OUT / "slot_tolerance.json", robustness)
    save_json(OUT / "cross_wavelength_same_mask.json", transfer)
    return robustness, transfer


def report(summary, converged, propagation_records, robustness, transfer):
    text = ["# 1310 nm / 1610 nm 衍射网络分类对照\n",
            "本目录包含实际完成的训练及分析。结论仅针对**局部 Lumerical 标定驱动的标量角谱代理模型**，输入/输出为解析余弦孔径，尚未生成这两个波长的完整器件 FDTD 端口映射。原 1560 nm 项目未覆盖。\n",
            "## 分类结果\n",
            "训练集 7200，固定验证集 1260；每个模型训练 20 轮，3 个配对种子（42/43/44），共 12 次正式训练。下表为各次最佳验证模型的平均准确率 ± 种子间样本标准差；这是验证成绩，不是独立测试成绩。\n",
            "|任务|1310 nm|1610 nm|\n|---|---:|---:|"]
    for task, label in (("coarse3", "三大类"), ("fine9", "九个细类")):
        cells = [f'{summary[task][str(wl)]["mean_accuracy"]*100:.2f}% ± {summary[task][str(wl)]["sample_sd"]*100:.2f} 百分点' for wl in (1310, 1610)]
        text.append(f"|{label}|{cells[0]}|{cells[1]}|")
    for task, label in (("coarse3", "三分类"), ("fine9", "九分类")):
        delta = (summary[task]["1610"]["mean_accuracy"]-summary[task]["1310"]["mean_accuracy"])*100
        wins = sum(r["paired_delta_1610_minus_1310"] > 0 for r in summary[task]["paired_by_seed"])
        text.append(f'\n{label}中 1610−1310 的平均差为 {delta:+.3f} 百分点，3个配对种子中1610更高的有 {wins} 个。')
    text += ["\n三大类：自然地貌、水域/海岸、人造建筑。九细类：山地、山谷、火山、珊瑚礁、湖岸、海岸、教堂、城堡、图书馆。数据来自合成可见光图像；波长表示光计算载波，不表示红外成像波段。\n",
             "粗分类使用原项目冻结的 16 维编码（LDA2+PCA14）。细分类使用同一组 568 维原始特征和相同样本划分，训练集拟合的 RBF384→LDA8+PCA8 编码为 16 个输入。两个波长使用相同编码；粗细任务间编码不同，不能把二者准确率直接相减来度量任务难度。\n",
             "九分类还报告“已知真实大类后，在相应 3 个细类端口中取最大”的诊断准确率。该指标用了真实大类，不等于可部署级联系统的准确率；详见 summary.json。\n",
             "![分类对照](classification_comparison.png)\n",
             "## 几何与模型\n",
             "保持 500 μm 横向宽度、4 段 125 μm 传播间距、16 输入、10 输出、3 层可训练槽。原 shape 文件中 1500 组独立槽长与 500 μm 宽度及 0.3 μm×3 重复的 0.9 μm 组间距冲突。本对照明确采用 555 组，每层覆盖 499.5 μm，两侧各留 0.25 μm。每组采用周期单元标定的有效薄屏透射，不显式解析组内 3 根槽边界。\n",
             "1310/1610 nm 分别使用自己的 TE 有效折射率及复透射幅度、展开相位插值，槽长限定 0–2 μm，不外推。角谱法包含倏逝分量衰减；500 μm 区域用 10000 点（0.05 μm）采样，传播计算区域为 1000 μm（2 倍填充），全场在各间距之间保留，只有槽层覆盖中央区域。输入/输出分别是 10/12 μm 宽的归一化余弦孔径，不声称是实测或完整锥形波导模式。\n",
             "随机初始槽长为 0.5±0.02 μm，各波长严格配对；Adam lr=8e-4，weight_decay=1e-4，余弦下降，batch64，梯度裁剪1，输入相位噪声0.005π。损失为活动端口归一化概率交叉熵，无可训练电子读出。三分类取前3端口，九分类取前9端口。\n",
             "薄屏模型忽略不同槽之间的非局部耦合、反射往返、斜入射响应变化和真实端口频散。因此这些结果不能与旧 1560 nm 91.75% 直接比较，更不能替代完整器件验证。\n",
             "## 传播差异\n",
             "下图左列使用完全相同的输入和物理槽长，右列使用分别训练的槽长。只用左列归因于波长；右列还包含优化设计差异。以相同输入峰值归一化，颜色为强度 dB。\n",
             "![传播对照](propagation_comparison.png)\n"]
    for wl in (1310, 1610):
        p = propagation_records[str(wl)]
        pulse = p["single_gaussian_no_slots_sigma0_1um"]
        text.append(f'- {wl} nm：n_eff={p["neff"]:.6f}，片上有效波长={p["slab_wavelength_um"]:.6f} μm；传播空间频率截止={p["propagating_cutoff_cycles_per_um"]:.3f} 周期/μm。同一初始强度标准差 1 μm 的无槽高斯束，125 μm 后标准差 {pulse[1]["intensity_sigma_um"]:.3f} μm，500 μm 后 {pulse[2]["intensity_sigma_um"]:.3f} μm。')
    text += ["\n1610 nm 的片上有效波长约大 31%，同样的窄光束衍射展宽更明显。1310 nm 对单位槽长的相位变化更敏感：相位斜率约 −2.309π/μm，1610 nm 约 −1.738π/μm。相同 10 nm 长度误差对应约 0.0726 / 0.0546 rad 相位误差；这是相位灵敏度，不能单独代表总分类鲁棒性。实际复透射幅度也随槽长和波长变化。\n",
             "![标定与展宽](calibration_and_spreading.png)\n",
             "编码后的 16 个特征通道不按原图空间坐标排列，且前几个特征经过监督判别降维。光场横向高空间频率不直接等于图像细节，因而不能单凭短波衍射小就推出短波更适合语义细分类。\n",
             "## 数值验证及实用限制\n"]
    grid = [r for r in converged if r["grid_n"] == 20000]
    pad = [r for r in converged if r["pad_factor"] == 4]
    text.append(f'- 全部12个最佳模型在 0.025 μm 网格复核；概率平均绝对变化最大 {max(r["probability_mean_abs_delta"] for r in grid):.6f}，准确率变化最大 {max(abs(r["accuracy_delta"]) for r in grid)*100:.3f} 百分点。')
    text.append(f'- seed42 的4个模型将填充从2倍增至4倍；概率平均绝对变化最大 {max(r["probability_mean_abs_delta"] for r in pad):.6f}，准确率变化最大 {max(abs(r["accuracy_delta"]) for r in pad)*100:.3f} 百分点。')
    text += ["- preflight.json 检查了输入能量归一、被动传播能量不增及非零有限梯度。slot_tolerance.json 是 seed42 上 5/10/20 nm 槽长随机扰动的3次重复诊断，未纳入再训练或模型选择。",
             "- accuracy 使用活动端口内的归一化概率；优化没有约束收光效率。以下收光比例是解析标量模型相对输入功率的估算，不是实测器件效率："]
    for task in ("coarse3", "fine9"):
        for wl in (1310, 1610):
            text.append(f'  - {task}, {wl} nm：活动端口平均收光 {summary[task][str(wl)]["mean_active_collection_fraction"]*100:.3f}%。')
    text += ["- 两个波长分别优化了不同槽长，并非共享结构的双波长网络。跨波长沿用同一槽长的诊断见 cross_wavelength_same_mask.json。",
             "- 固定验证集同时用于选择最佳轮次；只有3个种子，无独立测试集。配对 McNemar 结果见 summary.json，仅作探索，不把多种子预测当成独立新增验证样本。",
             "\n## 关于粗分/细分的解释\n",
             "应以分类对照判断当前系统中的适用性，并结合种子稳定性、数值误差及效率。即便某波长在九分类略好，也不能称为普适的‘细分波长’；孔径、传播距离、槽布局、输入编码、类别及训练预算均会改变排序。可检验的工程假设是 1610 nm 偏向更强空间混合、1310 nm 保留更高横向带宽；本实验并不预设这一假设会转化为长波粗分、短波细分。\n",
             "若要实现级联任务，应另行联合训练共享槽结构或明确的两级网络：第一级给出三大类，第二级在预测大类内给出细类；以级联最终九类准确率、收光效率和串扰为联合指标，并比较两种波长分配。当前分别训练的模型及使用真实大类的诊断不足以证明该级联系统。\n",
             "## 复现与文件\n",
             "在项目根目录运行（Python环境需 torch、numpy、scipy、scikit-learn、joblib、matplotlib）：\n",
             "```powershell\npython comparison_1310_1610_scene3_20261016/run_comparison.py --tasks coarse3 fine9 --seeds 42 43 44\npython comparison_1310_1610_scene3_20261016/analyze_comparison.py\n```\n",
             "已完成结果会跳过，不覆盖。正式结果在 resolved_grid10000/，其上级 coarse3/ 和 fine9/ 为早期0.10 μm网格的探索，不能混用。每次正式训练含 best/final_model.pt、槽长.npy、history.json、summary.json、validation_predictions.npz；本 analysis/ 保存全部图与分析 JSON。config.json 含数据与标定输入 SHA256。\n"]
    text += ["## 细分诊断、槽长误差及同槽跨波长\n",
             "已知真实大类时，九端口模型在相应三个细类内的诊断准确率（种子平均）：\n",
             "|所属大类|1310 nm|1610 nm|\n|---|---:|---:|"]
    for name in COARSE:
        vals = [summary["fine9"][str(wl)]["within_true_coarse_per_group_mean"][name]*100 for wl in (1310, 1610)]
        text.append(f"|{name}|{vals[0]:.2f}%|{vals[1]:.2f}%|")
    for wl in (1310, 1610):
        acc = summary["fine9"][str(wl)]["within_true_coarse_fine_accuracy_mean"]*100
        text.append(f'\n{wl} nm：真实大类条件下的总体细分准确率 {acc:.2f}%。')
    text += ["\n以下为 seed42 单独诊断，不与三种子正式均值混用：\n",
             "|任务|波长|20 nm 槽长高斯误差的平均准确率，3次扰动|\n|---|---:|---:|"]
    for r in robustness:
        if r["length_noise_sigma_nm"] == 20:
            text.append(f'|{r["task"]}|{r["wavelength_nm"]} nm|{r["mean_accuracy"]*100:.2f}%|')
    text += ["\n|任务|槽长训练波长→测试波长|直接切换波长的准确率|\n|---|---|---:|"]
    for r in transfer:
        text.append(f'|{r["task"]}|{r["trained_wavelength_nm"]}→{r["evaluated_wavelength_nm"]} nm|{r["validation"]["accuracy"]*100:.2f}%|')
    text.append("\n同一槽长直接换波长的表现说明：分别训练成功不意味着一套槽版能承担两个波长的粗细分任务。共享器件需将两种波长的复响应同时纳入训练目标。")
    local_path = BASE / "local_slot_fields/summary.json"
    if local_path.exists():
        text += ["## 相同槽单元的实际 Lumerical 场图\n",
                 "local_slot_fields.py 额外实跑了两个波长的无槽基线和同一 0.15 μm 宽、0.5 μm 长的 SiO2 填充槽；横向周期 0.3 μm，220 nm Si，其他参数与标定一致。这里的槽是周期阵列的一个单元，不是横向孤立槽。保存了原始 Hz 复场、求解器工程和日志。\n",
                 "![局部varFDTD场图](../../local_slot_fields/local_slot_field_comparison.png)\n",
                 "场图左侧为相对无槽入射基线归一化的 |Hz|²，右侧为相对无槽复场的包裹相位；青框为槽边界。入射侧存在反射叠加，不能用该侧总场当每次的入射功率。局部 |Hz|² 是场分量诊断，不是器件透过功率。\n"]
        for r in json.loads(local_path.read_text(encoding="utf-8")):
            text.append(f'- {r["wavelength_nm"]} nm：0.5 μm 槽的相对透射幅度 {r["transmitted_relative_amplitude"]:.6f}，沿槽长扫描展开后的相对相位 {r["calibration_unwrapped_phase_pi"]:.6f}π；与原标定幅度/相位差 {r["calibration_amplitude_difference"]:.2e}/{r["calibration_phase_difference_pi"]:.2e}π。')
        text.append("\n该图直接显示相同槽结构中的局部场差异；它并不使前述网络代理模型成为完整器件 FDTD。")
    (OUT / "REPORT.md").write_text("\n".join(text), encoding="utf-8")


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = aggregate()
    checks = convergence()
    prop = propagation()
    robustness, transfer = robustness_and_transfer()
    report(summary, checks, prop, robustness, transfer)
    save_json(OUT / "artifact_sha256.json", {str(p.relative_to(BASE)): sha(p) for p in sorted(BASE.rglob("*")) if p.is_file() and p.suffix in (".py", ".json", ".npy", ".npz", ".pt", ".joblib", ".png", ".md") and p.name != "artifact_sha256.json"})
    print("Analysis complete: " + str(OUT), flush=True)


if __name__ == "__main__":
    main()
