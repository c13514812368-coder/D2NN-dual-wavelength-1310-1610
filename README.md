# D2NN: 1310 nm / 1610 nm 波长对照研究

硅基片上衍射光学神经网络研究项目，包含原 1560 nm 三分类实验，以及 1310 nm、1610 nm 的局部电磁标定、分类对照、传播分析和数据集尺度诊断。

本仓库保存代码、配置、紧凑的数值结果和分析图表。原始图片、两个超过 3 GB 的完整模型包、Lumerical 二进制工程和较大的传播场缓存保存在本地，未纳入 Git。克隆仓库可以阅读代码及已有报告；完整复现需要按下文恢复相应数据。

## 当前研究结论与范围

1310/1610 nm 对照使用 Lumerical MODE FDE / varFDTD 局部标定驱动的标量角谱模型，采用解析输入、输出模式。它不是这两个波长的完整器件 FDTD 验证。

每个波长分别优化槽长，尚未实现共享槽结构的双波长联合分类器，也尚未完成可部署的粗分、细分级联。

正式对照沿用 7200 个训练样本、1260 个验证样本，使用种子 42、43、44，每次训练 20 轮。最佳验证准确率的均值及种子间样本标准差为：

| 任务 | 1310 nm | 1610 nm |
| --- | --- | --- |
| 三大类分类 | 90.85% ± 0.64 百分点 | 90.85% ± 0.09 百分点 |
| 九细类分类 | 79.81% ± 0.67 百分点 | 80.93% ± 0.33 百分点 |

验证集用于选择最佳轮次，没有独立测试集。当前结果不能推出“长波适合粗分、短波适合细分”的普遍结论。当前合成场景数据也没有保证细分类必须依赖局部细节。

完整证据、模型限制和数值收敛检查见 [双波长分析报告](comparison_1310_1610_scene3_20261016/resolved_grid10000/analysis/REPORT.md) 和 [数据集尺度诊断](dataset_scale_audit_20261019/REPORT.md)。

## 目录与入口

| 位置 | 内容 |
| --- | --- |
| `src/multiwavelenth_donn/` | 原项目的光学网络、映射、训练与 Lumerical 接口 |
| `run_experiment.py` | 原 1560 nm 项目的验证、训练、预测和特征处理入口 |
| `train_rich16_1560_20261011/` | 1560 nm 配置、小型检查点、训练指标与图表 |
| `feature_upgrade_20261011/` | 568 维图像特征、训练集拟合的编码器及 16 维相位编码 |
| `map/` | 原 1560 nm 映射的结构及元数据，完整映射包需另行恢复 |
| `calibration_1310_1610nm_20261016/` | 两个通信波长的标定脚本、复透射查表及标定报告 |
| `comparison_1310_1610_scene3_20261016/` | 分类、同槽长传播、局部槽场与跨波长诊断 |
| `dataset_scale_audit_20261019/` | 数据对整体、局部特征分类适用性的审计 |
| `calibration_780nm_20261015/` | 历史 780 nm 标定探索，不属于正式双波长结论 |
| `legacy/`、`provenance/` | 现有实现所需的历史代码及原配置记录 |
| `RESEARCH_ROADMAP_20261019.md` | 后续研究路线 |
| `BUILDING_MULTISCALE_DATA_PLAN_20261019.md` | 建筑外形与窗格细节的数据设计计划，尚未生成该数据集 |

正式双波长结果在 `comparison_1310_1610_scene3_20261016/resolved_grid10000/`。其父目录的 `coarse3/`、`fine9/` 是较粗网格的探索记录，不用于正式汇总。

## 安装环境

建议创建独立 Python 环境，并安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`environment_versions.json` 和 `requirements_versions_reference.txt` 保存历史环境信息，作为复现参考，并非所有系统通用的锁定环境。

运行新的电磁标定还需要自行安装并许可 Ansys Lumerical MODE。历史实验使用 v241，API 路径为 `D:\Program Files\Lumerical\v241\api\python\lumapi.py`。在其他电脑上运行前，应检查标定脚本及配置中的 API 路径。仓库不包含商业软件或许可证。

## 数据与大文件恢复

数据集是 Imagenet-1k-SD-1.4 合成图像的固定子集，不是原版 ImageNet 实拍图片，也不是红外成像数据。这里的波长是光计算载波波长。三大类为自然地貌、水域/海岸、人造建筑，每个大类包含三个细类。

将已有数据恢复到 `data/imagenet_1k_sd_scene_3coarse_fixed/`，保留原 `train/`、`val/` 目录层级、类别目录和文件名。应有 7200 张训练图、1260 张验证图。详细顺序可参考 `feature_upgrade_20261011/diagnostics.json`，数据来源与划分见该数据目录中的 README 和 manifest。图片没有随仓库重新分发。

原 1560 nm 的训练、验证和预测还需要恢复：

```text
map/mapped_network_package.pt
```

`train_rich16_1560_20261011/artifacts/trained_network_package.pt` 是另一个保留在本地的完整训练包。小型 `best_model.pt`、`final_model.pt` 检查点随仓库保存，但不能替代完整映射包。

`file_checksums.json` 是原始便携项目的归档校验表，包含仓库未分发的文件；只有恢复完整归档后，才适合运行 `verify-files`。它不是当前 Git 文件清单。

## 主要运行方式

以下命令均在项目根目录运行。阅读现有报告不需要 Lumerical；重新运行分析或训练需要相关 Python 依赖与上述数据。请先确认本机能承担相应的训练、仿真开销。

```powershell
# 原 1560 nm 实验：需要完整映射包和原图数据
python run_experiment.py verify
python run_experiment.py predict path/to/image.jpg

# 两波长、两任务、三个种子的正式对照；已有完成结果会复用
python comparison_1310_1610_scene3_20261016/run_comparison.py --tasks coarse3 fine9 --seeds 42 43 44

# 分析已有双波长结果，可重新生成传播场缓存
python comparison_1310_1610_scene3_20261016/analyze_comparison.py

# 以下两个入口涉及真实 Lumerical 求解或已有求解缓存
python calibration_1310_1610nm_20261016/run_calibration.py --help
python comparison_1310_1610_scene3_20261016/local_slot_fields.py
```

正式对照采用每层 555 组槽、3 层可训练槽、4 段 125 μm 间距、500 μm 横向区域。该选择解决了原归档中槽数量与宽度、周期不一致的问题，因此不能将新结果直接与旧 1560 nm 的准确率比较。

## 使用与授权

项目尚未选定开源许可证，沿用 `LICENSE` 中的保留权利声明。将仓库存储在 GitHub 不表示已授予开源再分发许可。第三方数据、软件和论文遵循各自的授权条款。
