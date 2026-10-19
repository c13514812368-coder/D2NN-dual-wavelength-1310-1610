# 1560 nm 场景三分类完整实验副本

本文件夹对应已完成实验 train_rich16_1560_20261011：最佳固定验证准确率 91.7460%（第 9 轮），无训练噪声下的最佳模型训练集准确率 91.6528%，第 20 轮验证准确率 91.0317%。三个大类为自然地貌、水域海岸、建筑与人造环境。

包含原 map、完整已保存实验、最佳和最终检查点、完整训练数据及编码、训练实现、光学源代码和便携运行入口。无需依赖原桌面 d2nn 或 Documents/1 目录；不包含 780 nm 实验。所有原文件保留。

## 文件说明

| 路径 | 内容 |
|---|---|
| map/ | 原始 1540/1560 nm 物理映射包和几何、仿真元数据；实际训练只使用 1560 nm |
| train_rich16_1560_20261011/ | 已完成实验：配置、日志、核验结果、曲线、混淆矩阵、逐层复场、最佳/最终模型及完整 trained_network_package.pt |
| feature_upgrade_20261011/ | encoded16.npz、encoder.joblib、原始特征缓存、编码诊断、数据审计和便携版特征/训练/预测代码 |
| legacy/train_impl.py | 本实验实际使用的训练基础实现 |
| src/multiwavelenth_donn/ | 原项目光学传播、map 加载、训练管线等源码 |
| data/imagenet_1k_sd_scene_3coarse_fixed/ | 7,200 张训练图片、1,260 张固定验证图片和原 manifest |
| provenance/ | 原始绝对路径配置、入口和历史报告脚本，仅用于追溯；不是运行入口 |
| run_experiment.py | 统一入口：验证、重新训练、预测、重建编码、校验文件 |
| environment_versions.json | 本机实际验证环境版本 |
| file_checksums.json | 交付文件的 SHA256 与大小；runtime 和 reruns 不在交付文件校验中 |
| runtime/verification.json | 本副本运行验证结果 |

数据来源是 Imagenet-1k-SD-1.4 的合成 RGB 图片，不是原始 ImageNet 摄影图，也不是红外实测图。
每个小类为 800 张训练、140 张验证：高山/山谷/火山，珊瑚礁/湖岸/海岸，教堂/城堡/图书馆；每个大类 2,400 张训练、420 张验证。没有独立测试集。

## 环境

建议 Python 3.12。进入本文件夹后安装 requirements.txt；requirements_versions_reference.txt 记录本机完整依赖版本，仅供对齐已验证环境，其他系统可先使用可安装的兼容版本。更换 scikit-learn 等版本时需重新验证 joblib 编码器兼容性。没有复制平台相关的大型虚拟环境。

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

安装和执行时请使用该虚拟环境的解释器，例如 macOS/Linux 为 .venv/bin/python，Windows 为 .venv\Scripts\python.exe；使用前按自己的系统激活环境。

不需要 Lumerical 即可使用已保存 map 训练、推理和分析；重新做真实 FDTD 标定仍需 Lumerical 与许可证。原 map 中有历史 Windows 路径，它们是来源记录；当前模型会从包内字段恢复模式数据，不要求那些 Windows 路径存在。本包不含缺失的原始 .fsp/.lms 工程。

## 先验证最佳模型

在本文件夹执行：

```bash
python run_experiment.py verify
```

读取最佳检查点并遍历全部 1,260 张验证图片的已保存编码，应得到 91.7460%，混淆矩阵 [[369,30,21],[26,385,9],[11,7,402]]。还检查全部图片数量及一个小批量的前向/反向有限性，不做参数更新。结果写入 runtime/verification.json，不改历史结果。已保存编码和原图片一起包含。

## 重新训练

```bash
python run_experiment.py train
```

使用原 20 epochs、batch 64、seed 42、CPU 4 线程、Adam lr=8e-4、weight decay=1e-4、余弦衰减、梯度裁剪 1.0 和输入噪声 0.005。输出到新的 reruns/train_1560_时间戳/，不会覆盖已保存实验。
训练可能生成另一个大型 trained_network_package.pt，应预留空间。不同硬件/依赖可能有数值差异；当前交付核验不代表又重跑了完整 20 轮。

## 单张或多张预测

```bash
python run_experiment.py predict path/to/image.jpg
```

使用冻结的 encoder.joblib 与原最佳模型。多图路径可依次列出，输出三类端口归一化响应及预测类别。

## 可选：从图片重建特征编码

```bash
python run_experiment.py prepare-features
```

图片转 RGB 64x64，提取 568 维手工特征，训练集 StandardScaler，RBF Nystroem(384) 后 LDA2 与标准化特征的 PCA14 拼接，再用训练集 0.5/99.5 分位数映射到 [0.2,1.8]，作为 pi 单位的相位编码。LDA 使用训练标签，验证集不参与拟合。已有 raw_features.npz 时复用原特征缓存。
本操作会重写副本内编码及诊断文件，版本变化可能影响数值；通常直接使用已保存编码重现原实验即可。主动重建后，旧的编码文件校验不再适用。

## 校验文件

```bash
python run_experiment.py verify-files
```

交付包内所有静态文件（包括 map、图片、检查点、源码）按 file_checksums.json 校验。运行产生的 runtime、reruns 文件不纳入；特征重建属于有意改写，需要重新记录校验信息。

## 结果图口径

accuracy_train_validation.png 是逐轮训练/验证原始准确率；accuracy_train_best_validation.png 中验证线是截至当前轮的最佳值，不是当轮验证准确率。日志及 train_metrics.json 保留原值。

91.7460% 是固定验证集最佳检查点结果，不是独立测试准确率或真实加工器件测量。提升包含数字手工特征与监督 LDA 压缩；训练后的槽几何尚未由完整 FDTD 重新验证。历史报告脚本依赖早期实验并会改写摘要，因此只归档在 provenance，统一核验使用 run_experiment.py verify。
