# 1310 nm / 1610 nm 对照实验

本实验分别训练两个波长的三分类模型，并用九个子类补充检验粗分/细分设想。原项目数据、标定、1560 nm 模型均保留。

2026-10-19 新增 [共享槽结构的双波长联合九分类](joint_fine9_shared_20261019/REPORT.md)，入口为 `run_joint_fine9.py`。三个种子、每次20轮的平均最佳验证准确率为 79.50% ± 0.87 百分点；同一套1665个槽长、固定总输入功率、两波长等功率端口强度求和。下文原 `resolved_grid10000/` 结果仍属于分别训练的单波长对照。

正式模型是局部 Lumerical 标定驱动的标量角谱网络，采用解析输入/输出孔径。它不是两个波长的新完整器件 FDTD 映射，也不复用旧 1540/1560 nm 场来冒充新波长。

- `resolved_grid10000/`：正式训练，0.05 μm 网格，种子 42/43/44，两个波长、三分类与九分类共 12 次。
- `resolved_grid10000/analysis/REPORT.md`：完成分析后的完整报告及复现说明。
- `resolved_grid10000/analysis/`：汇总结果、混淆矩阵图、逐层传播图、数值收敛复核、长度误差诊断、跨波长同槽长诊断。
- `local_slot_fields/`：同一 0.15 μm × 0.5 μm 槽的实际 Lumerical varFDTD 原始复场、无槽基线、求解器工程、场图。
- 根目录下 `coarse3/`、`fine9/`：0.10 μm 网格的首轮探索结果；与正式结果区分，不用于最终汇总。
- `run_comparison.py`：训练和预检查；数据/标定输入 SHA256 记录在正式 `config.json`。
- `analyze_comparison.py`：实际结果汇总、收敛、传播及误差诊断。
- `local_slot_fields.py`：局部槽场图的真实求解器运行。

关键几何决策：原 shape 同时记录 500 μm 宽度、0.3 μm 槽周期、3 次重复和 1500 组槽长，参数不相容。本实验按物理组间距 0.9 μm 建立 555 组/层，覆盖 499.5 μm；3 层共 1665 个独立槽长。槽长严格在已标定的 0–2 μm 内。组内槽采用周期单元的有效薄屏响应，未逐槽解析非局部耦合。

训练准确率使用活动输出端口归一化概率，不表示高收光效率。两个波长分别训练不同槽长，尚未实现共享结构的双波长器件或粗细级联网络。细分类采用训练集拟合的独立九类编码，因此粗细任务间不能直接相减比较准确率；波长间的同任务编码完全相同。

在项目根目录复现：

```powershell
python comparison_1310_1610_scene3_20261016/run_comparison.py --tasks coarse3 fine9 --seeds 42 43 44
python comparison_1310_1610_scene3_20261016/local_slot_fields.py
python comparison_1310_1610_scene3_20261016/analyze_comparison.py
```

训练跳过已经完成的正式结果；局部场复用已保存的原始缓存。软件依赖：Python、PyTorch、NumPy、SciPy、scikit-learn、joblib、matplotlib。局部 varFDTD 额外需要已安装、可用的 Lumerical v241 MODE 许可证及 API。
