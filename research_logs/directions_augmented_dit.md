# 研究方向：AugmentedDiT 架构对照

- 级别：下一步重点
- 最后更新：2026-08-31

## 研究目的

- 验证 PixelDiT 中的 `AugmentedDiTBlock` 脱离 pixel-level `PiTBlock` 分支后，能否作为独立的二维 flow-matching 主干完成地震 patch 重建。
- 在输入尺寸、内部 patch size、宽度、深度、attention head、数据、优化器和采样流程尽量一致的条件下，对比 AugmentedDiT 与传统 Diffusers DiT 的收敛、最终精度和计算成本。
- 区分 PixelDiT 的收益或退化来自 augmented patch backbone，还是来自后续 pixel-level 分支。

## 当前结论

- 当前判断：AugmentedDiT 的模型、wrapper、独立训练/采样入口、hwgpu 训练/重建脚本和定向测试已经完成，可以启动 T/P4、i64、NeRF bands0 的端到端对照；尚无训练结果，不能判断其效果优于或劣于传统 DiT。
- 主要证据：新模型已通过矩形前向、反向传播、条件通道拼接、REPA 中间特征、checkpoint 配置恢复和 `max_period` 行为测试；相关 AugmentedDiT、PixelDiT、DiT 训练入口定向测试共25项通过，两个 hwgpu shell 脚本通过 `bash -n`。
- 局限或尚未确认的问题：当前对照匹配了 T/P4 的宽度384、深度8、6个 attention heads 和64×64输入下256个 token，但不是严格等参数实验；按2维原始坐标、NeRF bands0和1个噪声通道计算，AugmentedDiT约21.83M参数，DiT约23.58M参数，前者少约7.4%。AugmentedDiT使用 `max_period=10`，而 Diffusers DiT 保留默认时间编码；RMSNorm、2D RoPE、QK Norm、无 QKV bias 和 gated FFN 也同时变化，因此端到端结果不能直接归因于某一个组件。

## 实验记录

### 2026-08-31：独立模型、训练入口与对照方案准备

- 相关月志：[2026-08-31](202608.md#2026-08-31)
- 目的：从 PixelDiT 中抽出 augmented patch backbone，构建可独立训练、保存和采样的二维模型，并准备与 DiT-T/P4 的 i64/bands0 对照。
- 方法与配置：`AugmentedDiT2DModel` 使用 patchify → `PatchTokenEmbedder` → 8层 `AugmentedDiTBlock` → 条件 patch 输出层 → fold；T 配置为 hidden size 384、6 heads、depth 8、patch size 4，时间编码 `max_period=10`。训练入口使用 `shot_dataset64`、NeRF bands0、batch size 32/process、2000 epochs、每100 epochs保存、upcast attention、默认 EMA（decay 0.999）、seed 0。
- 对照：`DiTSeisDimReconNeRF.py` 的 `DiT_T_4`，输入64、hidden size 384、6 heads、8层、patch size 4；数据、batch、epoch、优化器默认值、EMA、upcast attention和保存间隔相同。两个脚本均配置5个 worker节点加master，节点编号不同。
- 结果：实现与定向测试完成，尚未产生训练 checkpoint、验证指标或整炮重建结果。
- 观察：在当前3输入通道配置下，AugmentedDiT约21.83M参数，DiT约23.58M参数；模型尺度接近但不完全等参数。当前 Trainer 的 flow-matching 时间为连续 `t∈[0,1]`，`max_period=10` 能让更多时间编码维度在该区间产生明显变化。
- 解释：现阶段只能说明实现链路和比较入口可用。若未来 AugmentedDiT 指标变化，应先视为整套 augmented block 设计的总体差异，不能在没有进一步消融时归因于 RoPE、RMSNorm、QK Norm 或 FFN 中的单个因素。

#### 版本与实现

- Git：`2ff25af`（dirty）；核心模型、wrapper、独立入口和 hwgpu 脚本由 `b068c99` 引入，当前训练参数更新到 `2ff25af`。
- 未提交相关文件：`tests/test_augmented_dit.py`、`tests/test_augmented_dit_trainer.py`、`research_logs/202608.md`、`research_logs/directions_augmented_dit.md`、`research_logs/directions_board.md`。
- 入口脚本：`scripts/hwgpu/train_AugmentedDiTSeisDimReconNeRF_t_i64_NerfBands0.sh`、`scripts/hwgpu/recon_AugmentedDiTSeisDimReconNeRF_t_i64_NerfBands0.sh`；对照为 `scripts/hwgpu/train_DiTSeisDimReconNeRF_t_p4_i64_NeRFBands0.sh` 和对应 recon 脚本。
- 主要代码：`models/pixeldit.py`、`models/wrapper.py`、`AugmentedDiTSeisDimReconNeRF.py`、`core/trainer.py`、`core/sampler.py`。
- 测试代码：`tests/test_augmented_dit.py`、`tests/test_augmented_dit_trainer.py`；同时回归 `tests/test_pixeldit.py`、`tests/test_pixeldit_trainer.py` 和 `tests/test_dit_trainer.py`。
- 关键配置：AugmentedDiT-T/P4、i64、NeRF bands0、`max_period=10`、upcast attention、batch size 32/process、2000 epochs、EMA decay 0.999、seed 0、solver step size 0.05、clip `[-1,1]`。
- Checkpoint：尚未生成。
- 结果目录：训练根目录计划为 `$PROJ_DIR/train_AugmentedDiTSeisDimReconNeRF_t_i64_NerfBands0/`，重建根目录计划为 `$PROJ_DIR/recon_AugmentedDiTSeisDimReconNeRF_t_i64_NerfBands0/`；具体运行子目录待训练后记录。

## 测试设计

### 阶段1：实现和 smoke test

- 验证方形与矩形输入的 patchify/fold 输出 shape，确保输入高宽可被 patch size 整除。
- 验证 concat conditioning 后的通道数、默认类别标签、REPA 指定层特征和投影 shape。
- 验证 `max_period` 改变时间 embedding，且通过 Diffusers config 保存和恢复。
- 验证不兼容的 RoPE head dimension 能在构造期明确报错。
- 验证训练 parser 将 `model_arch`、patch size、`max_period` 和 upcast attention 传入 builder，并正确分发 train/sample。

### 阶段2：端到端 T/P4 对照

| 项目 | AugmentedDiT | Diffusers DiT |
|---|---:|---:|
| 输入 | 64×64 | 64×64 |
| 内部 patch size | 4 | 4 |
| hidden size | 384 | 384 |
| depth | 8 | 8 |
| heads | 6 | 6 |
| token 数 | 256 | 256 |
| NeRF bands | 0 | 0 |
| batch/process | 32 | 32 |
| epochs | 2000 | 2000 |
| upcast attention | 开启 | 开启 |
| EMA | 默认开启 | 默认开启 |

- 使用相同训练/验证炮、相同 patch 数据、相同随机种子、优化器、学习率、梯度裁剪和采样步长。
- 每100 epoch比较验证集 MSE、PSNR和整炮重建，记录 EMA checkpoint 的收敛趋势。
- 同时记录参数量、峰值显存、samples/s、optimizer step 数、wall-clock和 GPU-hours；由于两个脚本使用不同物理节点，速度比较前需确认GPU型号和负载一致。

### 阶段3：必要时做归因消融

- 若端到端结果存在稳定差异，先统一时间编码设置或 timestep scale，再判断 `max_period` 是否为主要混杂因素。
- 根据结果决定是否进行等参数宽度调整，或逐项控制 RMSNorm/LayerNorm、RoPE/绝对位置编码、QK Norm 和 gated FFN。
- 只有在至少3个随机种子或稳定的多 checkpoint 趋势下，才将差异归因于架构而非训练波动。

## 下一步

1. 启动 AugmentedDiT-T/P4 i64 bands0 训练，并确认首个 batch、显存、loss、EMA 更新和 checkpoint 保存正常。
2. 按每100 epoch运行标准 patch 推理、整炮重建和差值生成，与 DiT-T/P4 使用同一75炮验证集比较 PSNR、MSE和收敛曲线。
3. 记录两组实际节点型号、吞吐、GPU-hours、checkpoint和结果目录；若节点硬件不同，不直接比较 wall-clock。
4. 根据端到端结果决定是否补充 `max_period=10000` 或统一 timestep embedding 的严格 block 消融。
