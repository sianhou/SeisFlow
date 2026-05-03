# 地震 VAE 测试计划

## 1. 目标

Work Package 1 的目标是训练一个地震数据专用的 VAE / Autoencoder，用于单通道 `256x256` 地震 patch 的空间压缩表示学习。

目标映射关系为：

```text
输入 patch:    1 x 256 x 256
空间 latent:   C x N x N
重建输出:      1 x 256 x 256
```

训练得到的 latent 表示将用于后续 latent-space reconstruction，例如 flow matching 或 DiT-style reconstruction。

主要目标：

- 保留反射层连续性；
- 保留断层、边界和局部几何结构；
- 保留弱反射和有用纹理；
- 降低后续 latent-space reconstruction 的建模成本；
- 比较不同 latent 空间大小和通道容量对重建质量的影响。

## 2. 当前模型设计

相关代码文件：

```text
models/seismic_vae.py
train_seismic_vae.py
```

### 2.1 输入与输出

模型输入为归一化后的地震 patch：

```text
x: [B, 1, 256, 256]
```

Encoder 输出一个空间高斯后验分布：

```text
mu:     [B, C, N, N]
logvar: [B, C, N, N]
```

VAE 训练时使用重参数化采样：

```text
z = mu + eps * exp(0.5 * logvar)
```

Decoder 重建输出：

```text
recon: [B, 1, 256, 256]
```

后续 latent-space reconstruction 阶段，建议优先使用确定性 latent：

```text
z = mu
```

### 2.2 网络结构

Encoder 使用残差卷积块 `ResBlock`：

```text
Conv2d -> GroupNorm -> SiLU -> Conv2d -> GroupNorm
+ shortcut
-> SiLU
```

下采样通过 `stride=2` 的残差卷积块完成。

Decoder 使用：

```text
Conv2d -> UpBlock -> ... -> Tanh
```

其中 `UpBlock` 包含：

```text
ConvTranspose2d -> GroupNorm -> SiLU -> ResBlock
```

最后一层 `Tanh` 将输出限制在 `[-1, 1]`，因此输入 patch 也建议归一化到 `[-1, 1]` 附近。

### 2.3 默认 baseline

推荐默认配置为：

```text
input_size = 256
latent_channels = 4
latent_size = 32
hidden_channels = 32
```

对应结构为：

```text
1 x 256 x 256 -> 4 x 32 x 32 -> 1 x 256 x 256
```

该配置在空间压缩倍率上接近 Stable Diffusion VAE 的常见思路，同时使用地震数据专用的 encoder 和 decoder。

## 3. 数据集构建方案

Patch 数据集通过以下脚本构建：

```text
build_patch_dataset2.py
```

推荐命令：

```bash
python build_patch_dataset2.py \
  --segy ma2+GathAP.sgy \
  --slice 0 1501 \
  --resize 512 512 \
  --patch_size 256 \
  --overlap_size 128 \
  --normalize \
  --output_dir dataset_train_size256
```

参数说明：

- `--slice 0 1501`：截取输入数据的指定范围；
- `--resize 512 512`：将原始地震图像统一缩放到 `512x512`；
- `--patch_size 256`：提取 `256x256` patch；
- `--overlap_size 128`：使用 50% overlap；
- `--normalize`：对每个 patch 进行 max-abs 归一化。

推荐归一化方式：

```text
x_norm = x / max(abs(x))
```

该方式可以保持零振幅仍然位于 0，同时保留地震振幅的正负对称性。相比 min-max normalization，更适合地震振幅数据。

注意：不要在数据构建和训练时重复开启归一化。推荐在构建数据集时使用 `--normalize`，训练时不再额外开启 `--normalize`。

## 4. Loss 设计

训练脚本支持以下 loss 组合：

```text
total_loss =
  l1_weight * L1
+ mse_weight * MSE
+ kl_weight * KL
+ gradient_weight * GradientLoss
+ frequency_weight * FrequencyLoss
```

推荐初始配置：

```bash
--l1_weight 1.0
--mse_weight 0.0
--kl_weight 1e-6
--gradient_weight 0.1
--frequency_weight 0.0
```

各项 loss 的作用：

- `L1`：主要重建损失；
- `MSE`：可选重建损失，用于强调较大误差；
- `KL`：弱 VAE 正则项，避免 latent 完全无约束；
- `GradientLoss`：约束反射层边缘和局部结构连续性；
- `FrequencyLoss`：可选频域损失，用于高频一致性测试。

## 5. 实验计划

实验分为两组。

第一组为主实验，固定 `latent_channels=4`，比较不同空间 latent size。

第二组为容量补充实验，将 `latent_channels` 增加到 8，测试 `C=4` 是否存在容量不足。

## 5.1 第一组主实验

### 实验 1：C=4, N=64

训练命令：

```bash
python train_seismic_vae.py \
  --data_dir dataset_train_size256 \
  --input_size 256 \
  --latent_channels 4 \
  --latent_size 64 \
  --hidden_channels 32 \
  --output_dir output_seismic_vae \
  --log_id vae_c4_n64
```

Latent 形状：

```text
4 x 64 x 64
```

空间压缩倍率：

```text
256 / 64 = 4
```

预期表现：

- 重建质量最好；
- 弱反射、断层边界和细节保留最好；
- latent 空间最大，后续 flow matching / DiT 训练成本最高。

实验目的：

```text
作为高质量重建上限参考。
```

### 实验 2：C=4, N=32

训练命令：

```bash
python train_seismic_vae.py \
  --data_dir dataset_train_size256 \
  --input_size 256 \
  --latent_channels 4 \
  --latent_size 32 \
  --hidden_channels 32 \
  --output_dir output_seismic_vae \
  --log_id vae_c4_n32
```

Latent 形状：

```text
4 x 32 x 32
```

空间压缩倍率：

```text
256 / 32 = 8
```

预期表现：

- 在重建质量和 latent 建模成本之间取得较好平衡；
- 与 Stable Diffusion VAE 常见压缩比例接近；
- 推荐作为默认 baseline。

实验目的：

```text
主 baseline 实验。
```

### 实验 3：C=4, N=16

训练命令：

```bash
python train_seismic_vae.py \
  --data_dir dataset_train_size256 \
  --input_size 256 \
  --latent_channels 4 \
  --latent_size 16 \
  --hidden_channels 32 \
  --output_dir output_seismic_vae \
  --log_id vae_c4_n16
```

Latent 形状：

```text
4 x 16 x 16
```

空间压缩倍率：

```text
256 / 16 = 16
```

预期表现：

- latent 空间成本最低；
- 可能损失弱反射和高频细节；
- 可用于测试强压缩条件下结构是否仍可接受。

实验目的：

```text
强压缩边界测试。
```

## 5.2 第二组容量补充实验

### 实验 4：C=8, N=32

训练命令：

```bash
python train_seismic_vae.py \
  --data_dir dataset_train_size256 \
  --input_size 256 \
  --latent_channels 8 \
  --latent_size 32 \
  --hidden_channels 32 \
  --output_dir output_seismic_vae \
  --log_id vae_c8_n32
```

Latent 形状：

```text
8 x 32 x 32
```

实验目的：

测试在 `N=32` 时，将 latent channel 从 `4` 增加到 `8` 是否能够明显改善：

- PSNR；
- SSIM；
- 反射层连续性；
- 断层边缘清晰度；
- 高频一致性。

预期表现：

- 重建质量可能优于 `C=4, N=32`；
- 后续 latent reconstruction 的通道成本约增加一倍。

### 实验 5：C=8, N=16

训练命令：

```bash
python train_seismic_vae.py \
  --data_dir dataset_train_size256 \
  --input_size 256 \
  --latent_channels 8 \
  --latent_size 16 \
  --hidden_channels 32 \
  --output_dir output_seismic_vae \
  --log_id vae_c8_n16
```

Latent 形状：

```text
8 x 16 x 16
```

实验目的：

测试增加通道容量是否能够弥补更强空间压缩带来的信息损失。

预期表现：

- 可能优于 `C=4, N=16`；
- 空间成本低于 `C=4, N=32`；
- 如果质量接近 `C=4, N=32`，可能成为更高效的 latent 表示。

## 6. 日志记录

训练使用 `SimpleLogger2`。每个实验会在 `output_dir` 下创建独立 run 目录。

主要日志文件：

```text
log.txt
```

`log.txt` 中所有信息、事件和训练/验证指标写在同一个文件：

- `[I]`：参数、环境信息、事件；事件行会在 `[I]` 后标注 `info`、`warning`、`error` 等级别；
- `[H]`：指标表头；
- `[T]`：训练 step 指标；
- `[V]`：验证指标；
- 其他一字符行头可用于扩展。

重点查看 `[T]` 行：

  - total loss；
  - MSE loss；
  - KL loss；
  - MAE；
  - RMSE；
  - PSNR；
  - learning rate；
  - grad norm；
  - CUDA memory usage。

## 7. 评估指标

每个实验建议统一评估以下指标：

```text
MAE
MSE
RMSE
PSNR
SSIM
MS-SSIM
Gradient L1
Spectrum L1
High-frequency L1
```

当前训练日志已经记录：

```text
MAE
MSE
RMSE
PSNR
各项 loss
latent 统计
样本级重建统计
参数与梯度诊断
```

后续建议新增 `eval_seismic_vae.py`，用于对 checkpoint 进行统一评估、可视化和表格汇总。

## 8. 结果表模板

| 实验 | Latent | 压缩倍率 | 参数量 | MAE | PSNR | SSIM | Spectrum L1 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| E1 | 4x64x64 | 4x | - | - | - | - | - | 高质量上限 |
| E2 | 4x32x32 | 8x | - | - | - | - | - | 推荐 baseline |
| E3 | 4x16x16 | 16x | - | - | - | - | - | 强压缩 |
| E4 | 8x32x32 | 8x | - | - | - | - | - | 容量增强 |
| E5 | 8x16x16 | 16x | - | - | - | - | - | 容量/压缩折中 |

## 9. 选择标准

优先选择满足以下条件的 latent 配置：

- PSNR / SSIM 接近 `C=4, N=64`；
- 弱反射没有明显被洗掉；
- 断层边缘没有明显变钝；
- 高频细节没有过度平滑；
- latent 尺寸对后续 flow matching / DiT 训练成本可接受；
- latent 统计稳定，没有 NaN、Inf 或严重 saturation。

预期初始推荐：

```text
latent_channels = 4
latent_size = 32
```

如果重建质量不足，优先尝试：

```text
latent_channels = 8
latent_size = 32
```

如果 `C=8, N=16` 的质量接近 `C=4, N=32`，则可以考虑将其作为更低空间成本的 latent 表示，用于后续重建阶段。
