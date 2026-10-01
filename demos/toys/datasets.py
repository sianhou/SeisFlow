from typing import Tuple

import numpy as np
import torch


def generate_moons(
        n_samples: int = 100,
        noise: float = 0.2,
        scale: float = 2.0,
        **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成双月数据，并将样本均值平移至 (0, 0)。

    Args:
        n_samples: 样本数量。
        noise: 均匀随机偏移的幅度上限；None 表示不加噪声。

    Returns:
        X: 坐标张量，形状为 (n_samples, 2)。
        y: 类别标签，形状为 (n_samples,)。
    """
    if n_samples < 1:
        raise ValueError("n_samples 必须大于或等于 1")

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    theta_out = np.linspace(0, np.pi, n_samples_out)
    theta_in = np.linspace(0, np.pi, n_samples_in)

    outer_circ_x = np.cos(theta_out)
    outer_circ_y = np.sin(theta_out)

    inner_circ_x = 1 - np.cos(theta_in)
    inner_circ_y = 0.5 - np.sin(theta_in)

    X = np.column_stack([
        np.concatenate([outer_circ_x, inner_circ_x]),
        np.concatenate([outer_circ_y, inner_circ_y]),
    ])

    y = np.concatenate([
        np.zeros(n_samples_out, dtype=np.int64),
        np.ones(n_samples_in, dtype=np.int64),
    ])

    # 保留原函数的噪声方式：每个点的两个坐标使用相同偏移
    if noise is not None:
        X += np.random.rand(n_samples, 1) * noise

    # 中心化，保持形状不变
    X -= X.mean(axis=0, keepdims=True)

    # 按比例缩放坐标，中心保持不变
    X *= scale

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
