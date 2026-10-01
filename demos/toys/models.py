import torch


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        """构建直接拼接时间标量的原始 MLP。

        Args:
            dim: 输入坐标维度。
            out_dim: 输出维度；None 表示与 dim 相同。
            w: 隐藏层宽度。
            time_varying: 是否将时间标量拼接到输入，不执行时间归一化。
        """
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x, t=None):
        """根据坐标和可选的时间标量计算预测。

        Args:
            x: 形状为 (batch_size, dim) 的浮点张量。
            t: 与 x 位于同一设备的时间张量，形状为 (batch_size,)；
                启用时间条件时必须提供，否则忽略。

        Returns:
            形状为 (batch_size, out_dim) 的预测张量。
        """
        if self.time_varying:
            x = torch.cat([x, t[:, None]], dim=-1)
        return self.net(x)


class ResMLP(torch.nn.Module):
    def __init__(
        self,
        dim,
        out_dim=None,
        w=64,
        time_varying=False,
        num_blocks=2,
        time_emb_dim=32,
        num_steps=1000,
    ):
        """构建支持时间条件的残差 MLP。

        Args:
            dim: 输入坐标维度。
            out_dim: 输出维度；None 表示与 dim 相同。
            w: 隐藏层宽度。
            time_varying: 是否使用时间嵌入。
            num_blocks: 残差块数量，每个块包含一个线性层和 SiLU 激活。
                默认两个块，使空间主干的线性层数量与原始 MLP 相同。
            time_emb_dim: 正弦和余弦特征的总维度，取正偶数。
            num_steps: 扩散总步数，至少为 2；用于将整数时间索引
                0 至 num_steps - 1 归一化至 [0, 1]。
        """
        super().__init__()
        self.time_varying = time_varying
        self.num_steps = num_steps
        if out_dim is None:
            out_dim = dim

        self.input_proj = torch.nn.Linear(dim, w)
        if time_varying:
            self.register_buffer(
                "time_frequencies", torch.logspace(0, 2, time_emb_dim // 2)
            )
            self.time_mlp = torch.nn.Sequential(
                torch.nn.Linear(time_emb_dim + 1, w),
                torch.nn.SiLU(),
                torch.nn.Linear(w, w),
            )

        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(w, w),
                torch.nn.SiLU(),
            )
            for _ in range(num_blocks)
        ])
        self.output_proj = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x, t=None):
        """预测输入坐标对应的输出。

        Args:
            x: 形状为 (batch_size, dim) 的浮点张量。
            t: 与 x 位于同一设备的时间索引张量，形状为 (batch_size,)，
                范围为 0 至 num_steps - 1；启用时间条件时必须提供。
                不需在调用前归一化；禁用时间条件时忽略此参数。

        Returns:
            形状为 (batch_size, out_dim) 的预测张量。
        """
        hidden = self.input_proj(x)
        time_embedding = 0.0
        if self.time_varying:
            time_normalized = t.to(dtype=x.dtype)[:, None] / (self.num_steps - 1)
            angles = 2 * torch.pi * time_normalized * self.time_frequencies
            time_features = torch.cat(
                [time_normalized, angles.sin(), angles.cos()], dim=-1
            )
            time_embedding = self.time_mlp(time_features)

        for block in self.blocks:
            hidden = hidden + block(hidden + time_embedding)

        return self.output_proj(hidden)
