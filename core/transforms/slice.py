import torch


class SliceLastDimension:
    """
    只截取最后一维：x[..., start:end]
    - start/end 支持负数
    - end=None 表示到结尾
    """

    def __init__(self, start=0, end=None):
        self.start = start
        self.end = end

    def __call__(self, x: torch.Tensor):
        return x[..., self.start:self.end]
