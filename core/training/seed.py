import random

import numpy as np
import torch


def set_random_seed(seed: int = 0, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None and cudnn.is_available():
        cudnn.deterministic = deterministic
        cudnn.benchmark = not deterministic
