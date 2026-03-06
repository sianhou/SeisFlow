import torch


def generate_row_mask(x, missing_ratio=0.3):
    """
    x: (B, C, H, W) or (..., H, W)
    missing_ratio: 缺失行比例
    """
    H = x.shape[-2]
    B = x.shape[0]

    mask = torch.ones_like(x)

    num_missing = int(H * missing_ratio)

    for b in range(B):
        start = torch.randint(0, H - num_missing + 1, (1,))
        mask[b, :, start:start + num_missing, :] = 0

    return mask


def generate_random_row_mask(x, missing_ratio=0.3):
    B, _, H, W = x.shape
    num_missing = int(H * missing_ratio)

    mask = torch.ones(B, 1, H, W, device=x.device)

    for b in range(B):
        rows = torch.randperm(H, device=x.device)[:num_missing]
        mask[b, :, rows, :] = 0

    return mask
