import torch


def get_nerf_conditioning_channels(dim_channels, args):
    multiplier = 2 * int(args.nerf_bands)
    if args.nerf_include_input:
        multiplier += 1
    return int(dim_channels) * multiplier


def encode_nerf_conditioning(conditioning, args):
    encoded = []
    if args.nerf_include_input:
        encoded.append(conditioning)

    for band in range(args.nerf_bands):
        freq = 2.0 ** band
        phase = freq * torch.pi * conditioning
        encoded.append(torch.sin(phase))
        encoded.append(torch.cos(phase))

    if not encoded:
        raise ValueError("NeRF conditioning is empty; remove --no-nerf_include_input or set --nerf_bands > 0.")
    return torch.cat(encoded, dim=1)
