import math

import numpy as np


def calculate_psnr(img1, img2, max_pixel=255.0):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        img1: First image (numpy array)
        img2: Second image (numpy array)
        max_pixel: Maximum possible pixel value (255 for 8-bit images)

    Returns:
        PSNR value in dB
    """
    # Convert to float to avoid overflow
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Calculate MSE (Mean Squared Error)
    mse = np.mean((img1 - img2) ** 2)

    # If MSE is 0, images are identical
    if mse == 0:
        return float('inf')

    # Calculate PSNR
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

    return psnr
