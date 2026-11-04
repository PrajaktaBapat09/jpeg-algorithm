import numpy as np


def mse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(a, b, peak=255.0):
    """Peak signal-to-noise ratio in dB. Returns inf for identical images."""
    m = mse(a, b)
    return float("inf") if m == 0 else 10.0 * np.log10(peak ** 2 / m)
