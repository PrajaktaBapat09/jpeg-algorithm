"""4:2:0 chroma subsampling: keep colour at half resolution in each direction."""
import numpy as np


def subsample_420(ch):
    """Average every 2x2 group. Odd sizes are edge-padded first."""
    h, w = ch.shape
    padded = np.pad(ch, ((0, h % 2), (0, w % 2)), mode="edge")
    H, W = padded.shape
    return padded.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))


def upsample_420(ch, h, w):
    """Repeat each value into a 2x2 group, then crop to the original (h, w)."""
    return np.repeat(np.repeat(ch, 2, axis=0), 2, axis=1)[:h, :w]
