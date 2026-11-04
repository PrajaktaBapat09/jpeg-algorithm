"""RGB <-> YCbCr (JFIF / BT.601 full-range), done in float so nothing is lost."""
import numpy as np

# Rows produce Y, Cb, Cr from (R, G, B).
_FWD = np.array([
    [ 0.299000,  0.587000,  0.114000],
    [-0.168736, -0.331264,  0.500000],
    [ 0.500000, -0.418688, -0.081312],
])
_INV = np.linalg.inv(_FWD)          # exact numerical inverse -> clean round trip
_OFFSET = np.array([0.0, 128.0, 128.0])


def rgb_to_ycbcr(rgb):
    """rgb: (H, W, 3) array in [0, 255]. Returns float64 (H, W, 3) YCbCr."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return rgb @ _FWD.T + _OFFSET


def ycbcr_to_rgb(ycc):
    """ycc: (H, W, 3) float YCbCr. Returns float64 RGB (not yet rounded/clipped)."""
    ycc = np.asarray(ycc, dtype=np.float64)
    return (ycc - _OFFSET) @ _INV.T
