"""Quantization: the step where information is actually thrown away.

Each of the 64 DCT coefficients is divided by a table entry and rounded to an integer.
Big table entries = coarse rounding = more zeros = smaller file, lower quality.
Tables are the standard JPEG (Annex K) ones, scaled by a 1..100 quality setting.
"""
import numpy as np

LUMA_BASE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.int32)

CHROMA_BASE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.int32)


def scaled_table(base, quality):
    """Quality 1..100 -> table. 50 keeps the base table, 100 gives all ones (finest)."""
    quality = int(np.clip(quality, 1, 100))
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    table = np.floor((base * scale + 50.0) / 100.0)
    return np.clip(table, 1, 255).astype(np.int32)


def quantize(coefs, table):
    """coefs: (B, 8, 8) floats -> (B, 8, 8) ints. THE lossy step."""
    return np.rint(coefs / table).astype(np.int32)


def dequantize(q, table):
    """Multiply back. The rounding error from quantize() is not recoverable."""
    return q * table
