"""Zigzag scan: read an 8x8 block along anti-diagonals, low frequencies first.

After quantization the high-frequency corner is mostly zeros, so this ordering puts
the important numbers at the front and a long run of zeros at the back.
"""
import numpy as np


def zigzag_order(n=8):
    """Flat indices (row * n + col) in zigzag order. For n=8 starts 0, 1, 8, 16, 9, 2, ..."""
    order = []
    for s in range(2 * n - 1):                       # s = row + col (the anti-diagonal)
        rows = range(max(0, s - n + 1), min(s, n - 1) + 1)
        rows = rows if s % 2 == 1 else reversed(rows)
        order.extend(r * n + (s - r) for r in rows)
    return np.array(order)


def zigzag(blocks):
    """(B, n, n) -> (B, n*n) in zigzag order."""
    n = blocks.shape[-1]
    return blocks.reshape(len(blocks), -1)[:, zigzag_order(n)]


def inverse_zigzag(flat, n=8):
    """(B, n*n) zigzag order -> (B, n, n)."""
    out = np.zeros_like(flat)
    out[:, zigzag_order(n)] = flat
    return out.reshape(len(flat), n, n)
