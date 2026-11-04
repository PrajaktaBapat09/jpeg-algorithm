"""
Orthonormal DCT-II / DCT-III implemented three ways (fast factorized version comes later):

  1. dct1_naive / dct2_naive : direct formula, pure Python loops. O(N^2) in 1D, O(N^4) in 2D.
  2. dct_matrix + dct2_matrix: separable form  Y = C X C^T. O(N^3) per block.
  3. dct2_blocks / idct2_blocks: the matrix form applied to ALL blocks at once (batched).
"""
import math
import numpy as np


def _alpha(k, n):
    return math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)


# ---------------------------------------------------------------- 1. naive
def dct1_naive(x):
    """1D DCT-II straight from the definition: X[k] = a(k) * sum_n x[n] cos(pi(2n+1)k / 2N)."""
    n = len(x)
    out = [0.0] * n
    for k in range(n):
        s = 0.0
        for i in range(n):
            s += x[i] * math.cos(math.pi * (2 * i + 1) * k / (2 * n))
        out[k] = _alpha(k, n) * s
    return np.array(out)


def _cos_table(n):
    return [[math.cos(math.pi * (2 * i + 1) * k / (2 * n)) for i in range(n)] for k in range(n)]


def dct2_naive(block):
    """2D DCT-II straight from the double-sum definition. Four nested loops -> O(N^4)."""
    n = block.shape[0]
    t = _cos_table(n)
    out = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            s = 0.0
            for x in range(n):
                for y in range(n):
                    s += block[x, y] * t[u][x] * t[v][y]
            out[u, v] = _alpha(u, n) * _alpha(v, n) * s
    return out


def idct2_naive(coefs):
    """2D inverse (DCT-III) straight from the definition, O(N^4)."""
    n = coefs.shape[0]
    t = _cos_table(n)
    out = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            s = 0.0
            for u in range(n):
                for v in range(n):
                    s += _alpha(u, n) * _alpha(v, n) * coefs[u, v] * t[u][x] * t[v][y]
            out[x, y] = s
    return out


# ---------------------------------------------------------------- 2. matrix
def dct_matrix(n=8):
    """C[k, i] = a(k) cos(pi(2i+1)k / 2N). Orthonormal: C @ C.T == I."""
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    c = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    c[0, :] *= math.sqrt(1.0 / n)
    c[1:, :] *= math.sqrt(2.0 / n)
    return c


def dct2_matrix(block, c=None):
    """Separable 2D DCT of one block: Y = C X C^T."""
    c = dct_matrix(block.shape[0]) if c is None else c
    return c @ block @ c.T


def idct2_matrix(coefs, c=None):
    """Inverse: X = C^T Y C (C is orthonormal so its inverse is its transpose)."""
    c = dct_matrix(coefs.shape[0]) if c is None else c
    return c.T @ coefs @ c


# ---------------------------------------------------------------- 3. batched
def dct2_blocks(blocks):
    """blocks: (B, N, N) -> (B, N, N). One batched matmul pair for the whole image."""
    c = dct_matrix(blocks.shape[-1])
    return c @ blocks @ c.T


def idct2_blocks(coefs):
    c = dct_matrix(coefs.shape[-1])
    return c.T @ coefs @ c
