"""Run-length coding of the 63 AC coefficients of a block, plus DC prediction.

JPEG-style symbols:
  (run, value)  = `run` zeros followed by a non-zero `value`
  (15, 0)  ZRL  = sixteen zeros in a row (used when a gap is longer than 15)
  (0, 0)   EOB  = "end of block": everything left in this block is zero
"""
import numpy as np

EOB = (0, 0)
ZRL = (15, 0)


def category(v):
    """Number of bits needed for |v|: 0->0, 1->1, 2..3->2, 4..7->3, ... (JPEG 'size' category)."""
    return abs(int(v)).bit_length()


def rle_encode_block(ac):
    """ac: sequence of 63 ints (zigzag order, DC excluded) -> list of (run, value) symbols."""
    symbols = []
    run = 0
    for v in ac:
        if v == 0:
            run += 1
        else:
            while run > 15:                  # gap too long for one symbol
                symbols.append(ZRL)
                run -= 16
            symbols.append((run, int(v)))
            run = 0
    if run > 0:                              # trailing zeros -> single end-of-block marker
        symbols.append(EOB)
    return symbols


def rle_decode_block(symbols, n_ac=63):
    out = [0] * n_ac
    pos = 0
    for run, val in symbols:
        if (run, val) == EOB:
            break
        if (run, val) == ZRL:
            pos += 16
            continue
        pos += run
        out[pos] = val
        pos += 1
    return out


def dc_to_diffs(dc):
    """DC prediction: store each block's DC as the difference from the previous block's DC."""
    return np.diff(dc, prepend=0)


def diffs_to_dc(diffs):
    return np.cumsum(diffs)
