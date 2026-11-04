"""Estimate the compressed size before a real entropy coder exists.

We assume an ideal code per symbol (cost = -log2 of its probability), the same four
symbol groups JPEG uses (DC/AC x luma/chroma), plus the raw "amplitude" bits that
describe each value. Table/header overhead is ignored. A real Huffman coder will land
a little above this number.
"""
import math
from collections import Counter
from .rle import category


def _entropy_bits(symbols):
    n = len(symbols)
    if n == 0:
        return 0.0
    return -sum(c * math.log2(c / n) for c in Counter(symbols).values())


def estimate_bits(enc):
    groups = {"dc_luma": [], "dc_chroma": [], "ac_luma": [], "ac_chroma": []}
    amplitude_bits = 0
    for idx, ch in enumerate(enc.channels):
        kind = "luma" if idx == 0 else "chroma"
        for d in ch.dc_diffs.tolist():
            c = category(d)
            groups["dc_" + kind].append(c)
            amplitude_bits += c
        for symbols in ch.ac_symbols:
            for run, val in symbols:
                c = category(val)
                groups["ac_" + kind].append((run, c))
                amplitude_bits += c
    symbol_bits = sum(_entropy_bits(g) for g in groups.values())
    total = symbol_bits + amplitude_bits
    return {"total_bits": total, "symbol_bits": symbol_bits,
            "amplitude_bits": amplitude_bits, "bytes": total / 8.0}
