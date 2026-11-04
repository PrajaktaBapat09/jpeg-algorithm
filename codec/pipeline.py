"""Full lossy encode / decode (symbol level). No bitstream or file format yet.

encode: RGB -> YCbCr -> (optional 4:2:0) -> 8x8 DCT -> quantize -> zigzag -> DC diffs + RLE symbols
decode: the same steps in reverse.
"""
from dataclasses import dataclass
import numpy as np
from .colorspace import rgb_to_ycbcr, ycbcr_to_rgb
from .blocks import pad_to_multiple, split_blocks, merge_blocks
from .dct import dct2_blocks, idct2_blocks
from .quant import LUMA_BASE, CHROMA_BASE, scaled_table, quantize, dequantize
from .zigzag import zigzag, inverse_zigzag
from .rle import rle_encode_block, rle_decode_block, dc_to_diffs, diffs_to_dc
from .subsample import subsample_420, upsample_420

N = 8


@dataclass
class ChannelData:
    shape: tuple          # real (h, w) of this channel
    padded_shape: tuple   # (h, w) after padding to a multiple of 8
    dc_diffs: np.ndarray  # one int per block
    ac_symbols: list      # per block: list of (run, value)


@dataclass
class Encoded:
    height: int
    width: int
    quality: int
    subsample: bool
    channels: list        # [Y, Cb, Cr] ChannelData


def _tables(quality):
    luma = scaled_table(LUMA_BASE, quality)
    chroma = scaled_table(CHROMA_BASE, quality)
    return [luma, chroma, chroma]


def _encode_channel(ch, table):
    padded = pad_to_multiple(ch, N)
    coefs = dct2_blocks(split_blocks(padded, N))
    zz = zigzag(quantize(coefs, table))
    dc_diffs = dc_to_diffs(zz[:, 0])
    ac_symbols = [rle_encode_block(row) for row in zz[:, 1:].tolist()]
    return ChannelData(ch.shape, padded.shape, dc_diffs, ac_symbols)


def _decode_channel(cd, table):
    zz = np.zeros((len(cd.ac_symbols), N * N), dtype=np.int32)
    zz[:, 0] = diffs_to_dc(cd.dc_diffs)
    for i, symbols in enumerate(cd.ac_symbols):
        zz[i, 1:] = rle_decode_block(symbols)
    blocks = idct2_blocks(dequantize(inverse_zigzag(zz, N), table))
    h, w = cd.shape
    return merge_blocks(blocks, *cd.padded_shape, N)[:h, :w]


def encode(img, quality=50, subsample=False):
    h, w = img.shape[:2]
    ycc = rgb_to_ycbcr(img) - 128.0
    chans = [ycc[..., 0], ycc[..., 1], ycc[..., 2]]
    if subsample:
        chans[1], chans[2] = subsample_420(chans[1]), subsample_420(chans[2])
    channels = [_encode_channel(c, t) for c, t in zip(chans, _tables(quality))]
    return Encoded(h, w, quality, subsample, channels)


def decode(enc):
    chans = [_decode_channel(cd, t) for cd, t in zip(enc.channels, _tables(enc.quality))]
    if enc.subsample:
        chans[1] = upsample_420(chans[1], enc.height, enc.width)
        chans[2] = upsample_420(chans[2], enc.height, enc.width)
    ycc = np.stack(chans, axis=-1) + 128.0
    return np.clip(np.rint(ycbcr_to_rgb(ycc)), 0, 255).astype(np.uint8)
