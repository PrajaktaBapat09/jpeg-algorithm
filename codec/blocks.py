"""Split a 2D channel into NxN blocks and merge them back."""
import numpy as np


def pad_to_multiple(channel, n=8):
    """Pad bottom/right by repeating edge pixels so H and W are multiples of n."""
    h, w = channel.shape
    ph, pw = (-h) % n, (-w) % n
    return np.pad(channel, ((0, ph), (0, pw)), mode="edge")


def split_blocks(channel, n=8):
    """(H, W) -> (num_blocks, n, n), blocks in row-major (raster) order."""
    h, w = channel.shape
    assert h % n == 0 and w % n == 0, "pad first with pad_to_multiple"
    return (channel.reshape(h // n, n, w // n, n)
                   .swapaxes(1, 2)
                   .reshape(-1, n, n))


def merge_blocks(blocks, h, w, n=8):
    """Inverse of split_blocks. h, w are the PADDED dimensions."""
    return (blocks.reshape(h // n, w // n, n, n)
                  .swapaxes(1, 2)
                  .reshape(h, w))
