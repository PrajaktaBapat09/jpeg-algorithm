"""
Run:  python inspect_block.py [image.png] [quality] [block_row] [block_col]

Follows ONE 8x8 block of the luma channel through every stage so you can see what
each step does to the numbers.
"""
import sys
import numpy as np
from codec.colorspace import rgb_to_ycbcr
from codec.blocks import pad_to_multiple, split_blocks
from codec.dct import dct2_matrix, idct2_matrix
from codec.quant import LUMA_BASE, scaled_table, quantize, dequantize
from codec.zigzag import zigzag
from codec.rle import rle_encode_block, rle_decode_block
from codec.testimg import synthetic_image

np.set_printoptions(linewidth=140, suppress=True)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else None
    rest = [a for a in sys.argv[1:] if a.isdigit()]
    quality = int(rest[0]) if len(rest) > 0 else 50
    br = int(rest[1]) if len(rest) > 1 else 5
    bc = int(rest[2]) if len(rest) > 2 else 5

    if path:
        from PIL import Image
        img = np.asarray(Image.open(path).convert("RGB"))
    else:
        img = synthetic_image()

    y = pad_to_multiple(rgb_to_ycbcr(img)[..., 0] - 128.0, 8)
    blocks_per_row = y.shape[1] // 8
    block = split_blocks(y, 8)[br * blocks_per_row + bc]
    table = scaled_table(LUMA_BASE, quality)

    coefs = dct2_matrix(block)
    q = quantize(coefs[None], table)[0]
    zz = zigzag(q[None])[0]
    symbols = rle_encode_block(zz[1:].tolist())
    back = idct2_matrix(dequantize(q, table))

    print(f"quality {quality}, block row {br}, col {bc}\n")
    print("1) original pixels (level-shifted by -128):\n", np.rint(block).astype(int))
    print("\n2) DCT coefficients:\n", np.rint(coefs).astype(int))
    print("\n3) quantization table:\n", table)
    print("\n4) after quantization (integers, mostly zeros):\n", q)
    print("\n5) zigzag order:\n", zz.tolist())
    print(f"\n6) non-zero values: {np.count_nonzero(q)} of 64")
    print("   DC value:", int(zz[0]), "(stored as a difference from the previous block)")
    print("   AC run-length symbols (run, value):", symbols)
    assert rle_decode_block(symbols) == zz[1:].tolist()
    print("\n7) reconstructed pixels (after dequantize + inverse DCT):\n", np.rint(back).astype(int))
    print("\n8) error (reconstructed - original):\n", np.rint(back - block).astype(int))
    print(f"\nmax abs pixel error in this block: {np.max(np.abs(back - block)):.1f}")


if __name__ == "__main__":
    main()
