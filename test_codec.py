"""Run:  python test_codec.py     (or: pytest -q test_codec.py)"""
import numpy as np
from codec.colorspace import rgb_to_ycbcr, ycbcr_to_rgb
from codec.blocks import pad_to_multiple, split_blocks, merge_blocks
from codec.dct import (dct1_naive, dct2_naive, idct2_naive, dct_matrix,
                       dct2_matrix, idct2_matrix, dct2_blocks, idct2_blocks)
from codec.metrics import psnr
from codec.testimg import synthetic_image

RNG = np.random.default_rng(42)


def rand_block(n=8):
    return RNG.uniform(-128, 127, (n, n))


def test_dct_matrix_is_orthonormal():
    for n in (4, 8, 16):
        c = dct_matrix(n)
        assert np.allclose(c @ c.T, np.eye(n), atol=1e-12)


def test_1d_naive_equals_matrix():
    x = RNG.uniform(-128, 127, 8)
    assert np.allclose(dct1_naive(x), dct_matrix(8) @ x, atol=1e-9)


def test_2d_naive_equals_matrix():
    for _ in range(5):
        b = rand_block()
        assert np.allclose(dct2_naive(b), dct2_matrix(b), atol=1e-8)


def test_inverse_naive_equals_matrix():
    y = rand_block()
    assert np.allclose(idct2_naive(y), idct2_matrix(y), atol=1e-8)


def test_round_trip_block():
    b = rand_block()
    assert np.allclose(idct2_matrix(dct2_matrix(b)), b, atol=1e-10)
    assert np.allclose(idct2_naive(dct2_naive(b)), b, atol=1e-8)


def test_parseval_energy_preserved():
    b = rand_block()
    assert np.isclose(np.sum(b ** 2), np.sum(dct2_matrix(b) ** 2))


def test_flat_block_has_only_dc():
    b = np.full((8, 8), 50.0)
    y = dct2_matrix(b)
    assert np.isclose(y[0, 0], 8 * 50.0)            # DC = N * mean for orthonormal 2D DCT
    y[0, 0] = 0
    assert np.allclose(y, 0, atol=1e-10)


def test_matches_scipy_if_available():
    try:
        from scipy.fft import dctn
    except ImportError:
        return
    b = rand_block()
    assert np.allclose(dct2_matrix(b), dctn(b, type=2, norm="ortho"), atol=1e-10)


def test_batched_equals_per_block():
    blocks = RNG.uniform(-128, 127, (20, 8, 8))
    got = dct2_blocks(blocks)
    for i in range(20):
        assert np.allclose(got[i], dct2_matrix(blocks[i]), atol=1e-10)
    assert np.allclose(idct2_blocks(got), blocks, atol=1e-10)


def test_colorspace_round_trip():
    img = synthetic_image()
    back = ycbcr_to_rgb(rgb_to_ycbcr(img))
    assert np.allclose(back, img, atol=1e-9)


def test_blocks_round_trip_with_padding():
    ch = RNG.uniform(0, 255, (250, 253))
    padded = pad_to_multiple(ch, 8)
    assert padded.shape == (256, 256)
    blocks = split_blocks(padded, 8)
    assert blocks.shape == (32 * 32, 8, 8)
    back = merge_blocks(blocks, *padded.shape, 8)[:250, :253]
    assert np.array_equal(back, ch)


def test_block_order_is_raster():
    ch = np.arange(16 * 16, dtype=float).reshape(16, 16)
    blocks = split_blocks(ch, 8)
    assert np.array_equal(blocks[1], ch[0:8, 8:16])   # 2nd block = top-right
    assert np.array_equal(blocks[2], ch[8:16, 0:8])   # 3rd block = bottom-left


def test_full_pipeline_is_lossless_in_float():
    img = synthetic_image()
    ycc = rgb_to_ycbcr(img) - 128.0
    h, w = img.shape[:2]
    chans = []
    for c in range(3):
        p = pad_to_multiple(ycc[..., c], 8)
        co = dct2_blocks(split_blocks(p, 8))
        rec = merge_blocks(idct2_blocks(co), *p.shape, 8)[:h, :w]
        chans.append(rec)
    out = np.clip(np.rint(ycbcr_to_rgb(np.stack(chans, -1) + 128.0)), 0, 255).astype(np.uint8)
    assert np.array_equal(out, img)
    assert psnr(img, out) == float("inf")


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\nall {len(tests)} tests passed")
