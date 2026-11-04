"""Run:  python test_pipeline.py     (or: pytest -q test_pipeline.py)"""
import numpy as np
from codec.quant import LUMA_BASE, CHROMA_BASE, scaled_table, quantize, dequantize
from codec.zigzag import zigzag_order, zigzag, inverse_zigzag
from codec.rle import (rle_encode_block, rle_decode_block, dc_to_diffs, diffs_to_dc,
                       category, EOB, ZRL)
from codec.subsample import subsample_420, upsample_420
from codec.pipeline import encode, decode
from codec.bitcost import estimate_bits
from codec.metrics import psnr
from codec.testimg import synthetic_image

RNG = np.random.default_rng(7)


# ------------------------------------------------------------ quantization
def test_quality_50_keeps_base_table():
    assert np.array_equal(scaled_table(LUMA_BASE, 50), LUMA_BASE)
    assert np.array_equal(scaled_table(CHROMA_BASE, 50), CHROMA_BASE)


def test_quality_100_is_all_ones():
    assert np.all(scaled_table(LUMA_BASE, 100) == 1)


def test_higher_quality_never_has_bigger_steps():
    prev = scaled_table(LUMA_BASE, 1)
    for q in range(2, 101):
        cur = scaled_table(LUMA_BASE, q)
        assert np.all(cur <= prev)
        prev = cur


def test_quantization_error_is_at_most_half_a_step():
    table = scaled_table(LUMA_BASE, 30)
    coefs = RNG.uniform(-500, 500, (50, 8, 8))
    err = np.abs(coefs - dequantize(quantize(coefs, table), table))
    assert np.all(err <= table / 2 + 1e-9)


# ------------------------------------------------------------ zigzag
def test_zigzag_starts_like_jpeg():
    assert zigzag_order(8)[:10].tolist() == [0, 1, 8, 16, 9, 2, 3, 10, 17, 24]


def test_zigzag_is_a_permutation():
    for n in (4, 8, 16):
        assert sorted(zigzag_order(n).tolist()) == list(range(n * n))


def test_zigzag_round_trip():
    blocks = RNG.integers(-50, 50, (10, 8, 8)).astype(np.int32)
    assert np.array_equal(inverse_zigzag(zigzag(blocks), 8), blocks)


# ------------------------------------------------------------ run-length
def test_category():
    assert [category(v) for v in (0, 1, -1, 2, 3, 4, 7, 8, -255)] == [0, 1, 1, 2, 2, 3, 3, 4, 8]


def test_all_zero_block_is_just_eob():
    assert rle_encode_block([0] * 63) == [EOB]


def test_long_gap_uses_zrl():
    ac = [0] * 20 + [5] + [0] * 42
    assert rle_encode_block(ac) == [ZRL, (4, 5), EOB]


def test_no_eob_when_last_coefficient_is_nonzero():
    ac = [0] * 62 + [3]
    symbols = rle_encode_block(ac)
    assert symbols == [ZRL, ZRL, ZRL, (14, 3)]      # 62 zeros = 16 + 16 + 16 + 14
    assert rle_decode_block(symbols) == ac


def test_rle_round_trip_random_sparse_blocks():
    for _ in range(300):
        ac = np.zeros(63, dtype=int)
        k = RNG.integers(0, 12)
        ac[RNG.integers(0, 63, k)] = RNG.integers(-40, 40, k)
        assert rle_decode_block(rle_encode_block(ac.tolist())) == ac.tolist()


def test_dc_prediction_round_trip():
    dc = RNG.integers(-300, 300, 100)
    assert np.array_equal(diffs_to_dc(dc_to_diffs(dc)), dc)


# ------------------------------------------------------------ subsampling
def test_subsample_shapes_with_odd_sizes():
    ch = RNG.uniform(-128, 127, (251, 253))
    small = subsample_420(ch)
    assert small.shape == (126, 127)
    assert upsample_420(small, 251, 253).shape == (251, 253)


def test_subsample_of_flat_channel_is_exact():
    ch = np.full((17, 21), 12.5)
    assert np.allclose(upsample_420(subsample_420(ch), 17, 21), ch)


# ------------------------------------------------------------ whole pipeline
def test_decode_keeps_shape_and_dtype():
    for h, w in ((250, 253), (64, 64), (9, 17)):
        img = synthetic_image(h, w)
        for sub in (False, True):
            out = decode(encode(img, quality=60, subsample=sub))
            assert out.shape == img.shape and out.dtype == np.uint8


def test_quality_100_is_nearly_lossless():
    img = synthetic_image()
    assert psnr(img, decode(encode(img, quality=100))) > 45


def test_psnr_and_size_increase_with_quality():
    img = synthetic_image()
    psnrs, sizes = [], []
    for q in (10, 30, 50, 70, 90):
        enc = encode(img, quality=q)
        psnrs.append(psnr(img, decode(enc)))
        sizes.append(estimate_bits(enc)["total_bits"])
    assert psnrs == sorted(psnrs)
    assert sizes == sorted(sizes)


def test_subsampling_saves_bits():
    img = synthetic_image()
    full = estimate_bits(encode(img, quality=50, subsample=False))["total_bits"]
    half = estimate_bits(encode(img, quality=50, subsample=True))["total_bits"]
    assert half < full


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\nall {len(tests)} tests passed")
