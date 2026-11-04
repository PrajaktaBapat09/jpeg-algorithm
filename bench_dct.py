"""Run:  python bench_dct.py
Shows how much faster the matrix form is than the textbook double sum."""
import time
import numpy as np
from codec.dct import dct2_naive, dct2_matrix, dct2_blocks, dct_matrix

rng = np.random.default_rng(0)


def timeit(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


N_SMALL = 200
small = rng.uniform(-128, 127, (N_SMALL, 8, 8))
c = dct_matrix(8)

t_naive = timeit(lambda: [dct2_naive(b) for b in small], repeat=1)
t_matrix = timeit(lambda: [dct2_matrix(b, c) for b in small])
t_batch = timeit(lambda: dct2_blocks(small))

print(f"{N_SMALL} blocks of 8x8")
print(f"  naive  O(N^4), python loops : {t_naive*1e3:9.2f} ms  ({t_naive/N_SMALL*1e6:8.1f} us/block)")
print(f"  matrix O(N^3), per block    : {t_matrix*1e3:9.2f} ms  ({t_matrix/N_SMALL*1e6:8.1f} us/block)")
print(f"  matrix batched              : {t_batch*1e3:9.2f} ms  ({t_batch/N_SMALL*1e6:8.1f} us/block)")
print(f"  speedup naive -> matrix     : {t_naive/t_matrix:8.0f}x")
print(f"  speedup naive -> batched    : {t_naive/t_batch:8.0f}x")

big = rng.uniform(-128, 127, (512 * 512 // 64 * 3, 8, 8))   # ~ a 512x512 RGB image
t_big = timeit(lambda: dct2_blocks(big))
print(f"\nbatched DCT on {len(big)} blocks (~512x512 RGB): {t_big*1e3:.2f} ms")
