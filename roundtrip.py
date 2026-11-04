"""
Run:  python roundtrip.py [path/to/image.png]

Pipeline: RGB -> YCbCr -> level shift -> pad -> 8x8 blocks -> DCT -> (nothing) -> IDCT
          -> merge -> crop -> un-shift -> RGB -> round/clip -> compare with original.
Pillow is used ONLY to read/write files. All the maths is ours.
"""
import sys
import numpy as np
from codec.colorspace import rgb_to_ycbcr, ycbcr_to_rgb
from codec.blocks import pad_to_multiple, split_blocks, merge_blocks
from codec.dct import dct2_blocks, idct2_blocks
from codec.metrics import psnr
from codec.testimg import synthetic_image

N = 8


def load(path):
    if path is None:
        print("no image given -> using synthetic 250x253 test image")
        return synthetic_image()
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def forward(img):
    """RGB uint8 -> list of 3 coefficient arrays (B,8,8) + padded shape."""
    ycc = rgb_to_ycbcr(img) - 128.0                      # level shift to be centred on 0
    coefs = []
    for c in range(3):
        padded = pad_to_multiple(ycc[..., c], N)
        coefs.append(dct2_blocks(split_blocks(padded, N)))
    return coefs, padded.shape


def inverse(coefs, padded_shape, h, w):
    chans = []
    for co in coefs:
        rec = merge_blocks(idct2_blocks(co), *padded_shape, N)[:h, :w]
        chans.append(rec)
    return ycbcr_to_rgb(np.stack(chans, axis=-1) + 128.0)


def main():
    img = load(sys.argv[1] if len(sys.argv) > 1 else None)
    h, w = img.shape[:2]
    print(f"image: {w}x{h}")

    coefs, pshape = forward(img)
    print(f"padded to {pshape[1]}x{pshape[0]} -> {coefs[0].shape[0]} blocks per channel")

    rec_f = inverse(coefs, pshape, h, w)
    rec = np.clip(np.rint(rec_f), 0, 255).astype(np.uint8)
    print(f"[float pipeline]  max abs error = {np.max(np.abs(rec.astype(int) - img.astype(int)))},"
          f"  PSNR = {psnr(img, rec)} dB")

    # What if we store YCbCr as 8-bit integers (as real JPEG does)? Colour transform itself becomes lossy.
    ycc8 = np.clip(np.rint(rgb_to_ycbcr(img)), 0, 255)
    rec8 = np.clip(np.rint(ycbcr_to_rgb(ycc8)), 0, 255).astype(np.uint8)
    print(f"[8-bit YCbCr only] PSNR = {psnr(img, rec8):.2f} dB   (rounding YCbCr is already lossy)")

    # Energy compaction: where does the signal live in frequency?
    y_co = coefs[0]
    mean_abs = np.mean(np.abs(y_co), axis=0)
    energy = np.mean(y_co ** 2, axis=0)
    share = energy / energy.sum()
    print(f"\nY channel: DC alone holds {share[0,0]*100:.1f}% of coefficient energy;"
          f" top-left 3x3 holds {share[:3,:3].sum()*100:.1f}%")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        im0 = ax[0].imshow(np.log10(1 + mean_abs), cmap="magma")
        ax[0].set_title("mean |DCT coef| per position (log10)")
        ax[0].set_xlabel("horizontal freq v"); ax[0].set_ylabel("vertical freq u")
        fig.colorbar(im0, ax=ax[0])
        ax[1].imshow(img); ax[1].set_title("input")
        fig.tight_layout(); fig.savefig("energy_compaction.png", dpi=120)
        print("saved energy_compaction.png")
    except ImportError:
        print("matplotlib not installed; skipping plot")

    try:
        from PIL import Image
        Image.fromarray(rec).save("reconstructed.png")
        print("saved reconstructed.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
