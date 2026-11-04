"""
Run:  python experiment_quality.py [image.png] [max_side]

Sweeps the quality setting and records, for our codec and for Pillow's JPEG:
  - bits per pixel  (our number is an entropy ESTIMATE until the Huffman coder exists)
  - PSNR
Saves rd_curve.png. Big photos are shrunk to max_side (default 512) so the pure-Python
run-length step stays quick. Pillow is used only as the comparison baseline.
"""
import io
import sys
import numpy as np
from codec.pipeline import encode, decode
from codec.bitcost import estimate_bits
from codec.metrics import psnr
from codec.testimg import synthetic_image

QUALITIES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]


def load(path, max_side):
    if path is None:
        print("no image given -> using synthetic test image (use a real photo for a real curve)")
        return synthetic_image()
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    return np.asarray(im)


def ours(img, q, subsample):
    enc = encode(img, quality=q, subsample=subsample)
    bpp = estimate_bits(enc)["total_bits"] / (img.shape[0] * img.shape[1])
    return bpp, psnr(img, decode(enc))


def pillow_jpeg(img, q, subsampling):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=q, subsampling=subsampling, optimize=True)
    dec = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return 8.0 * len(buf.getvalue()) / (img.shape[0] * img.shape[1]), psnr(img, dec)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    max_side = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    img = load(path, max_side)
    print(f"image {img.shape[1]}x{img.shape[0]}\n")

    curves = {"ours 4:4:4": [], "ours 4:2:0": [], "Pillow 4:4:4": [], "Pillow 4:2:0": []}
    print(f"{'q':>3} | {'ours444 bpp':>11} {'psnr':>6} | {'ours420 bpp':>11} {'psnr':>6} |"
          f" {'pil444 bpp':>10} {'psnr':>6} | {'pil420 bpp':>10} {'psnr':>6}")
    for q in QUALITIES:
        a, b = ours(img, q, False), ours(img, q, True)
        c, d = pillow_jpeg(img, q, 0), pillow_jpeg(img, q, 2)
        for name, pt in zip(curves, (a, b, c, d)):
            curves[name].append(pt)
        print(f"{q:>3} | {a[0]:>11.3f} {a[1]:>6.2f} | {b[0]:>11.3f} {b[1]:>6.2f} |"
              f" {c[0]:>10.3f} {c[1]:>6.2f} | {d[0]:>10.3f} {d[1]:>6.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        styles = {"ours 4:4:4": "o-", "ours 4:2:0": "s-", "Pillow 4:4:4": "o--", "Pillow 4:2:0": "s--"}
        for name, pts in curves.items():
            ax[0].plot([p[0] for p in pts], [p[1] for p in pts], styles[name], label=name, markersize=4)
            ax[1].plot(QUALITIES, [p[1] for p in pts], styles[name], label=name, markersize=4)
        ax[0].set_xlabel("bits per pixel"); ax[0].set_ylabel("PSNR (dB)")
        ax[0].set_title("rate vs distortion (higher-left is better)"); ax[0].grid(alpha=.3); ax[0].legend()
        ax[1].set_xlabel("quality setting"); ax[1].set_ylabel("PSNR (dB)")
        ax[1].set_title("quality setting vs PSNR"); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig("rd_curve.png", dpi=120)
        print("\nsaved rd_curve.png")
    except ImportError:
        print("\nmatplotlib not installed; skipping plot")


if __name__ == "__main__":
    main()
