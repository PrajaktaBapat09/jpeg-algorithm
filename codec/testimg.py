"""Synthetic test image: smooth gradient + hard-edged shapes + stripes + mild noise.
Deliberately NOT a multiple of 8 in size so the padding path gets exercised."""
import numpy as np


def synthetic_image(h=250, w=253, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), dtype=np.float64)
    img[..., 0] = 255 * xx / (w - 1)                      # horizontal ramp (smooth)
    img[..., 1] = 255 * yy / (h - 1)                      # vertical ramp (smooth)
    img[..., 2] = 128 + 60 * np.sin(xx / 15.0)            # slow wave
    circle = (xx - w * 0.65) ** 2 + (yy - h * 0.4) ** 2 < (min(h, w) * 0.22) ** 2
    img[circle] = (240, 220, 40)                          # sharp edge
    img[int(h * 0.75):int(h * 0.9), int(w * 0.1):int(w * 0.5)] = (20, 20, 20)
    img[int(h * 0.78):int(h * 0.87):2, int(w * 0.1):int(w * 0.5)] = (230, 230, 230)  # fine stripes
    img += rng.normal(0, 3, img.shape)                    # sensor-ish noise
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)
