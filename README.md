# Image Codec From Scratch

A JPEG-style image compression pipeline implemented from scratch (colour transform,
8x8 block DCT, and later quantization and entropy coding).
Pillow is used only to read and write image files.

## Setup
```
pip install -r requirements.txt
```

## Run
```
python test_codec.py                 # 13 correctness tests
python bench_dct.py                  # naive vs matrix vs batched DCT timing
python roundtrip.py                  # full pipeline on a synthetic image
python roundtrip.py your_photo.png   # full pipeline on your own image
```

## Layout
```
codec/
  colorspace.py   RGB <-> YCbCr
  blocks.py       pad, split into 8x8 blocks, merge
  dct.py          DCT: naive, matrix form, batched
  metrics.py      MSE, PSNR
  testimg.py      synthetic test image
test_codec.py     tests
bench_dct.py      benchmark
roundtrip.py      end-to-end round trip and energy-compaction plot
```
