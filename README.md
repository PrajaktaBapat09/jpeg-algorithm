# Image Codec From Scratch

A JPEG-style image compression pipeline implemented from scratch: colour transform,
8x8 block DCT, quantization, zigzag scan, run-length coding (Huffman coding and a
file format come next). Pillow is used only to read/write images and as the
comparison baseline.

## Setup
```
pip install -r requirements.txt
```

## Run
```
python test_codec.py                      # DCT, blocks, colour tests
python test_pipeline.py                   # quantization, zigzag, RLE, pipeline tests
python bench_dct.py                       # naive vs matrix vs batched DCT timing
python roundtrip.py [photo.png]           # lossless-in-float round trip + energy plot
python inspect_block.py [photo.png] 50    # follow one 8x8 block through every stage
python experiment_quality.py [photo.png]  # quality sweep, rate-distortion plot vs Pillow JPEG
```

## Layout
```
codec/
  colorspace.py   RGB <-> YCbCr
  blocks.py       pad, split into 8x8 blocks, merge
  dct.py          DCT: naive, matrix form, batched
  quant.py        quantization tables, quality scaling, quantize/dequantize
  zigzag.py       zigzag scan order
  rle.py          run-length symbols (ZRL / EOB), DC prediction
  subsample.py    4:2:0 chroma subsampling
  pipeline.py     encode / decode at symbol level
  bitcost.py      entropy-based size estimate (before real Huffman coding)
  metrics.py      MSE, PSNR
  testimg.py      synthetic test image
```
