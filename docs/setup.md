# Setup

## Requirements

- Rust, installed through [rustup](https://rustup.rs). The version is pinned in
  `rust-toolchain.toml`, so rustup fetches the right one on first build. Do not
  install Rust any other way or you will silently build with a different
  compiler than everyone else.
- `curl` or `wget`, for fetching models.
- `bash`, for the scripts. macOS ships bash 3.2; the scripts are written for it.

Nothing else. No Python, no system BLAS, no CUDA toolkit, no C++ build of ONNX
Runtime.

Python is optional: it regenerates test data, exports BioCLIP and PE-Core,
fetches the NEON test set and runs the clustering analysis (see the
[README](../README.md)).

## First build

```sh
cd rust
cargo run --bin cli
```

Expected:

```
backend:   burn-flex
model:     none - run scripts/fetch-model.sh, then rebuild
```

If you see that, your toolchain works.

## Getting weights

```sh
./scripts/fetch-model.sh            # all models and test photos
./scripts/fetch-model.sh --help     # list them
./scripts/fetch-model.sh nomic-embed-text-v1_5 nomic-embed-vision-v1_5   # every file of these models
```

The Nomic models (about 550 MB and 370 MB) embed text and images into one
space. Try them with `cargo run --bin cli -- compare <image file or text>...`,
which prints the cosine similarity of every pair.

Files land in `models/` and are gitignored. Each download is checksummed. A
model whose checksum is recorded as `-` is not pinned yet; the script prints the
hash it computed so you can paste it in.

Set `NAHPU_MODEL_DIR` to keep weights somewhere else, for example on an external
drive:

```sh
NAHPU_MODEL_DIR=/Volumes/work/models ./scripts/fetch-model.sh
cd rust && NAHPU_MODEL_DIR=/Volumes/work/models cargo build
```

## Reading text

```sh
./scripts/fetch-model.sh PP-OCRv6_small_det PP-OCRv6_small_rec   # about 10 MB and 21 MB
cd rust && cargo run --bin cli -- ocr <image file>
```

The second prints how many lines of text it found, then the lines in reading
order: columns one after the other, a form row by row. Without any it prints
"no text detected". It reads JPEG, PNG, WebP, TIFF, BMP and GIF, turned upright
by their EXIF orientation.

Known limits:

- No HEIC and no PDFs; convert them to one of the formats above first.
- Recognition runs one line at a time, on the CPU by default. In a debug build
  on a 20-core laptop, including loading the models, a 6-line label takes 1 to
  4 s, a 20-line page 3 to 12 s and a 1400 x 2100 herbarium sheet 5 to 19 s. The
  low figures are with the machine otherwise idle.
- No text-direction classifier: a tall line is turned a quarter
  counter-clockwise, so text that reads bottom to top comes out upside down.
  Handwriting is only as good as the small model.
- The hard NEON photos mostly fail. `scripts/fetch-model.sh` puts the four
  named here in `models/`. On beetle labels turned sideways and partly hidden
  by the specimen (`ocr02_beetle_pasimachus_sublaevis_label_sideways.jpg`,
  `ocr03_beetle_pasimachus_marginatus_label_occluded.jpg`) only the upright
  number label and a few scattered words come out. A number inked on a skull is
  nearly read when upright (`hard01_mammal_napaeozapus_skull_ink_number.jpg`:
  "NEON" and "p1047" for a slashed-zero 01047), not at all when tiny and upside
  down (`hard02_mammal_sorex_rohweri_skull_ink_number.jpg`).

## Troubleshooting

**`no ONNX at ... building without a model`**

Not an error. The build skips model codegen when weights are missing so a fresh
clone still compiles. Run `scripts/fetch-model.sh` and rebuild.

**GPU feature does not build**

Check you picked the one for your platform: `metal` on macOS, `vulkan` on Linux
and Windows. `wgpu` picks a backend at runtime and is the fallback when you are
not sure.

**Debug runs feel far too slow**

They should not any more. Dependencies build optimised even in debug
(`[profile.dev.package."*"]` in `rust/Cargo.toml`) because unoptimised tensor
kernels ran 47x slower here. If you removed that, put it back.
