# nahpu-ai

Experimental repository for NAHPU's machine learning work. Scratch space: things
here are tried, measured, and either promoted into `nahpu_api` or written up and
dropped.

Deep learning runs on [Burn](https://burn.dev), chosen in the Summer 2026
three-way evaluation of Candle, Burn, and ONNX Runtime on macOS, Windows ARM64,
and Windows x86. Burn won on portability: NAHPU runs on hardware we do not
control.

## Layout

```
rust/       Cargo workspace - inference engines and shared crates
scripts/    fetch models, run experiments
models/     downloaded weights (gitignored)
docs/       decisions and guides
```

Python experiments (fine-tuning, ONNX export) get a `python/` folder when the
first one lands.

## Quick start

```sh
cd rust
cargo run --bin cli        # builds without any model, reports what is available
```

That is the check that your toolchain works. To actually run inference you need
weights:

```sh
./scripts/fetch-model.sh   # downloads into models/, verifies the checksum
cd rust && cargo run --bin cli
```

The build works without models on purpose, so a fresh clone compiles without a
several-hundred-megabyte download.

Rank labels for a photo with zero-shot classification (jina-clip-v1):

```sh
cd rust && cargo run --release --bin cli -- classify \
    --labels ../eval/neon-labels.csv ../models/fixture-neon-shrew.jpg
```

For BioCLIP, export it once (it has no official ONNX), then rebuild and pass
`--model bioclip`:

```sh
python3 -m venv scripts/bioclip/.venv
scripts/bioclip/.venv/bin/pip install -r scripts/bioclip/requirements.txt
scripts/bioclip/.venv/bin/python -I scripts/bioclip/export_onnx.py
```

`--prompts` picks the prompt templates and `--wording` picks common names, taxon
names or both. Add `--features metal` on a Mac for about 20x faster images.

## Crates

| Crate | What |
|---|---|
| `nahpu_embed_core` | Traits and helpers. No framework dependency. |
| `nahpu_embed_burn` | Burn engines. Backend picked by cargo feature. |
| `cli` | Scratch binary for trying things by hand. |

`nahpu_embed_core` is the extension point. Add an engine by implementing
`TextEmbedder` or `ImageEmbedder`; nothing else has to change.

## Backends

CPU is `burn-flex`, pure Rust with SIMD. It needs no system BLAS and no per-OS
build flags, which is the portability the framework choice rests on.

```sh
cargo build                      # burn-flex (default)
cargo build --features metal     # macOS GPU
cargo build --features vulkan    # Linux / Windows GPU
```

GPU backends are opt-in so a plain build stays light.

## Docs

- [Setup](docs/setup.md) - toolchain, first build, troubleshooting
