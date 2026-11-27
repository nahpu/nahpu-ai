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

PE-Core-L is exported the same way and passed as `--model pe-core-l14-336`. The
export downloads a 2.7 GB checkpoint and writes two ONNX files of about 1.3 GB
each:

```sh
python3 -m venv scripts/pe/.venv
scripts/pe/.venv/bin/pip install -r scripts/pe/requirements.txt
scripts/pe/.venv/bin/python -I scripts/pe/export_onnx.py
```

The other four sizes of PE-Core are exported by name (see [PE-Core](#pe-core)).

`--prompts` picks the prompt templates and `--wording` picks common names, taxon
names or both. Add `--features metal` on a Mac for about 20x faster images.

## Evaluation

The NEON test set is 700 specimen photos, 100 for each of 7 classes. The
manifest is committed. The images are not, so fetch them once:

```sh
python3 -m venv scripts/neon/.venv
scripts/neon/.venv/bin/pip install -r scripts/neon/requirements.txt
scripts/neon/.venv/bin/python -I scripts/neon/fetch_dataset.py
```

On Windows, here, for the BioCLIP and PE-Core exports above and for the scripts
below, run `python` instead of `python3` and use `.venv/Scripts/` instead of
`.venv/bin/`.

Then score one model on every setting (framing, prompts, wording):

```sh
cd rust && cargo run --release --features metal --bin cli -- eval --sweep \
    --model bioclip --labels ../eval/neon-labels.csv \
    --manifest ../data/neon/manifest.csv --out ../eval/results/bioclip
```

This takes about a minute on Metal. Add `--limit 20` for a quick check. Each run
writes `report.md`, `metrics.json` and the predictions to `--out`. To check how
much picking the best setting flatters the score, run the sweep, then
`python3 scripts/reference/held_out.py data/neon/manifest.csv eval/results/bioclip`
from the repo root.

Nomic v1.5 is scored the same way, with `--model nomic-embed-v1.5`. Nomic
publishes no logit scale, so the `p` that `classify` prints for it is not
calibrated. The ranking does not depend on it.

Current results: [PE-Core-L](eval/results/pe-core-l14-336/report.md) 94.6%,
[BioCLIP](eval/results/bioclip/report.md) 86.9%,
[Nomic v1.5](eval/results/nomic-embed-v1.5/report.md) 74.0%,
[jina-clip-v1](eval/results/jina-clip-v1/report.md) 66.1%. The other sizes of
PE-Core are [below](#pe-core).

### Clustering

This checks how well the image vectors group photos with no text at all. First
save the vectors at each model's best setting:

```sh
cd rust && cargo run --release --features metal --bin cli -- embed \
    --model bioclip --prompts specimen --labels ../eval/neon-labels.csv \
    --manifest ../data/neon/manifest.csv --out ../eval/embeddings/bioclip
```

For jina, use `--model jina-clip-v1 --prompts ensemble`. For Nomic, use
`--model nomic-embed-v1.5 --prompts photo --wording both`. Then, from the repo
root:

```sh
python3 -m venv scripts/analysis/.venv
scripts/analysis/.venv/bin/pip install -r scripts/analysis/requirements.txt
scripts/analysis/.venv/bin/python -I scripts/analysis/check_embeddings.py eval/embeddings/bioclip eval/results/bioclip
scripts/analysis/.venv/bin/python -I scripts/analysis/cluster.py data/neon/manifest.csv \
    eval/results/clustering eval/embeddings/bioclip eval/embeddings/jina-clip-v1 \
    eval/embeddings/nomic-embed-v1.5
```

`check_embeddings.py` scores zero-shot from the saved files. It must match
`cli eval`, which shows each row belongs to the right photo.

Current results: [clustering report](eval/results/clustering/report.md). With
one nearest labelled photo, BioCLIP gets 97.9%, Nomic 90.3% and jina 85.6%.
All three beat their zero-shot scores. The report also covers the five sizes of
PE-Core: their folders in `eval/embeddings/` go on the end of the last command.

### OCR

There is no transcribed test set for OCR yet. This compares the text read on
four NEON label photos with what PaddleOCR read on the same photos:

```sh
cd rust && cargo run --bin cli -- ocr-eval \
    --references crates/nahpu_ocr_burn/tests/data/neon \
    --photos ../models --out ../eval/results/pp-ocrv6-small
```

`scripts/fetch-model.sh` fetches the photos into `models/`. The report lists
what was read on each photo and the lines that differ from PaddleOCR's.

Current results: [OCR report](eval/results/pp-ocrv6-small/report.md). The
character error rate against PaddleOCR is 1.0% to 1.5% on the four photos.

### PE-Core

[PE-Core](https://huggingface.co/timm/PE-Core-L-14-336) is Meta's Perception
Encoder: general-purpose models with a text and an image tower, in five sizes,
all Apache-2.0. The names are `pe-core-t16-384`, `pe-core-s16-384`,
`pe-core-b16-224`, `pe-core-l14-336` and `pe-core-g14-448`. Export a size by
name, then rebuild:

```sh
scripts/pe/.venv/bin/python -I scripts/pe/export_onnx.py pe-core-b16-224
```

Then use the `eval` and `embed` commands above with `--model pe-core-b16-224`.
For `embed`, the best setting is `--prompts specimen --wording both` for T, S
and B, `--prompts specimen` for L and the default for G. Use a GPU backend: on
Vulkan with an RTX 4070, T to L take 60 to 190 ms a photo.

G is exported in half precision. In single precision its image tower is 7.6 GB,
more than an 8 GB card holds. In half precision it is 3.8 GB and takes about
540 ms a photo on the same GPU. Layer normalisation stays in single precision,
because in half precision Burn returns NaN for it.

`scripts/pe/eval_pytorch.py` scores a size in PyTorch, to check a Burn run
against. It writes the same files as `cli eval --sweep` and `cli embed`. Score
the size into a folder ending in `-pytorch`, which git ignores, then compare the
two runs prediction by prediction:

```sh
scripts/pe/.venv/bin/python -I scripts/pe/eval_pytorch.py pe-core-l14-336 \
    data/neon/manifest.csv eval/neon-labels.csv \
    eval/results/pe-core-l14-336-pytorch eval/embeddings/pe-core-l14-336-pytorch
python3 scripts/pe/compare_with_burn.py eval/results/pe-core-l14-336 \
    eval/results/pe-core-l14-336-pytorch
```

For every size the two agree on 97.3% or more of predictions in every setting.
G runs in half precision in PyTorch too, on a CUDA GPU with 8 GB.
`pip` installs a CPU build of PyTorch on Windows; for a CUDA one, add
`--extra-index-url https://download.pytorch.org/whl/cu126 "torch==2.14.1+cu126"`
to the `pip install` above.

Current results, on the 700 NEON photos with centre-crop. Zero-shot is each
model's best prompt setting. The other two columns use no text and are from the
[clustering report](eval/results/clustering/report.md). Genus within a class is
the mean of the report's seven per-class figures:

| Model | Run in | Checkpoint | Zero-shot | Nearest labelled photo | Genus within a class |
|---|---|---|---|---|---|
| [PE-Core-T](eval/results/pe-core-t16-384/report.md) | Burn | 0.28 GB | 71.0% | 92.0% | 59.6% |
| [PE-Core-S](eval/results/pe-core-s16-384/report.md) | Burn | 0.35 GB | 78.9% | 94.3% | 57.4% |
| [PE-Core-B](eval/results/pe-core-b16-224/report.md) | Burn | 1.8 GB | 88.7% | 95.4% | 61.1% |
| [PE-Core-L](eval/results/pe-core-l14-336/report.md) | Burn | 2.7 GB | 94.6% | 99.1% | 70.5% |
| [PE-Core-G](eval/results/pe-core-g14-448/report.md) | Burn, half precision | 9.7 GB | 97.4% | 99.1% | 71.4% |
| BioCLIP | Burn | 0.6 GB | 86.9% | 97.9% | 69.4% |
| Nomic v1.5 | Burn | 0.9 GB | 74.0% | 90.3% | 48.8% |
| jina-clip-v1 | Burn | 0.9 GB | 66.1% | 85.6% | 52.4% |

L is the smallest size that beats BioCLIP on every column. B has a vision tower
the size of BioCLIP's and is ahead of it on zero-shot only.

## Crates

| Crate | What |
|---|---|
| `nahpu_embed_core` | Traits and helpers. No framework dependency. |
| `nahpu_embed_burn` | Burn engines. Backend picked by cargo feature. |
| `nahpu_burn_support` | What the Burn crates share: the device for the chosen backend, weight loading. |
| `nahpu_ocr_burn` | PP-OCRv6 text reading on Burn: the lines in an image, in reading order. |
| `cli` | Scratch binary for trying things by hand. |

`nahpu_embed_core` is the extension point. Add an engine by implementing
`TextEmbedder` or `ImageEmbedder`; nothing else has to change.

## Models

| Model | What |
|---|---|
| `all-MiniLM-L6-v2` | Text, 384 dimensions. |
| `jina-clip-v1` | Text and images in one space, 768 dimensions. Zero-shot classification. |
| `bioclip` | Text and images in one space, 512 dimensions. Exported, not fetched (above). |
| `pe-core-*` | Text and images in one space, in five sizes of 512 to 1280 dimensions. Exported, not fetched (above). |
| `nomic-embed-text-v1_5` | Text, 768 dimensions, in the same space as the vision model. |
| `nomic-embed-vision-v1_5` | Images, 768 dimensions, so a text query finds an image. |
| `PP-OCRv6_small_det` | Finds the text lines in an image. |
| `PP-OCRv6_small_rec` | Reads each line; its `.yml` holds the character dictionary. |

`./scripts/fetch-model.sh <model>` fetches one model. With no argument it
fetches every model and the test photos.

Compare text and images by hand with Nomic. This scores one query against two
of the test photos:

```sh
cd rust && cargo run --bin cli -- compare "a photo of a beetle" \
    ../models/emb01_beetle_chlaenius_aestivus_dorsal.jpg \
    ../models/emb10_mammal_peromyscus_ventral.jpg
```

It prints the cosine of every pair. The query scores highest with the beetle
photo.

Read the text in an image, in reading order:

```sh
cd rust && cargo run --bin cli -- ocr crates/nahpu_ocr_burn/tests/data/form.png
```

Formats and known limits are in [Setup](docs/setup.md#reading-text).

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
