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
./scripts/fetch-model.sh            # all models
./scripts/fetch-model.sh --help     # list them
```

Files land in `models/` and are gitignored. Each download is checksummed. A
model whose checksum is recorded as `-` is not pinned yet; the script prints the
hash it computed so you can paste it in.

Set `NAHPU_MODEL_DIR` to keep weights somewhere else, for example on an external
drive:

```sh
NAHPU_MODEL_DIR=/Volumes/work/models ./scripts/fetch-model.sh
cd rust && NAHPU_MODEL_DIR=/Volumes/work/models cargo build
```

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
