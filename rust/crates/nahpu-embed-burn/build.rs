//! Build-time codegen: burn-onnx turns an ONNX export into a Burn model.
//!
//! ONNX files are large and gitignored, so a fresh clone has none. Rather than
//! failing the build we skip codegen and set `has_model`, letting the crate
//! compile without a multi-hundred-megabyte download. `scripts/fetch-model.sh`
//! fetches the weights when you actually want to run something.

use std::path::PathBuf;

use burn_onnx::{LoadStrategy, ModelGen};

/// Baseline model. Already validated in rust-inference-bench, so it is what a
/// new backend has to reproduce before a new model is worth trying.
const BASELINE: &str = "all-MiniLM-L6-v2.onnx";

fn main() {
    println!("cargo::rustc-check-cfg=cfg(has_model)");

    let onnx = models_dir().join(BASELINE);
    println!("cargo::rerun-if-changed={}", onnx.display());

    if !onnx.exists() {
        println!(
            "cargo::warning=no ONNX at {} — building without a model. \
             Run scripts/fetch-model.sh to enable inference.",
            onnx.display()
        );
        return;
    }

    // `Bytes` generates only `from_bytes()`, so no path is baked into the
    // generated source. Weights are handed in at runtime, which is what lets
    // the app download them on demand instead of shipping them.
    ModelGen::new()
        .input(onnx.to_str().expect("model path is not valid UTF-8"))
        .out_dir("model/")
        .load_strategy(LoadStrategy::Bytes)
        .run_from_script();

    println!("cargo::rustc-cfg=has_model");
}

/// Repo-level `models/`, overridable so CI and other machines can point
/// elsewhere without editing the build script.
fn models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("NAHPU_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../models")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from("../../../models"))
}
