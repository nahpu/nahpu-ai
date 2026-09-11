use std::path::{Path, PathBuf};

use burn_onnx::{LoadStrategy, ModelGen};

const BASELINE_MODEL: &str = "all-MiniLM-L6-v2.onnx";

fn main() {
    declare_has_model_cfg();

    let model = models_dir().join(BASELINE_MODEL);
    rebuild_when_model_changes(&model);

    if !model.exists() {
        warn_model_missing(&model);
        return;
    }

    generate_model_without_baked_paths(&model);
    enable_has_model_cfg();
}

fn generate_model_without_baked_paths(model: &Path) {
    ModelGen::new()
        .input(model.to_str().expect("model path is not valid UTF-8"))
        .out_dir("model/")
        .load_strategy(LoadStrategy::Bytes)
        .run_from_script();
}

fn models_dir() -> PathBuf {
    if let Ok(overridden) = std::env::var("NAHPU_MODEL_DIR") {
        return PathBuf::from(overridden);
    }
    let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../models");
    fallback.canonicalize().unwrap_or(fallback)
}

fn declare_has_model_cfg() {
    println!("cargo::rustc-check-cfg=cfg(has_model)");
}

fn enable_has_model_cfg() {
    println!("cargo::rustc-cfg=has_model");
}

fn rebuild_when_model_changes(model: &Path) {
    println!("cargo::rerun-if-changed={}", model.display());
}

fn warn_model_missing(model: &Path) {
    println!(
        "cargo::warning=no ONNX at {} — building without a model. \
         Run scripts/fetch-model.sh to enable inference.",
        model.display()
    );
}
