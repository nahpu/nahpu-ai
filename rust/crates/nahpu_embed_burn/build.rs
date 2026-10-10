use std::path::{Path, PathBuf};

use burn_onnx::{LoadStrategy, ModelGen};

struct OnnxModel {
    stem: &'static str,
    cfg: &'static str,
    weights_env: &'static str,
}

const MODELS: [OnnxModel; 5] = [
    OnnxModel {
        stem: "all-MiniLM-L6-v2",
        cfg: "has_minilm",
        weights_env: "MINILM_WEIGHTS",
    },
    OnnxModel {
        stem: "jina-clip-v1-text",
        cfg: "has_jina_text",
        weights_env: "JINA_TEXT_WEIGHTS",
    },
    OnnxModel {
        stem: "jina-clip-v1-vision",
        cfg: "has_jina_vision",
        weights_env: "JINA_VISION_WEIGHTS",
    },
    OnnxModel {
        stem: "bioclip-text",
        cfg: "has_bioclip_text",
        weights_env: "BIOCLIP_TEXT_WEIGHTS",
    },
    OnnxModel {
        stem: "bioclip-vision",
        cfg: "has_bioclip_vision",
        weights_env: "BIOCLIP_VISION_WEIGHTS",
    },
];

fn main() {
    println!("cargo::rerun-if-env-changed=NAHPU_MODEL_DIR");

    let models = models_dir();
    println!("cargo::rustc-env=NAHPU_MODELS_DIR={}", models.display());
    watch_for_models_appearing(&models);

    for model in &MODELS {
        println!("cargo::rustc-check-cfg=cfg({})", model.cfg);

        let onnx = models.join(format!("{}.onnx", model.stem));
        println!("cargo::rerun-if-changed={}", onnx.display());
        if !onnx.exists() {
            warn_model_missing(&onnx);
            continue;
        }

        generate_without_baked_paths(&onnx);
        println!("cargo::rustc-env={}={}", model.weights_env, generated_weights(model.stem).display());
        println!("cargo::rustc-cfg={}", model.cfg);
    }
}

fn generate_without_baked_paths(onnx: &Path) {
    ModelGen::new()
        .input(onnx.to_str().expect("model path is not valid UTF-8"))
        .out_dir("model/")
        .load_strategy(LoadStrategy::Bytes)
        .run_from_script();
}

fn generated_weights(stem: &str) -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"))
        .join("model")
        .join(format!("{stem}.bpk"))
}

fn models_dir() -> PathBuf {
    if let Ok(overridden) = std::env::var("NAHPU_MODEL_DIR") {
        return PathBuf::from(overridden);
    }
    let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../models");
    fallback.canonicalize().unwrap_or(fallback)
}

/// Watches the directory as well as the files: cargo cannot reliably notice a
/// path that does not exist yet, and the first fetch creates both.
fn watch_for_models_appearing(models: &Path) {
    let parent = models.parent().unwrap_or(models);
    println!("cargo::rerun-if-changed={}", parent.display());
    println!("cargo::rerun-if-changed={}", models.display());
}

fn warn_model_missing(onnx: &Path) {
    println!(
        "cargo::warning=no ONNX at {} — building without it. \
         Run scripts/fetch-model.sh to enable it.",
        onnx.display()
    );
}
