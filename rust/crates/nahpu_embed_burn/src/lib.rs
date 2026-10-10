use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::Bytes;

mod bioclip;
mod clip;
mod jina;
mod minilm;

pub use bioclip::{load_bioclip_text, load_bioclip_vision, BIOCLIP_LOGIT_SCALE};
pub use clip::{load_clip, ClipModel, CLIP_MODEL_NAMES};
pub use jina::{load_jina_text, load_jina_vision, JINA_LOGIT_SCALE};
pub use minilm::load_minilm;

#[cfg(all(
    feature = "flex",
    not(any(feature = "metal", feature = "vulkan", feature = "wgpu"))
))]
pub type SelectedBackend = burn::backend::Flex;
#[cfg(feature = "metal")]
pub type SelectedBackend = burn::backend::Metal;
#[cfg(all(feature = "vulkan", not(feature = "metal")))]
pub type SelectedBackend = burn::backend::Vulkan;
#[cfg(all(feature = "wgpu", not(any(feature = "metal", feature = "vulkan"))))]
pub type SelectedBackend = burn::backend::Wgpu;

pub const BACKEND_NAME: &str = if cfg!(feature = "metal") {
    "burn-metal"
} else if cfg!(feature = "vulkan") {
    "burn-vulkan"
} else if cfg!(feature = "wgpu") {
    "burn-wgpu"
} else {
    "burn-flex"
};

pub const HAS_MINILM: bool = cfg!(has_minilm);
pub const HAS_JINA: bool = cfg!(all(has_jina_text, has_jina_vision));
pub const HAS_BIOCLIP: bool = cfg!(all(has_bioclip_text, has_bioclip_vision));

pub fn models_dir() -> &'static Path {
    Path::new(env!("NAHPU_MODELS_DIR"))
}

fn tokenizer_path(stem: &str) -> PathBuf {
    models_dir().join(format!("{stem}.tokenizer.json"))
}

fn read_tokenizer(stem: &str) -> Result<tokenizers::Tokenizer> {
    let path = tokenizer_path(stem);
    tokenizers::Tokenizer::from_file(&path)
        .map_err(|e| anyhow::anyhow!("could not read tokenizer {}: {e}", path.display()))
}

/// Moves the file into Burn's buffer without a second copy of the weights.
fn read_weights(path: &str) -> Result<Bytes> {
    let weights = std::fs::read(path).with_context(|| format!("could not read weights {path}"))?;
    Ok(Bytes::from_elems(weights))
}

fn rows_of(output: Tensor<SelectedBackend, 2>) -> Vec<Vec<f32>> {
    let [_, width] = output.dims();
    output
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("output tensor is f32")
        .chunks(width)
        .map(<[f32]>::to_vec)
        .collect()
}

#[allow(dead_code)]
fn missing_model(stem: &str) -> anyhow::Error {
    anyhow::anyhow!("{stem} is not compiled in — run scripts/fetch-model.sh, then rebuild")
}
